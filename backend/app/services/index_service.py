from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone

from fastapi import HTTPException

from app.models.document import ChunkRecord
from app.services.bge_service import bge_model_service
from app.services.embedding_service import embedding_service
from app.services.qdrant_service import qdrant_service
from app.services.storage_service import state_store


class IndexService:
    def tokenize(self, text: str) -> list[str]:
        return embedding_service.tokenize(text)

    def _build_bm25(self, chunk_payloads: dict[str, dict]) -> dict:
        inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        doc_freq: Counter[str] = Counter()
        doc_len: dict[str, int] = {}

        for chunk_id, payload in chunk_payloads.items():
            tokens = self.tokenize(payload["text"])
            tf = Counter(tokens)
            doc_len[chunk_id] = len(tokens)
            for term, term_freq in tf.items():
                inverted_index[term][chunk_id] = term_freq
                doc_freq[term] += 1

        avg_doc_len = (sum(doc_len.values()) / len(doc_len)) if doc_len else 0.0

        return {
            "inverted_index": dict(inverted_index),
            "doc_freq": dict(doc_freq),
            "doc_len": doc_len,
            "avg_doc_len": avg_doc_len,
            "doc_count": len(chunk_payloads),
            "k1": 1.5,
            "b": 0.75,
        }

    def _load_chunk_vectors(self) -> dict[str, dict]:
        return state_store.load_index().get("chunk_vectors", {})

    def build_index(self, doc_ids: list[str] | None = None) -> dict:
        existing = state_store.load_index()
        chunk_vectors: dict[str, dict] = existing.get("chunk_vectors", {})

        selected_chunks = state_store.list_chunks(doc_ids)
        selected_doc_ids = set(doc_ids) if doc_ids else None

        if not selected_chunks:
            raise HTTPException(status_code=400, detail="No parsed chunks found. Parse documents first.")

        if selected_doc_ids is None:
            chunk_vectors = {}
            # full rebuild: clear qdrant collection after vector dimension is known
        else:
            chunk_vectors = {
                cid: payload
                for cid, payload in chunk_vectors.items()
                if payload.get("doc_id") not in selected_doc_ids
            }
            qdrant_service.delete_by_doc_ids(list(selected_doc_ids))

        texts = [chunk.text for chunk in selected_chunks]
        vectors = bge_model_service.embed_batch(texts)
        if not vectors:
            raise HTTPException(status_code=500, detail="Failed to compute embeddings")

        vector_size = len(vectors[0])
        if selected_doc_ids is None:
            qdrant_service.clear_collection(vector_size)
        else:
            qdrant_service.ensure_collection(vector_size)

        qdrant_service.upsert_chunks(selected_chunks, vectors)

        for chunk, vector in zip(selected_chunks, vectors):
            chunk_vectors[chunk.id] = {
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "section": chunk.section,
                "page": chunk.page,
                "metadata": chunk.metadata,
                "vector": vector,
            }

        bm25 = self._build_bm25(chunk_vectors)

        payload = {
            "chunk_vectors": chunk_vectors,
            "bm25": bm25,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "embedding": bge_model_service.health(),
            "qdrant": qdrant_service.health(),
        }
        state_store.save_index(payload)

        return {
            "indexed_chunks": len(chunk_vectors),
            "indexed_docs": len({entry["doc_id"] for entry in chunk_vectors.values()}),
            "built_at": payload["built_at"],
            "embedding_backend": payload["embedding"]["embedding_backend"],
            "reranker_backend": payload["embedding"]["reranker_backend"],
            "qdrant_mode": payload["qdrant"]["mode"],
        }

    def rebuild_index(self, doc_ids: list[str] | None = None) -> dict:
        if doc_ids is None:
            state_store.save_index({"chunk_vectors": {}, "bm25": {}})
        return self.build_index(doc_ids=doc_ids)

    def bm25_search(self, query: str, top_k: int = 5) -> list[dict]:
        index = state_store.load_index()
        bm25 = index.get("bm25") or {}
        chunk_vectors = index.get("chunk_vectors") or {}
        if not bm25 or not chunk_vectors:
            raise HTTPException(status_code=400, detail="Index is empty. Build index first.")

        tokens = self.tokenize(query)
        if not tokens:
            return []

        doc_count = bm25.get("doc_count", 0)
        avg_doc_len = bm25.get("avg_doc_len", 1.0) or 1.0
        k1 = bm25.get("k1", 1.5)
        b = bm25.get("b", 0.75)
        inverted_index = bm25.get("inverted_index", {})
        doc_freq = bm25.get("doc_freq", {})
        doc_len = bm25.get("doc_len", {})

        scores: defaultdict[str, float] = defaultdict(float)

        for term in tokens:
            postings = inverted_index.get(term, {})
            if not postings:
                continue
            n_t = doc_freq.get(term, 0)
            idf = math.log(1 + (doc_count - n_t + 0.5) / (n_t + 0.5)) if doc_count else 0.0

            for chunk_id, tf in postings.items():
                length = doc_len.get(chunk_id, 0)
                denom = tf + k1 * (1 - b + b * (length / avg_doc_len))
                score = idf * ((tf * (k1 + 1)) / (denom or 1e-9))
                scores[chunk_id] += score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self._to_hit(chunk_id, score, chunk_vectors[chunk_id]) for chunk_id, score in ranked]

    def vector_search(self, query: str, top_k: int = 5) -> list[dict]:
        index = state_store.load_index()
        chunk_vectors = index.get("chunk_vectors") or {}
        if not chunk_vectors:
            raise HTTPException(status_code=400, detail="Index is empty. Build index first.")

        query_vector = bge_model_service.embed(query)

        # Primary path: real vector retrieval via Qdrant.
        try:
            hits = qdrant_service.vector_search(query_vector=query_vector, top_k=top_k)
            if hits:
                return hits
        except Exception:
            # Qdrant may fail when existing collection vector size mismatches current embedding dimension.
            # We fallback to local cosine retrieval to keep serving traffic.
            pass

        # Fallback path: local cosine retrieval.
        scores = []
        for chunk_id, payload in chunk_vectors.items():
            vector = payload.get("vector", [])
            if len(vector) != len(query_vector):
                continue
            score = embedding_service.cosine_similarity(query_vector, vector)
            scores.append((chunk_id, score, payload))

        scores.sort(key=lambda item: item[1], reverse=True)
        return [self._to_hit(chunk_id, score, payload) for chunk_id, score, payload in scores[:top_k]]

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.6) -> list[dict]:
        vector_hits = self.vector_search(query, top_k=max(top_k * 3, 10))
        bm25_hits = self.bm25_search(query, top_k=max(top_k * 3, 10))

        vector_scores = {hit["chunk_id"]: hit["score"] for hit in vector_hits}
        bm25_scores = {hit["chunk_id"]: hit["score"] for hit in bm25_hits}

        all_chunk_ids = set(vector_scores) | set(bm25_scores)
        merged: list[tuple[str, float]] = []

        max_vector = max(vector_scores.values(), default=1.0) or 1.0
        max_bm25 = max(bm25_scores.values(), default=1.0) or 1.0

        index = self._load_chunk_vectors()

        for chunk_id in all_chunk_ids:
            if chunk_id not in index:
                continue
            v = vector_scores.get(chunk_id, 0.0) / max_vector
            b = bm25_scores.get(chunk_id, 0.0) / max_bm25
            merged_score = alpha * v + (1 - alpha) * b
            merged.append((chunk_id, merged_score))

        merged.sort(key=lambda x: x[1], reverse=True)
        return [self._to_hit(chunk_id, score, index[chunk_id]) for chunk_id, score in merged[:top_k]]

    def rerank(self, query: str, candidates: list[str], top_k: int = 5) -> list[dict]:
        return bge_model_service.rerank(query=query, documents=candidates, top_k=top_k)

    def rerank_hits(self, query: str, hits: list[dict], top_k: int = 5) -> list[dict]:
        if not hits:
            return []
        ranked = bge_model_service.rerank(query=query, documents=[item["text"] for item in hits], top_k=top_k)
        score_by_text = {item["text"]: item["score"] for item in ranked}

        merged = []
        for hit in hits:
            if hit["text"] not in score_by_text:
                continue
            merged.append({**hit, "rerank_score": score_by_text[hit["text"]]})

        merged.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return merged[:top_k]

    def _to_hit(self, chunk_id: str, score: float, payload: dict) -> dict:
        return {
            "chunk_id": chunk_id,
            "doc_id": payload.get("doc_id"),
            "score": float(round(score, 6)),
            "text": payload.get("text", ""),
            "section": payload.get("section"),
            "page": payload.get("page"),
            "metadata": payload.get("metadata", {}),
        }


index_service = IndexService()
