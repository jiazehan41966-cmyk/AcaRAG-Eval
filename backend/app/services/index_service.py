from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone

from fastapi import HTTPException

from app.core.config import get_settings
from app.models.document import ChunkRecord
from app.services.bge_service import bge_model_service
from app.services.embedding_service import embedding_service
from app.services.graph_service import graph_service
from app.services.qdrant_service import qdrant_service
from app.services.storage_service import state_store


class IndexService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def tokenize(self, text: str) -> list[str]:
        return embedding_service.tokenize(text)

    def _load_index_payload(self) -> dict:
        return state_store.load_index()

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

    def build_index(self, doc_ids: list[str] | None = None) -> dict:
        existing = self._load_index_payload()
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
        if not self.settings.force_mock_embedding:
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
        graph_index = graph_service.build_graph_index(chunk_vectors)

        payload = {
            "chunk_vectors": chunk_vectors,
            "bm25": bm25,
            "graph_index": graph_index,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "embedding": bge_model_service.health(),
            "qdrant": qdrant_service.health()
            if not self.settings.force_mock_embedding
            else {"mode": "disabled_mock", "collection": qdrant_service.collection_name},
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
            state_store.save_index({"chunk_vectors": {}, "bm25": {}, "graph_index": {}})
        return self.build_index(doc_ids=doc_ids)

    def bm25_search(self, query: str, top_k: int = 5, index: dict | None = None) -> list[dict]:
        index = index if index is not None else self._load_index_payload()
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

    def vector_search(self, query: str, top_k: int = 5, index: dict | None = None) -> list[dict]:
        index = index if index is not None else self._load_index_payload()
        chunk_vectors = index.get("chunk_vectors") or {}
        if not chunk_vectors:
            raise HTTPException(status_code=400, detail="Index is empty. Build index first.")

        query_vector = bge_model_service.embed(query)

        # Primary path: real vector retrieval via Qdrant.
        if not self.settings.force_mock_embedding:
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

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.6, index: dict | None = None) -> list[dict]:
        index = index if index is not None else self._load_index_payload()
        vector_hits = self.vector_search(query, top_k=max(top_k * 3, 10), index=index)
        bm25_hits = self.bm25_search(query, top_k=max(top_k * 3, 10), index=index)

        vector_scores = {hit["chunk_id"]: hit["score"] for hit in vector_hits}
        bm25_scores = {hit["chunk_id"]: hit["score"] for hit in bm25_hits}

        all_chunk_ids = set(vector_scores) | set(bm25_scores)
        merged: list[tuple[str, float]] = []

        max_vector = max(vector_scores.values(), default=1.0) or 1.0
        max_bm25 = max(bm25_scores.values(), default=1.0) or 1.0

        chunk_vectors = index.get("chunk_vectors", {})

        for chunk_id in all_chunk_ids:
            if chunk_id not in chunk_vectors:
                continue
            v = vector_scores.get(chunk_id, 0.0) / max_vector
            b = bm25_scores.get(chunk_id, 0.0) / max_bm25
            merged_score = alpha * v + (1 - alpha) * b
            merged.append((chunk_id, merged_score))

        merged.sort(key=lambda x: x[1], reverse=True)
        return [self._to_hit(chunk_id, score, chunk_vectors[chunk_id]) for chunk_id, score in merged[:top_k]]

    def rerank(self, query: str, candidates: list[str], top_k: int = 5) -> list[dict]:
        return bge_model_service.rerank(query=query, documents=candidates, top_k=top_k)

    def graph_search(self, query: str, top_k: int = 5) -> list[dict]:
        index = self._load_index_payload()
        chunk_vectors = index.get("chunk_vectors") or {}
        if not chunk_vectors:
            raise HTTPException(status_code=400, detail="Index is empty. Build index first.")

        graph_index = index.get("graph_index") or {}
        if not graph_index or graph_index.get("version") != 2:
            graph_index = graph_service.build_graph_index(chunk_vectors)

        candidate_k = max(top_k * 3, 10)
        graph_hits = graph_service.graph_search(
            query=query,
            top_k=candidate_k,
            chunk_payloads=chunk_vectors,
            graph_index=graph_index,
        )
        hybrid_hits = self.hybrid_search(query=query, top_k=candidate_k, alpha=0.6, index=index)
        return self._merge_graph_hybrid_hits(graph_hits=graph_hits, hybrid_hits=hybrid_hits, top_k=top_k)

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

    @staticmethod
    def _merge_graph_hybrid_hits(graph_hits: list[dict], hybrid_hits: list[dict], top_k: int) -> list[dict]:
        max_graph = max([float(item.get("score", 0.0)) for item in graph_hits], default=1.0) or 1.0
        max_hybrid = max([float(item.get("score", 0.0)) for item in hybrid_hits], default=1.0) or 1.0
        merged: dict[str, dict] = {}

        for item in hybrid_hits:
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                continue
            merged[chunk_id] = {
                **item,
                "score": float(item.get("score", 0.0)) / max_hybrid,
                "metadata": {
                    **(item.get("metadata", {}) or {}),
                    "retrieval_sources": ["hybrid"],
                    "hybrid_score": float(item.get("score", 0.0)),
                    "graph_score": 0.0,
                    "matched_entities": [],
                },
            }

        for item in graph_hits:
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                continue
            graph_norm = float(item.get("score", 0.0)) / max_graph
            matched_entities = list((item.get("metadata") or {}).get("matched_entities", []))
            existing = merged.get(chunk_id, item)
            metadata = {
                **(existing.get("metadata", {}) or {}),
                **(item.get("metadata", {}) or {}),
            }
            sources = set(metadata.get("retrieval_sources", []))
            sources.add("graph")
            metadata["retrieval_sources"] = sorted(sources)
            metadata["graph_score"] = float(item.get("score", 0.0))
            metadata["matched_entities"] = sorted(set(metadata.get("matched_entities", [])) | set(matched_entities))
            hybrid_norm = float(existing.get("score", 0.0)) if chunk_id in merged else 0.0
            entity_bonus = min(len(metadata["matched_entities"]) * 0.03, 0.18)
            merged[chunk_id] = {
                **existing,
                **item,
                "metadata": metadata,
                "score": round(hybrid_norm + (0.35 * graph_norm) + entity_bonus, 6),
            }

        ranked = sorted(merged.values(), key=lambda x: x.get("score", 0.0), reverse=True)
        return ranked[:top_k]

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
