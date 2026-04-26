from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone

from fastapi import HTTPException

from app.services.embedding_service import embedding_service
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

    def build_index(self, doc_ids: list[str] | None = None) -> dict:
        existing = state_store.load_index()
        chunk_vectors: dict[str, dict] = existing.get("chunk_vectors", {})

        selected_chunks = state_store.list_chunks(doc_ids)
        selected_doc_ids = set(doc_ids) if doc_ids else None

        if selected_doc_ids is None:
            chunk_vectors = {}
        else:
            # Remove old vectors for documents that will be re-indexed.
            chunk_vectors = {
                cid: payload
                for cid, payload in chunk_vectors.items()
                if payload.get("doc_id") not in selected_doc_ids
            }

        if not selected_chunks:
            raise HTTPException(status_code=400, detail="No parsed chunks found. Parse documents first.")

        for chunk in selected_chunks:
            chunk_vectors[chunk.id] = {
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "section": chunk.section,
                "page": chunk.page,
                "metadata": chunk.metadata,
                "vector": embedding_service.embed(chunk.text),
            }

        bm25 = self._build_bm25(chunk_vectors)

        payload = {
            "chunk_vectors": chunk_vectors,
            "bm25": bm25,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        state_store.save_index(payload)

        return {
            "indexed_chunks": len(chunk_vectors),
            "indexed_docs": len({entry["doc_id"] for entry in chunk_vectors.values()}),
            "built_at": payload["built_at"],
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

        query_vec = embedding_service.embed(query)
        scores = []
        for chunk_id, payload in chunk_vectors.items():
            score = embedding_service.cosine_similarity(query_vec, payload.get("vector", []))
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

        index = state_store.load_index().get("chunk_vectors", {})

        for chunk_id in all_chunk_ids:
            v = vector_scores.get(chunk_id, 0.0) / max_vector
            b = bm25_scores.get(chunk_id, 0.0) / max_bm25
            merged_score = alpha * v + (1 - alpha) * b
            merged.append((chunk_id, merged_score))

        merged.sort(key=lambda x: x[1], reverse=True)
        return [self._to_hit(chunk_id, score, index[chunk_id]) for chunk_id, score in merged[:top_k]]

    def rerank(self, query: str, candidates: list[str], top_k: int = 5) -> list[dict]:
        query_tokens = set(self.tokenize(query))
        ranked: list[tuple[str, float]] = []

        for candidate in candidates:
            candidate_tokens = set(self.tokenize(candidate))
            overlap = len(query_tokens & candidate_tokens)
            union = len(query_tokens | candidate_tokens) or 1
            score = overlap / union
            ranked.append((candidate, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "text": text,
                "score": score,
                "rank": rank + 1,
            }
            for rank, (text, score) in enumerate(ranked[:top_k])
        ]

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
