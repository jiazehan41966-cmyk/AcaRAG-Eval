from __future__ import annotations

from typing import Iterable

from app.core.config import get_settings
from app.services.embedding_service import embedding_service


class BGEModelService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._embedder = None
        self._reranker = None
        self._embedder_error: str | None = None
        self._reranker_error: str | None = None

    def _load_embedder(self) -> None:
        if self._embedder is not None or self._embedder_error is not None:
            return
        if self.settings.force_mock_embedding:
            self._embedder_error = "force_mock_embedding=true"
            return
        try:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(model_name=self.settings.embedding_model_name)
        except Exception as exc:  # pragma: no cover - import/runtime environment dependent
            self._embedder_error = str(exc)

    def _load_reranker(self) -> None:
        if self._reranker is not None or self._reranker_error is not None:
            return
        if self.settings.force_mock_embedding:
            self._reranker_error = "force_mock_embedding=true"
            return
        try:
            from fastembed.rerank.cross_encoder.onnx_text_cross_encoder import OnnxTextCrossEncoder

            self._reranker = OnnxTextCrossEncoder(model_name=self.settings.reranker_model_name)
        except Exception as exc:  # pragma: no cover - import/runtime environment dependent
            self._reranker_error = str(exc)

    def embed(self, text: str) -> list[float]:
        vectors = self.embed_batch([text])
        return vectors[0] if vectors else []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        self._load_embedder()
        if self._embedder is None:
            return [embedding_service.embed(text) for text in texts]

        vectors: list[list[float]] = []
        try:
            for vec in self._embedder.embed(texts):
                vectors.append(vec.tolist() if hasattr(vec, "tolist") else list(vec))
            if vectors:
                return vectors
        except Exception as exc:  # pragma: no cover - model runtime dependent
            self._embedder_error = str(exc)

        return [embedding_service.embed(text) for text in texts]

    def rerank(self, query: str, documents: Iterable[str], top_k: int | None = None) -> list[dict]:
        docs = list(documents)
        if not docs:
            return []

        self._load_reranker()
        scored: list[tuple[str, float]] = []

        if self._reranker is not None:
            try:
                scores = list(self._reranker.rerank(query, docs))
                scored = [(doc, float(score)) for doc, score in zip(docs, scores)]
            except Exception as exc:  # pragma: no cover - model runtime dependent
                self._reranker_error = str(exc)

        if not scored:
            query_tokens = set(embedding_service.tokenize(query))
            for doc in docs:
                doc_tokens = set(embedding_service.tokenize(doc))
                overlap = len(query_tokens & doc_tokens)
                union = len(query_tokens | doc_tokens) or 1
                scored.append((doc, overlap / union))

        scored.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            scored = scored[:top_k]

        return [
            {
                "text": text,
                "score": float(round(score, 6)),
                "rank": rank + 1,
            }
            for rank, (text, score) in enumerate(scored)
        ]

    def health(self) -> dict:
        embedding_backend = (
            "fastembed" if self._embedder is not None else "mock_hash" if self._embedder_error else "uninitialized"
        )
        reranker_backend = (
            "fastembed" if self._reranker is not None else "mock_overlap" if self._reranker_error else "uninitialized"
        )
        return {
            "embedding_backend": embedding_backend,
            "reranker_backend": reranker_backend,
            "embedding_model": self.settings.embedding_model_name,
            "reranker_model": self.settings.reranker_model_name,
            "embedding_error": self._embedder_error,
            "reranker_error": self._reranker_error,
        }


bge_model_service = BGEModelService()
