from __future__ import annotations

import uuid
from pathlib import Path
import shutil

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import get_settings
from app.models.document import ChunkRecord


class QdrantService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: QdrantClient | None = None
        self._mode: str = "unknown"

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    @property
    def collection_name(self) -> str:
        return self.settings.qdrant_collection

    def _build_client(self) -> QdrantClient:
        try:
            remote = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                prefer_grpc=self.settings.qdrant_prefer_grpc,
                timeout=10,
            )
            remote.get_collections()
            self._mode = "remote"
            return remote
        except Exception:
            if not self.settings.qdrant_fallback_local:
                raise

        local_path = Path(self.settings.qdrant_local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        local = QdrantClient(path=str(local_path))
        self._mode = "local"
        return local

    def _reset_local_client(self) -> None:
        if self._mode != "local":
            return
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass

        local_path = Path(self.settings.qdrant_local_path)
        if local_path.exists():
            shutil.rmtree(local_path, ignore_errors=True)
        local_path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(local_path))

    def health(self) -> dict:
        return {
            "mode": self._mode,
            "url": self.settings.qdrant_url,
            "collection": self.collection_name,
        }

    def collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(item.name == self.collection_name for item in collections)

    def _collection_vector_size(self) -> int | None:
        if not self.collection_exists():
            return None
        info = self.client.get_collection(self.collection_name)
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, models.VectorParams):
            return vectors_config.size
        if isinstance(vectors_config, dict) and vectors_config:
            first_key = next(iter(vectors_config.keys()))
            first_vector = vectors_config[first_key]
            return first_vector.size
        return None

    def ensure_collection(self, vector_size: int) -> None:
        _ = self.client
        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            return

        existing_size = self._collection_vector_size()
        if existing_size is None:
            return

        if existing_size != vector_size:
            if self._mode == "local":
                self._reset_local_client()
            if self.collection_exists():
                self.client.delete_collection(collection_name=self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            return

    def clear_collection(self, vector_size: int) -> None:
        _ = self.client
        if self._mode == "local":
            self._reset_local_client()

        if self.collection_exists():
            self.client.delete_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def delete_by_doc_ids(self, doc_ids: list[str]) -> None:
        if not doc_ids or not self.collection_exists():
            return
        conditions = [models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)) for doc_id in doc_ids]
        selector = models.FilterSelector(filter=models.Filter(should=conditions))
        self.client.delete(collection_name=self.collection_name, points_selector=selector, wait=True)

    def upsert_chunks(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")

        points: list[models.PointStruct] = []
        for chunk, vec in zip(chunks, vectors):
            payload = {
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "section": chunk.section,
                "page": chunk.page,
                "metadata": chunk.metadata,
            }
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id))
            points.append(models.PointStruct(id=point_id, vector=vec, payload=payload))

        try:
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        except ValueError as exc:
            if "could not broadcast input array" not in str(exc):
                raise
            vector_size = len(vectors[0]) if vectors else 0
            if vector_size <= 0:
                raise
            if self._mode == "local":
                self._reset_local_client()
            self.clear_collection(vector_size=vector_size)
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def vector_search(self, query_vector: list[float], top_k: int) -> list[dict]:
        if not self.collection_exists():
            return []

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        hits: list[dict] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                {
                    "chunk_id": payload.get("chunk_id", str(point.id)),
                    "doc_id": payload.get("doc_id"),
                    "score": float(round(point.score, 6)),
                    "text": payload.get("text", ""),
                    "section": payload.get("section"),
                    "page": payload.get("page"),
                    "metadata": payload.get("metadata", {}),
                }
            )
        return hits


qdrant_service = QdrantService()
