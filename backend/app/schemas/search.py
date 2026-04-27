from pydantic import BaseModel, Field


class IndexBuildRequest(BaseModel):
    doc_ids: list[str] | None = None
    run_async: bool = False


class IndexBuildResponse(BaseModel):
    status: str
    indexed_chunks: int | None = None
    indexed_docs: int | None = None
    built_at: str | None = None
    embedding_backend: str | None = None
    reranker_backend: str | None = None
    qdrant_mode: str | None = None
    task_id: str | None = None
    task_backend: str | None = None


class IndexJobResponse(BaseModel):
    job_id: str
    status: str
    action: str | None = None
    backend: str | None = None
    celery_task_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    result: dict | None = None
    error: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    candidates: list[str] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    strategy: str
    hits: list[dict]
