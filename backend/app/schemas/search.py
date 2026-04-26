from pydantic import BaseModel, Field


class IndexBuildRequest(BaseModel):
    doc_ids: list[str] | None = None


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
