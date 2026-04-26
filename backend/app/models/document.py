from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    id: str
    filename: str
    file_path: str
    file_ext: str
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parse_status: str = "uploaded"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    id: str
    doc_id: str
    text: str
    chunk_index: int
    section: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str
    section: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
