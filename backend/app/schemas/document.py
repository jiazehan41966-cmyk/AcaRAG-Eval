from pydantic import BaseModel, Field


class ParseDocumentRequest(BaseModel):
    chunk_size: int = Field(default=600, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    run_async: bool = False


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    parse_status: str


class ParseDocumentResponse(BaseModel):
    doc_id: str
    parse_status: str
    chunks: int
    task_id: str | None = None
    task_backend: str | None = None


class DocumentMetadataResponse(BaseModel):
    doc_id: str
    metadata: dict
