from pydantic import BaseModel, Field


class ParseDocumentRequest(BaseModel):
    chunk_size: int = Field(default=600, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    parse_status: str


class ParseDocumentResponse(BaseModel):
    doc_id: str
    parse_status: str
    chunks: int


class DocumentMetadataResponse(BaseModel):
    doc_id: str
    metadata: dict
