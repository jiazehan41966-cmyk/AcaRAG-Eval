from fastapi import APIRouter, File, UploadFile

from app.schemas.document import (
    DocumentMetadataResponse,
    DocumentUploadResponse,
    ParseDocumentRequest,
    ParseDocumentResponse,
)
from app.services.document_service import document_service
from app.services.parser_service import parser_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    record = await document_service.upload_document(file)
    return DocumentUploadResponse(doc_id=record.id, filename=record.filename, parse_status=record.parse_status)


@router.get("/{doc_id}")
def get_document(doc_id: str):
    record = document_service.get_document(doc_id)
    return record.model_dump(mode="json")


@router.post("/{doc_id}/parse", response_model=ParseDocumentResponse)
def parse_document(doc_id: str, request: ParseDocumentRequest):
    record = document_service.get_document(doc_id)
    record, chunks = parser_service.parse_document(
        record,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    return ParseDocumentResponse(doc_id=record.id, parse_status=record.parse_status, chunks=len(chunks))


@router.get("/{doc_id}/chunks")
def get_document_chunks(doc_id: str):
    _ = document_service.get_document(doc_id)
    chunks = document_service.list_chunks(doc_id)
    return [chunk.model_dump(mode="json") for chunk in chunks]


@router.get("/{doc_id}/metadata", response_model=DocumentMetadataResponse)
def get_document_metadata(doc_id: str):
    record = document_service.get_document(doc_id)
    return DocumentMetadataResponse(doc_id=record.id, metadata=record.metadata)
