from uuid import uuid4

from fastapi import APIRouter, File, UploadFile

from app.schemas.document import (
    DocumentMetadataResponse,
    DocumentUploadResponse,
    ParseDocumentRequest,
    ParseDocumentResponse,
)
from app.services.document_service import document_service
from app.services.parser_service import parser_service
from app.services.storage_service import state_store
from app.tasks.parse_tasks import enqueue_parse_job

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    record = await document_service.upload_document(file)
    return DocumentUploadResponse(doc_id=record.id, filename=record.filename, parse_status=record.parse_status)


@router.get("")
def list_documents():
    docs = document_service.list_documents()
    return [item.model_dump(mode="json") for item in docs]


@router.get("/parse/jobs")
def list_parse_jobs(limit: int = 100):
    jobs = state_store.list_eval_jobs(limit=limit * 3)
    return [item for item in jobs if item.get("job_type") == "parse"][:limit]


@router.get("/parse/jobs/{job_id}")
def get_parse_job(job_id: str):
    item = state_store.get_eval_job(job_id)
    if not item or item.get("job_type") != "parse":
        return {"job_id": job_id, "status": "not_found", "error": "parse job not found"}
    return item


@router.get("/{doc_id}")
def get_document(doc_id: str):
    record = document_service.get_document(doc_id)
    return record.model_dump(mode="json")


@router.post("/{doc_id}/parse", response_model=ParseDocumentResponse)
def parse_document(doc_id: str, request: ParseDocumentRequest):
    if request.run_async:
        _ = document_service.get_document(doc_id)
        job_id = uuid4().hex
        queued = enqueue_parse_job(
            job_id=job_id,
            doc_id=doc_id,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        return ParseDocumentResponse(
            doc_id=doc_id,
            parse_status="queued",
            chunks=0,
            task_id=job_id,
            task_backend=queued.get("backend"),
        )

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
