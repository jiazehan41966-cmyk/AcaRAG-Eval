from __future__ import annotations

from app.core.config import get_settings
from app.services.document_service import document_service
from app.services.parser_service import parser_service
from app.tasks.celery_app import celery_app


def parse_document_task(doc_id: str, chunk_size: int, chunk_overlap: int) -> dict:
    record = document_service.get_document(doc_id)
    record, chunks = parser_service.parse_document(
        record,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return {
        "doc_id": record.id,
        "parse_status": record.parse_status,
        "chunks": len(chunks),
    }


if celery_app is not None:  # pragma: no cover - broker/worker runtime

    @celery_app.task(name="app.tasks.parse_tasks.parse_document_celery")
    def parse_document_celery(doc_id: str, chunk_size: int, chunk_overlap: int) -> dict:
        if not get_settings().database_url:
            raise RuntimeError("Celery parse tasks require DATABASE_URL to avoid multi-process JSON write races.")
        return parse_document_task(doc_id=doc_id, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

else:
    parse_document_celery = None
