from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from app.core.config import get_settings
from app.services.document_service import document_service
from app.services.parser_service import parser_service
from app.services.storage_service import state_store
from app.tasks.celery_app import celery_app

_executor = ThreadPoolExecutor(max_workers=2)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def run_parse_job(job_id: str, doc_id: str, chunk_size: int, chunk_overlap: int) -> dict[str, Any]:
    existing = state_store.get_eval_job(job_id) or {}
    running_payload = {
        **existing,
        "job_id": job_id,
        "job_type": "parse",
        "status": "running",
        "doc_id": doc_id,
        "backend": existing.get("backend", "thread"),
        "celery_task_id": existing.get("celery_task_id"),
        "updated_at": _utc_now(),
    }
    state_store.save_eval_job(job_id, running_payload)

    try:
        result = parse_document_task(doc_id=doc_id, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        completed_payload = {
            **running_payload,
            "status": "completed",
            "result": result,
            "updated_at": _utc_now(),
            "error": None,
        }
        state_store.save_eval_job(job_id, completed_payload)
        return completed_payload
    except Exception as exc:
        failed_payload = {
            **running_payload,
            "status": "failed",
            "error": str(exc),
            "updated_at": _utc_now(),
        }
        state_store.save_eval_job(job_id, failed_payload)
        raise


if celery_app is not None:  # pragma: no cover - broker/worker runtime

    @celery_app.task(name="app.tasks.parse_tasks.parse_document_celery")
    def parse_document_celery(job_id: str, doc_id: str, chunk_size: int, chunk_overlap: int) -> dict:
        if not get_settings().database_url:
            raise RuntimeError("Celery parse tasks require DATABASE_URL to avoid multi-process JSON write races.")
        return run_parse_job(job_id=job_id, doc_id=doc_id, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

else:
    parse_document_celery = None


def enqueue_parse_job(job_id: str, doc_id: str, chunk_size: int, chunk_overlap: int) -> dict[str, Any]:
    queued_payload = {
        "job_id": job_id,
        "job_type": "parse",
        "status": "queued",
        "doc_id": doc_id,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "backend": "thread",
        "celery_task_id": None,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "error": None,
    }

    if parse_document_celery is not None and get_settings().database_url:
        async_result = parse_document_celery.delay(job_id, doc_id, chunk_size, chunk_overlap)
        queued_payload["backend"] = "celery"
        queued_payload["celery_task_id"] = async_result.id
        state_store.save_eval_job(job_id, queued_payload)
        return queued_payload

    state_store.save_eval_job(job_id, queued_payload)
    _executor.submit(run_parse_job, job_id, doc_id, chunk_size, chunk_overlap)
    return queued_payload
