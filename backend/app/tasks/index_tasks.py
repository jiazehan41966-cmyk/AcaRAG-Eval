from __future__ import annotations

from datetime import datetime, timezone
from threading import Thread
from typing import Any

from app.core.config import get_settings
from app.services.index_service import index_service
from app.services.storage_service import state_store
from app.tasks.celery_app import celery_app


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_request(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_ids": payload.get("doc_ids"),
        "run_async": bool(payload.get("run_async", False)),
    }


def _normalize_action(action: str) -> str:
    normalized = (action or "").strip().lower()
    if normalized not in {"build", "rebuild"}:
        return "build"
    return normalized


def run_index_task(action: str, doc_ids: list[str] | None = None) -> dict:
    normalized_action = _normalize_action(action)
    if normalized_action == "rebuild":
        return index_service.rebuild_index(doc_ids=doc_ids)
    return index_service.build_index(doc_ids=doc_ids)


def run_index_job(job_id: str, action: str, request_payload: dict[str, Any]) -> dict[str, Any]:
    normalized_action = _normalize_action(action)
    normalized = _normalize_request(request_payload)
    existing = state_store.get_eval_job(job_id) or {}

    running_payload = {
        "job_id": job_id,
        "job_type": "index",
        "action": normalized_action,
        "status": "running",
        "backend": existing.get("backend"),
        "celery_task_id": existing.get("celery_task_id"),
        "request": normalized,
        "created_at": existing.get("created_at", _utc_now()),
        "updated_at": _utc_now(),
        "started_at": _utc_now(),
    }
    state_store.save_eval_job(job_id, running_payload)

    try:
        result = run_index_task(action=normalized_action, doc_ids=normalized.get("doc_ids"))
        completed_payload = {
            "job_id": job_id,
            "job_type": "index",
            "action": normalized_action,
            "status": "completed",
            "backend": running_payload.get("backend"),
            "celery_task_id": running_payload.get("celery_task_id"),
            "request": normalized,
            "created_at": running_payload["created_at"],
            "updated_at": _utc_now(),
            "finished_at": _utc_now(),
            "result": result,
        }
        state_store.save_eval_job(job_id, completed_payload)
        return completed_payload
    except Exception as exc:
        failed_payload = {
            "job_id": job_id,
            "job_type": "index",
            "action": normalized_action,
            "status": "failed",
            "backend": running_payload.get("backend"),
            "celery_task_id": running_payload.get("celery_task_id"),
            "request": normalized,
            "created_at": running_payload["created_at"],
            "updated_at": _utc_now(),
            "finished_at": _utc_now(),
            "error": str(exc),
        }
        state_store.save_eval_job(job_id, failed_payload)
        return failed_payload


if celery_app is not None:  # pragma: no cover - broker/worker runtime

    @celery_app.task(name="app.tasks.index_tasks.run_index_job_celery")
    def run_index_job_celery(job_id: str, action: str, request_payload: dict[str, Any]) -> dict[str, Any]:
        if not get_settings().database_url:
            raise RuntimeError("Celery index tasks require DATABASE_URL to avoid multi-process JSON write races.")
        return run_index_job(job_id=job_id, action=action, request_payload=request_payload)

else:
    run_index_job_celery = None


def enqueue_index_job(job_id: str, action: str, request_payload: dict[str, Any]) -> dict[str, Any]:
    normalized_action = _normalize_action(action)
    queued_payload = {
        "job_id": job_id,
        "job_type": "index",
        "action": normalized_action,
        "status": "queued",
        "request": _normalize_request(request_payload),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "backend": "thread",
    }
    state_store.save_eval_job(job_id, queued_payload)

    if run_index_job_celery is not None and get_settings().database_url:
        try:
            async_result = run_index_job_celery.delay(job_id, normalized_action, request_payload)
            queued_payload["backend"] = "celery"
            queued_payload["celery_task_id"] = async_result.id
            queued_payload["updated_at"] = _utc_now()
            state_store.save_eval_job(job_id, queued_payload)
            return queued_payload
        except Exception:
            pass

    thread = Thread(target=run_index_job, args=(job_id, normalized_action, request_payload), daemon=True)
    thread.start()
    return queued_payload
