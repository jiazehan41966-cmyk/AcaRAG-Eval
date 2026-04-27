from __future__ import annotations

from datetime import datetime, timezone
from threading import Thread
from typing import Any

from app.core.config import get_settings
from app.services.eval_service import eval_service
from app.services.storage_service import state_store
from app.tasks.celery_app import celery_app


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_request(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": str(payload.get("run_name") or f"eval_async_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"),
        "question_set": payload.get("question_set") or [],
        "top_k": int(payload.get("top_k", 5)),
        "golden_set_path": payload.get("golden_set_path"),
        "run_ragas": bool(payload.get("run_ragas", False)),
        "run_deepeval": bool(payload.get("run_deepeval", False)),
        "baseline_name": payload.get("baseline_name"),
        "set_as_baseline": bool(payload.get("set_as_baseline", False)),
        "compare_with_baseline": bool(payload.get("compare_with_baseline", True)),
    }


def run_eval_task(
    run_name: str,
    question_set: list[dict[str, Any]] | None = None,
    top_k: int = 5,
    golden_set_path: str | None = None,
    run_ragas: bool = False,
    run_deepeval: bool = False,
    baseline_name: str | None = None,
    set_as_baseline: bool = False,
    compare_with_baseline: bool = True,
) -> dict[str, Any]:
    return eval_service.run_eval(
        run_name=run_name,
        question_set=question_set or [],
        top_k=top_k,
        golden_set_path=golden_set_path,
        run_ragas=run_ragas,
        run_deepeval=run_deepeval,
        baseline_name=baseline_name,
        set_as_baseline=set_as_baseline,
        compare_with_baseline=compare_with_baseline,
    )


def run_eval_job(job_id: str, request_payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_request(request_payload)
    existing = state_store.get_eval_job(job_id) or {}
    running_payload = {
        "job_id": job_id,
        "job_type": "eval",
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
        result = run_eval_task(**normalized)
        completed_payload = {
            "job_id": job_id,
            "job_type": "eval",
            "status": "completed",
            "backend": running_payload.get("backend"),
            "celery_task_id": running_payload.get("celery_task_id"),
            "request": normalized,
            "created_at": running_payload["created_at"],
            "updated_at": _utc_now(),
            "finished_at": _utc_now(),
            "run_id": result.get("run_id"),
            "run_name": result.get("run_name"),
            "cases": result.get("cases"),
            "report_path": result.get("report_path"),
            "regression_report_path": result.get("regression_report_path"),
            "baseline_name": result.get("baseline_name"),
            "result": result,
        }
        state_store.save_eval_job(job_id, completed_payload)
        return completed_payload
    except Exception as exc:
        failed_payload = {
            "job_id": job_id,
            "job_type": "eval",
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


if celery_app is not None:  # pragma: no cover - requires broker/worker runtime

    @celery_app.task(name="app.tasks.eval_tasks.run_eval_job_celery")
    def run_eval_job_celery(job_id: str, request_payload: dict[str, Any]) -> dict[str, Any]:
        if not get_settings().database_url:
            raise RuntimeError("Celery eval tasks require DATABASE_URL to avoid multi-process JSON write races.")
        return run_eval_job(job_id=job_id, request_payload=request_payload)

else:
    run_eval_job_celery = None


def enqueue_eval_job(job_id: str, request_payload: dict[str, Any]) -> dict[str, Any]:
    queued_payload = {
        "job_id": job_id,
        "job_type": "eval",
        "status": "queued",
        "request": _normalize_request(request_payload),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "backend": "thread",
    }
    state_store.save_eval_job(job_id, queued_payload)

    if run_eval_job_celery is not None and get_settings().database_url:
        try:
            async_result = run_eval_job_celery.delay(job_id, request_payload)
            queued_payload["backend"] = "celery"
            queued_payload["celery_task_id"] = async_result.id
            queued_payload["updated_at"] = _utc_now()
            state_store.save_eval_job(job_id, queued_payload)
            return queued_payload
        except Exception:
            pass

    thread = Thread(target=run_eval_job, args=(job_id, request_payload), daemon=True)
    thread.start()
    return queued_payload
