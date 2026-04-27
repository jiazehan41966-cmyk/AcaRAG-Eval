from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter

from app.schemas.eval import (
    EvalJobResponse,
    EvalRunDetailResponse,
    EvalRunRequest,
    EvalRunResponse,
    EvalRunSummaryResponse,
)
from app.services.eval_service import eval_service
from app.services.storage_service import state_store
from app.tasks.eval_tasks import enqueue_eval_job

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.post("/run", response_model=EvalRunResponse)
def run_eval(request: EvalRunRequest):
    if request.run_async:
        job_id = uuid4().hex
        queued = enqueue_eval_job(job_id=job_id, request_payload=request.model_dump())
        return EvalRunResponse(
            run_id=None,
            run_name=request.run_name,
            status="queued",
            cases=None,
            report_path=None,
            regression_report_path=None,
            baseline_name=request.baseline_name,
            task_id=job_id,
            task_backend=queued.get("backend"),
            failure_summary={},
        )

    payload = eval_service.run_eval(
        run_name=request.run_name,
        question_set=request.question_set,
        top_k=request.top_k,
        golden_set_path=request.golden_set_path,
        run_ragas=request.run_ragas,
        run_deepeval=request.run_deepeval,
        baseline_name=request.baseline_name,
        set_as_baseline=request.set_as_baseline,
        compare_with_baseline=request.compare_with_baseline,
    )
    return EvalRunResponse(
        run_id=payload["run_id"],
        run_name=payload["run_name"],
        status=payload["status"],
        cases=payload["cases"],
        report_path=payload["report_path"],
        regression_report_path=payload.get("regression_report_path"),
        baseline_name=payload.get("baseline_name"),
        task_id=None,
        task_backend=None,
        failure_summary=payload.get("failure_summary", {}),
    )


@router.get("/runs", response_model=list[EvalRunSummaryResponse])
def list_eval_runs():
    runs = eval_service.list_eval_runs()
    return [
        EvalRunSummaryResponse(
            run_id=item["run_id"],
            run_name=item["run_name"],
            status=item["status"],
            cases=item["cases"],
            created_at=item["created_at"],
            metrics=item.get("metrics", {}),
            channels=item.get("channels", {}),
            failure_summary=item.get("failure_summary", {}),
            report_path=item.get("report_path", ""),
            regression_report_path=item.get("regression_report_path"),
            baseline_name=item.get("baseline_name"),
        )
        for item in runs
    ]


@router.get("/runs/{run_id}", response_model=EvalRunDetailResponse)
def get_eval_run(run_id: str):
    payload = eval_service.get_eval_run(run_id)
    return EvalRunDetailResponse(
        run_id=payload["run_id"],
        run_name=payload["run_name"],
        status=payload["status"],
        metrics=payload.get("metrics", {}),
        channels=payload.get("channels", {}),
        ragas=payload.get("ragas", {}),
        deepeval=payload.get("deepeval", {}),
        failure_summary=payload.get("failure_summary", {}),
        baseline_name=payload.get("baseline_name"),
        baseline_comparison=payload.get("baseline_comparison"),
        report_path=payload.get("report_path", ""),
        regression_report_path=payload.get("regression_report_path"),
        created_at=payload["created_at"],
        results=payload.get("results", []),
    )


@router.get("/jobs", response_model=list[EvalJobResponse])
def list_eval_jobs(limit: int = 100):
    jobs = state_store.list_eval_jobs(limit=limit * 3)
    eval_jobs = [item for item in jobs if item.get("job_type") in {None, "eval"}][:limit]
    return [
        EvalJobResponse(
            job_id=item.get("job_id", ""),
            job_type=item.get("job_type"),
            status=item.get("status", "unknown"),
            backend=item.get("backend"),
            celery_task_id=item.get("celery_task_id"),
            created_at=item.get("created_at"),
            updated_at=item.get("updated_at"),
            run_id=item.get("run_id"),
            run_name=item.get("run_name"),
            error=item.get("error"),
        )
        for item in eval_jobs
    ]


@router.get("/jobs/{job_id}", response_model=EvalJobResponse)
def get_eval_job(job_id: str):
    item = state_store.get_eval_job(job_id) or {
        "job_id": job_id,
        "status": "not_found",
        "error": "job not found",
    }
    if item.get("job_type") not in {None, "eval"}:
        item = {
            "job_id": job_id,
            "status": "not_found",
            "error": "eval job not found",
        }
    return EvalJobResponse(
        job_id=item.get("job_id", job_id),
        job_type=item.get("job_type"),
        status=item.get("status", "unknown"),
        backend=item.get("backend"),
        celery_task_id=item.get("celery_task_id"),
        created_at=item.get("created_at"),
        updated_at=item.get("updated_at"),
        run_id=item.get("run_id"),
        run_name=item.get("run_name"),
        error=item.get("error"),
    )
