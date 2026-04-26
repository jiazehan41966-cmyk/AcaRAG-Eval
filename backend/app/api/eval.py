from fastapi import APIRouter

from app.schemas.eval import (
    EvalRunDetailResponse,
    EvalRunRequest,
    EvalRunResponse,
    EvalRunSummaryResponse,
)
from app.services.eval_service import eval_service

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.post("/run", response_model=EvalRunResponse)
def run_eval(request: EvalRunRequest):
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
