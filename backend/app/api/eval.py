from fastapi import APIRouter

from app.schemas.eval import EvalRunDetailResponse, EvalRunRequest, EvalRunResponse
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
    )
    return EvalRunResponse(
        run_id=payload["run_id"],
        run_name=payload["run_name"],
        status=payload["status"],
        cases=payload["cases"],
        report_path=payload["report_path"],
    )


@router.get("/runs/{run_id}", response_model=EvalRunDetailResponse)
def get_eval_run(run_id: str):
    payload = eval_service.get_eval_run(run_id)
    return EvalRunDetailResponse(
        run_id=payload["run_id"],
        run_name=payload["run_name"],
        status=payload["status"],
        metrics=payload["metrics"],
        ragas=payload["ragas"],
        report_path=payload["report_path"],
        created_at=payload["created_at"],
    )
