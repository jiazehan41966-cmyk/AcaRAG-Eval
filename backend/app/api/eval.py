from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/eval", tags=["evaluation"])


class EvalRunRequest(BaseModel):
    run_name: str = Field(min_length=1)
    question_set: list[dict] = Field(default_factory=list)


@router.post("/run")
def run_eval(request: EvalRunRequest):
    # v0 placeholder: wire this endpoint now to keep API stable.
    return {
        "run_name": request.run_name,
        "status": "queued",
        "message": "Evaluation pipeline placeholder. Integrate RAGAS/DeepEval in next milestone.",
        "cases": len(request.question_set),
    }


@router.get("/runs/{run_id}")
def get_eval_run(run_id: str):
    return {
        "run_id": run_id,
        "status": "not_implemented",
        "message": "Run storage not connected yet.",
    }
