from pydantic import BaseModel, Field


class EvalRunRequest(BaseModel):
    run_name: str = Field(min_length=1)
    question_set: list[dict] = Field(default_factory=list)
    golden_set_path: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    run_ragas: bool = False


class EvalRunResponse(BaseModel):
    run_id: str
    run_name: str
    status: str
    cases: int
    report_path: str


class EvalRunDetailResponse(BaseModel):
    run_id: str
    run_name: str
    status: str
    metrics: dict
    ragas: dict
    report_path: str
    created_at: str
