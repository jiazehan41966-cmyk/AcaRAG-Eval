from pydantic import BaseModel, Field


class EvalRunRequest(BaseModel):
    run_name: str = Field(min_length=1)
    question_set: list[dict] = Field(default_factory=list)
    golden_set_path: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    run_ragas: bool = False
    run_deepeval: bool = False
    baseline_name: str | None = None
    set_as_baseline: bool = False
    compare_with_baseline: bool = True
    run_async: bool = False


class EvalRunResponse(BaseModel):
    run_id: str | None = None
    run_name: str | None = None
    status: str
    cases: int | None = None
    report_path: str | None = None
    regression_report_path: str | None = None
    baseline_name: str | None = None
    task_id: str | None = None
    task_backend: str | None = None
    failure_summary: dict = Field(default_factory=dict)


class EvalRunSummaryResponse(BaseModel):
    run_id: str
    run_name: str
    status: str
    cases: int
    created_at: str
    metrics: dict = Field(default_factory=dict)
    channels: dict = Field(default_factory=dict)
    failure_summary: dict = Field(default_factory=dict)
    report_path: str
    regression_report_path: str | None = None
    baseline_name: str | None = None


class EvalRunDetailResponse(BaseModel):
    run_id: str
    run_name: str
    status: str
    metrics: dict = Field(default_factory=dict)
    channels: dict = Field(default_factory=dict)
    ragas: dict = Field(default_factory=dict)
    deepeval: dict = Field(default_factory=dict)
    failure_summary: dict = Field(default_factory=dict)
    baseline_name: str | None = None
    baseline_comparison: dict | None = None
    report_path: str
    regression_report_path: str | None = None
    created_at: str
    results: list[dict] = Field(default_factory=list)


class EvalJobResponse(BaseModel):
    job_id: str
    job_type: str | None = None
    status: str
    backend: str | None = None
    celery_task_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    error: str | None = None
