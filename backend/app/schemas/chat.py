from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    question: str
    route: str
    answer: str
    trace_id: str | None = None
    citations: list[dict]
    faithfulness: dict
    trace: dict
