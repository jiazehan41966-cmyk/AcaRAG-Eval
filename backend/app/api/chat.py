from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.agent_service import agent_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=ChatResponse)
def ask(request: ChatRequest):
    result = agent_service.run(request.question, top_k=request.top_k)
    return ChatResponse(**result)
