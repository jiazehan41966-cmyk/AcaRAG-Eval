"""LangGraph workflow placeholder.

Current version uses app.services.agent_service.AgentService as a lightweight stand-in.
Replace with a real StateGraph implementation in milestone 3.
"""

from typing import Any


def run_agentic_rag(question: str, top_k: int = 5) -> dict[str, Any]:
    from app.services.agent_service import agent_service

    return agent_service.run(question=question, top_k=top_k)
