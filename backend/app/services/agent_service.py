from __future__ import annotations

from typing import Any

from app.workflows.agentic_rag_graph import run_agentic_rag


class AgentService:
    def run(self, question: str, top_k: int = 5, force_route: str | None = None) -> dict[str, Any]:
        return run_agentic_rag(question=question, top_k=top_k, force_route=force_route)


agent_service = AgentService()
