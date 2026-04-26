from __future__ import annotations

from app.services.llm_service import llm_service


def _append_trace(state: dict, step: dict) -> None:
    state.setdefault("trace_steps", []).append(step)


def run(state: dict) -> dict:
    question = state.get("question", "")
    route = state.get("route", "hybrid_rag")
    attempts = state.get("retrieve_attempts", 0)

    rewritten = llm_service.rewrite_query(question, route)
    if attempts > 0:
        rewritten = f"{rewritten} 关键术语 同义词 论文结论"

    state["rewritten_query"] = rewritten
    _append_trace(
        state,
        {
            "node": "rewrite_query",
            "attempt": attempts,
            "rewritten_query": rewritten,
            "llm_enabled": llm_service.is_enabled(),
        },
    )
    return state
