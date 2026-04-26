from __future__ import annotations

from app.services.llm_service import llm_service


def _append_trace(state: dict, step: dict) -> None:
    state.setdefault("trace_steps", []).append(step)


def run(state: dict) -> dict:
    question = state.get("question", "")
    route = state.get("route", "hybrid_rag")
    attempts = int(state.get("retrieve_attempts", 0))

    rewritten, meta = llm_service.rewrite_query(question, route)
    if attempts > 0:
        rewritten = f"{rewritten} 关键术语 同义词 论文结论"

    state["rewritten_query"] = rewritten
    state.setdefault("prompt_logs", []).append(
        {
            "node": "rewrite_query",
            "prompt": meta.get("prompt"),
            "input": meta.get("input"),
            "output": rewritten,
            "model": meta.get("model"),
            "source": meta.get("source"),
        }
    )
    state["token_usage"] = {
        "prompt_tokens": int(state.get("token_usage", {}).get("prompt_tokens", 0)) + int(meta.get("prompt_tokens", 0)),
        "completion_tokens": int(state.get("token_usage", {}).get("completion_tokens", 0))
        + int(meta.get("completion_tokens", 0)),
        "total_tokens": int(state.get("token_usage", {}).get("total_tokens", 0)) + int(meta.get("total_tokens", 0)),
        "token_cost": float(state.get("token_usage", {}).get("token_cost", 0.0)) + float(meta.get("token_cost", 0.0)),
    }

    _append_trace(
        state,
        {
            "node": "rewrite_query",
            "attempt": attempts,
            "rewritten_query": rewritten,
            "llm_enabled": llm_service.is_enabled(),
            "prompt_tokens": meta.get("prompt_tokens", 0),
            "completion_tokens": meta.get("completion_tokens", 0),
            "token_cost": meta.get("token_cost", 0.0),
        },
    )
    return state
