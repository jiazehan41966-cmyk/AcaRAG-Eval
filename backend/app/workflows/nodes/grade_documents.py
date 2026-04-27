from __future__ import annotations

from app.services.index_service import index_service
from app.services.llm_service import llm_service


def _append_trace(state: dict, step: dict) -> None:
    state.setdefault("trace_steps", []).append(step)


def _accumulate_tokens(state: dict, meta: dict) -> None:
    usage = state.setdefault(
        "token_usage",
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "token_cost": 0.0},
    )
    usage["prompt_tokens"] = int(usage.get("prompt_tokens", 0)) + int(meta.get("prompt_tokens", 0))
    usage["completion_tokens"] = int(usage.get("completion_tokens", 0)) + int(meta.get("completion_tokens", 0))
    usage["total_tokens"] = int(usage.get("total_tokens", 0)) + int(meta.get("total_tokens", 0))
    usage["token_cost"] = float(usage.get("token_cost", 0.0)) + float(meta.get("token_cost", 0.0))


def run(state: dict) -> dict:
    query = state.get("rewritten_query") or state.get("question", "")
    question = state.get("question", "")
    hits = list(state.get("hits", []))
    top_k = int(state.get("top_k", 5))
    attempts = int(state.get("retrieve_attempts", 0))

    ranked_hits = index_service.rerank_hits(query=query, hits=hits, top_k=max(top_k * 2, 8))
    graded_hits, grade_meta = llm_service.grade_documents(question=question, hits=ranked_hits)

    _accumulate_tokens(state, grade_meta)
    state.setdefault("prompt_logs", []).append(
        {
            "node": "grade_documents",
            "prompt": grade_meta.get("prompt"),
            "input": grade_meta.get("input"),
            "output": grade_meta.get("output"),
            "model": grade_meta.get("model"),
            "source": grade_meta.get("source"),
        }
    )

    filtered = [item for item in graded_hits if item.get("llm_relevant", False)]
    if not filtered:
        filtered = sorted(graded_hits, key=lambda x: x.get("llm_relevance_score", 0.0), reverse=True)[:top_k]
    else:
        filtered = sorted(filtered, key=lambda x: x.get("llm_relevance_score", 0.0), reverse=True)[:top_k]

    avg_rerank = (
        sum(item.get("rerank_score", 0.0) for item in filtered) / len(filtered)
        if filtered
        else 0.0
    )
    avg_llm_relevance = (
        sum(item.get("llm_relevance_score", 0.0) for item in filtered) / len(filtered)
        if filtered
        else 0.0
    )

    should_retry = (len(filtered) == 0 or avg_llm_relevance < 0.45) and attempts < 2

    state["hits"] = filtered
    state["should_retry"] = should_retry

    _append_trace(
        state,
        {
            "node": "grade_documents",
            "hit_count": len(filtered),
            "avg_rerank_score": round(avg_rerank, 6),
            "avg_llm_relevance": round(avg_llm_relevance, 6),
            "should_retry": should_retry,
            "llm_enabled": llm_service.is_enabled(),
            "prompt_tokens": grade_meta.get("prompt_tokens", 0),
            "completion_tokens": grade_meta.get("completion_tokens", 0),
            "token_cost": grade_meta.get("token_cost", 0.0),
        },
    )
    return state
