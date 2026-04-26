from __future__ import annotations

from app.services.index_service import index_service


def _append_trace(state: dict, step: dict) -> None:
    state.setdefault("trace_steps", []).append(step)


def run(state: dict) -> dict:
    query = state.get("rewritten_query") or state.get("question", "")
    hits = list(state.get("hits", []))
    top_k = int(state.get("top_k", 5))

    ranked_hits = index_service.rerank_hits(query=query, hits=hits, top_k=top_k)
    avg_rerank = sum(item.get("rerank_score", 0.0) for item in ranked_hits) / len(ranked_hits) if ranked_hits else 0.0

    should_retry = (len(ranked_hits) == 0 or avg_rerank < 0.15) and int(state.get("retrieve_attempts", 0)) < 2

    state["hits"] = ranked_hits
    state["should_retry"] = should_retry

    _append_trace(
        state,
        {
            "node": "grade_documents",
            "hit_count": len(ranked_hits),
            "avg_rerank_score": round(avg_rerank, 6),
            "should_retry": should_retry,
        },
    )
    return state
