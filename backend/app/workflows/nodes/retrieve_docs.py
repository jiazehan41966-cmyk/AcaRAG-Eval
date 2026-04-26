from __future__ import annotations

import time

from app.services.index_service import index_service


def _append_trace(state: dict, step: dict) -> None:
    state.setdefault("trace_steps", []).append(step)


def run(state: dict) -> dict:
    query = state.get("rewritten_query") or state.get("question", "")
    top_k = int(state.get("top_k", 5))

    start = time.perf_counter()
    try:
        hits = index_service.hybrid_search(query=query, top_k=max(top_k * 2, 8), alpha=0.6)
        error = None
    except Exception as exc:
        hits = []
        error = str(exc)

    attempts = int(state.get("retrieve_attempts", 0)) + 1
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    state["retrieve_attempts"] = attempts
    state["hits"] = hits
    state.setdefault("retrieved_docs", []).append(
        {
            "attempt": attempts,
            "query": query,
            "latency_ms": latency_ms,
            "hits": [
                {
                    "chunk_id": item.get("chunk_id"),
                    "doc_id": item.get("doc_id"),
                    "score": item.get("score"),
                    "section": item.get("section"),
                }
                for item in hits
            ],
        }
    )

    _append_trace(
        state,
        {
            "node": "retrieve_docs",
            "query": query,
            "attempt": attempts,
            "hit_count": len(hits),
            "latency_ms": latency_ms,
            "error": error,
        },
    )
    return state
