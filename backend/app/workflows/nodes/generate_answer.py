from __future__ import annotations

from app.services.llm_service import llm_service


def _append_trace(state: dict, step: dict) -> None:
    state.setdefault("trace_steps", []).append(step)


def run(state: dict) -> dict:
    if state.get("answer"):
        _append_trace(state, {"node": "generate_answer", "source": "precomputed"})
        return state

    question = state.get("question", "")
    route = state.get("route", "hybrid_rag")
    hits = list(state.get("hits", []))

    answer = llm_service.generate_answer(question=question, route=route, contexts=hits)
    citations = [
        {
            "chunk_id": item.get("chunk_id"),
            "doc_id": item.get("doc_id"),
            "section": item.get("section"),
            "page": item.get("page"),
            "score": item.get("score"),
            "rerank_score": item.get("rerank_score"),
            "text": item.get("text"),
        }
        for item in hits
    ]

    state["answer"] = answer
    state["citations"] = citations

    _append_trace(
        state,
        {
            "node": "generate_answer",
            "source": "llm" if llm_service.is_enabled() else "heuristic",
            "chars": len(answer),
            "citations": len(citations),
        },
    )
    return state
