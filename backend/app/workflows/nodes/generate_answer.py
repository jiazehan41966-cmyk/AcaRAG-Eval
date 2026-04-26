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

    answer, meta = llm_service.generate_answer(question=question, route=route, contexts=hits)
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
    state.setdefault("prompt_logs", []).append(
        {
            "node": "generate_answer",
            "prompt": meta.get("prompt"),
            "input": meta.get("input"),
            "output": answer,
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
            "node": "generate_answer",
            "source": meta.get("source", "unknown"),
            "chars": len(answer),
            "citations": len(citations),
            "prompt_tokens": meta.get("prompt_tokens", 0),
            "completion_tokens": meta.get("completion_tokens", 0),
            "token_cost": meta.get("token_cost", 0.0),
        },
    )
    return state
