from __future__ import annotations

from app.services.faithfulness_service import faithfulness_service


def _append_trace(state: dict, step: dict) -> None:
    state.setdefault("trace_steps", []).append(step)


def run(state: dict) -> dict:
    answer = state.get("answer", "")
    hits = list(state.get("hits", []))
    contexts = [item.get("text", "") for item in hits]

    faithfulness = faithfulness_service.check(answer=answer, contexts=contexts)
    state["faithfulness"] = faithfulness

    _append_trace(state, {"node": "check_faithfulness", **faithfulness})
    return state
