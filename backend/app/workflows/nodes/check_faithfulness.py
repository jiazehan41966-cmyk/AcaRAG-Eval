from __future__ import annotations

from app.services.faithfulness_service import faithfulness_service


def _append_trace(state: dict, step: dict) -> None:
    state.setdefault("trace_steps", []).append(step)


def _accumulate_tokens(state: dict, result: dict) -> None:
    usage = state.setdefault(
        "token_usage",
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "token_cost": 0.0},
    )
    usage["prompt_tokens"] = int(usage.get("prompt_tokens", 0)) + int(result.get("prompt_tokens", 0))
    usage["completion_tokens"] = int(usage.get("completion_tokens", 0)) + int(result.get("completion_tokens", 0))
    usage["total_tokens"] = int(usage.get("total_tokens", 0)) + int(result.get("total_tokens", 0))
    usage["token_cost"] = float(usage.get("token_cost", 0.0)) + float(result.get("token_cost", 0.0))


def run(state: dict) -> dict:
    question = state.get("question", "")
    answer = state.get("answer", "")
    hits = list(state.get("hits", []))
    contexts = [item.get("text", "") for item in hits]

    faithfulness = faithfulness_service.check(
        question=question,
        answer=answer,
        contexts=contexts,
    )
    state["faithfulness"] = {
        "score": faithfulness.get("score", 0.0),
        "supported_claim_ratio": faithfulness.get("supported_claim_ratio", 0.0),
        "reason": faithfulness.get("reason", "unknown"),
        "source": faithfulness.get("source", "heuristic"),
    }
    _accumulate_tokens(state, faithfulness)
    state.setdefault("prompt_logs", []).append(
        {
            "node": "check_faithfulness",
            "prompt": None,
            "input": question,
            "output": state["faithfulness"],
            "model": "judge",
            "source": faithfulness.get("source", "heuristic"),
        }
    )

    _append_trace(
        state,
        {
            "node": "check_faithfulness",
            "score": state["faithfulness"]["score"],
            "supported_claim_ratio": state["faithfulness"]["supported_claim_ratio"],
            "reason": state["faithfulness"]["reason"],
            "source": state["faithfulness"]["source"],
            "prompt_tokens": faithfulness.get("prompt_tokens", 0),
            "completion_tokens": faithfulness.get("completion_tokens", 0),
            "token_cost": faithfulness.get("token_cost", 0.0),
        },
    )
    return state
