from __future__ import annotations


def _append_trace(state: dict, step: dict) -> None:
    state.setdefault("trace_steps", []).append(step)


def run(state: dict) -> dict:
    question = state.get("question", "")
    q = question.lower()
    metadata_keywords = ["作者", "年份", "venue", "引用", "citation", "metadata", "paper id"]
    compare_keywords = ["对比", "比较", "区别", "difference", "versus", "相比", "优于"]

    if any(keyword in q for keyword in metadata_keywords):
        route = "mcp_tool"
    elif any(keyword in q for keyword in compare_keywords):
        route = "graph_rag"
    else:
        route = "hybrid_rag"

    state["route"] = route
    _append_trace(state, {"node": "classify_question", "route": route})
    return state
