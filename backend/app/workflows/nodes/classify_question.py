from __future__ import annotations


def _append_trace(state: dict, step: dict) -> None:
    state.setdefault("trace_steps", []).append(step)


def run(state: dict) -> dict:
    forced_route = state.get("force_route")
    if forced_route in {"hybrid_rag", "graph_rag", "mcp_tool"}:
        state["route"] = forced_route
        _append_trace(state, {"node": "classify_question", "route": forced_route, "forced": True})
        return state

    question = state.get("question", "")
    q = question.lower()
    metadata_keywords = ["作者", "年份", "venue", "引用", "citation", "metadata", "paper id", "doc_id"]
    compare_keywords = ["对比", "比较", "区别", "difference", "versus", "compare", "contrast", "相比", "优于"]
    graph_keywords = [
        "multi-hop",
        "multihop",
        "graph",
        "entity",
        "relation",
        "chain",
        "trace",
        "routing",
        "faithfulness",
        "failure attribution",
        "evidence chain",
        "cross-paper",
        "跨论文",
        "跨文档",
        "跨实体",
        "实体",
        "关系",
        "证据链",
        "链路",
        "路由",
        "召回",
        "一致性",
        "闭环",
        "关联",
        "多跳",
    ]

    if any(keyword in q for keyword in metadata_keywords):
        route = "mcp_tool"
    elif any(keyword in q for keyword in compare_keywords + graph_keywords):
        route = "graph_rag"
    else:
        route = "hybrid_rag"

    state["route"] = route
    _append_trace(state, {"node": "classify_question", "route": route})
    return state
