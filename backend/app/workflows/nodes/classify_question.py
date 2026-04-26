def run(state: dict) -> dict:
    question = state.get("question", "")
    q = question.lower()
    if any(k in q for k in ["作者", "年份", "citation", "metadata"]):
        state["route"] = "mcp_tool"
    elif any(k in q for k in ["对比", "比较", "difference", "versus"]):
        state["route"] = "graph_rag"
    else:
        state["route"] = "hybrid_rag"
    return state
