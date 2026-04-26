from __future__ import annotations

from app.services.mcp_client_service import mcp_client_service


def _append_trace(state: dict, step: dict) -> None:
    state.setdefault("trace_steps", []).append(step)


def _record_tool_call(state: dict, tool_name: str, args: dict, result: dict | list) -> None:
    state.setdefault("tool_calls", []).append(
        {
            "tool": tool_name,
            "args": args,
            "result": result,
        }
    )


def run(state: dict) -> dict:
    question = str(state.get("question", ""))
    lowered = question.lower()

    answer = "MCP 查询未命中具体工具。"
    hits: list[dict] = []

    if any(x in lowered for x in ["评测", "eval", "report", "run"]):
        run_id = mcp_client_service.extract_run_id(question)
        if run_id:
            result = mcp_client_service.query_eval_report(run_id=run_id)
            _record_tool_call(state, "query_eval_report", {"run_id": run_id}, result)
            if result.get("found"):
                answer = (
                    f"评测运行 `{run_id}` 状态：{result.get('status')}；"
                    f"用例数：{result.get('cases')}；"
                    f"指标：{result.get('metrics')}；"
                    f"失败归因：{result.get('failure_summary')}。"
                )
            else:
                answer = f"未找到评测运行 `{run_id}`。"
        else:
            fail_type = "retrieval_miss"
            if "rerank" in lowered:
                fail_type = "rerank_error"
            if "citation" in lowered:
                fail_type = "citation_error"
            if "unsupported" in lowered or "幻觉" in lowered:
                fail_type = "unsupported_claim"

            result = mcp_client_service.query_failure_cases(failure_type=fail_type, threshold=0.0)
            _record_tool_call(state, "query_failure_cases", {"failure_type": fail_type}, result)
            if result:
                lines = [
                    f"- {item.get('question')} (run={item.get('run_id')}, score={item.get('failure_score')})"
                    for item in result[:5]
                ]
                answer = f"已检索到 `{fail_type}` 失败样例：\n" + "\n".join(lines)
            else:
                answer = f"当前未检索到 `{fail_type}` 失败样例。"
    else:
        docs = mcp_client_service.search_paper(keyword=question, top_k=5)
        _record_tool_call(state, "search_paper", {"keyword": question, "top_k": 5}, docs)

        if not docs:
            answer = "未检索到相关论文元数据。"
        elif len(docs) == 1:
            doc = docs[0]
            meta = mcp_client_service.get_paper_metadata(paper_id=doc.get("doc_id"))
            _record_tool_call(state, "get_paper_metadata", {"paper_id": doc.get("doc_id")}, meta)
            answer = (
                f"命中论文：{meta.get('title')}\n"
                f"authors={meta.get('authors')}\n"
                f"pages={meta.get('pages')}\n"
                f"chunk_count={meta.get('chunk_count')}\n"
                f"doc_id={meta.get('paper_id')}"
            )
        else:
            lines = [f"- {item.get('title')} (doc_id={item.get('doc_id')})" for item in docs[:5]]
            answer = "命中多篇论文，请进一步指定 doc_id：\n" + "\n".join(lines)

    state["answer"] = answer
    state["hits"] = hits
    state["citations"] = []

    _append_trace(
        state,
        {
            "node": "call_mcp_tool",
            "tool_count": len(state.get("tool_calls", [])),
            "tools": [item["tool"] for item in state.get("tool_calls", [])],
        },
    )
    return state
