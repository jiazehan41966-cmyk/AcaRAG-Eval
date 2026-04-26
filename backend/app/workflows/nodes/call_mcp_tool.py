from __future__ import annotations

from app.services.storage_service import state_store


def _append_trace(state: dict, step: dict) -> None:
    state.setdefault("trace_steps", []).append(step)


def run(state: dict) -> dict:
    docs = state_store.list_documents()
    if not docs:
        answer = "当前没有可用论文元数据，请先上传并解析文档。"
    else:
        lines = []
        for doc in docs[:5]:
            meta = doc.metadata or {}
            title = meta.get("title") or doc.filename
            authors = meta.get("authors") or "Unknown"
            lines.append(f"- {title} | authors={authors} | doc_id={doc.id}")
        answer = "已命中元数据查询路由，当前可用论文如下：\n" + "\n".join(lines)

    state["answer"] = answer
    state["hits"] = []
    state["citations"] = []

    _append_trace(state, {"node": "call_mcp_tool", "tool": "paper_metadata_mcp_server (placeholder)", "doc_count": len(docs)})
    return state
