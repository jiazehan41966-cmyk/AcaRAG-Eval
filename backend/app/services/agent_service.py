from __future__ import annotations

import time
from typing import Any

from app.services.faithfulness_service import faithfulness_service
from app.services.index_service import index_service


class AgentService:
    def classify_question(self, question: str) -> str:
        q = question.lower()
        metadata_keywords = ["作者", "年份", "venue", "引用", "citation", "metadata", "paper id"]
        compare_keywords = ["对比", "比较", "区别", "difference", "versus", "相比"]

        if any(keyword in q for keyword in metadata_keywords):
            return "mcp_tool"
        if any(keyword in q for keyword in compare_keywords):
            return "graph_rag"
        return "hybrid_rag"

    def run(self, question: str, top_k: int = 5) -> dict[str, Any]:
        trace: dict[str, Any] = {"steps": []}

        t0 = time.perf_counter()
        route = self.classify_question(question)
        trace["steps"].append({"node": "classify_question", "route": route})

        # v0: all routes fall back to hybrid retrieval, while preserving the route decision in trace.
        retrieve_start = time.perf_counter()
        hits = index_service.hybrid_search(question, top_k=top_k)
        trace["steps"].append(
            {
                "node": "retrieve_docs",
                "strategy": "hybrid",
                "hit_count": len(hits),
                "latency_ms": round((time.perf_counter() - retrieve_start) * 1000, 2),
            }
        )

        answer = self.generate_answer(question, hits)
        trace["steps"].append({"node": "generate_answer", "chars": len(answer)})

        faithfulness = faithfulness_service.check(answer, [hit["text"] for hit in hits])
        trace["steps"].append({"node": "check_faithfulness", **faithfulness})

        trace["total_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "question": question,
            "route": route,
            "answer": answer,
            "citations": [
                {
                    "chunk_id": hit["chunk_id"],
                    "doc_id": hit["doc_id"],
                    "section": hit.get("section"),
                    "page": hit.get("page"),
                    "score": hit["score"],
                }
                for hit in hits
            ],
            "faithfulness": faithfulness,
            "trace": trace,
        }

    def generate_answer(self, question: str, hits: list[dict]) -> str:
        if not hits:
            return "没有检索到可用证据，建议先上传并解析文档，再构建索引。"

        evidence_lines = []
        for i, hit in enumerate(hits[:3], start=1):
            snippet = hit["text"][:220].strip()
            section = hit.get("section") or "Unknown Section"
            marker = f"[{i}]"
            evidence_lines.append(f"{marker} ({section}) {snippet}")

        summary = "\n".join(evidence_lines)
        return (
            f"问题：{question}\n"
            "基于当前检索证据，先给出一个保守回答：\n"
            f"{summary}\n"
            "如果你希望，我下一步可以切换到 LLM 生成模式（含更自然的归纳与跨段综合）。"
        )


agent_service = AgentService()
