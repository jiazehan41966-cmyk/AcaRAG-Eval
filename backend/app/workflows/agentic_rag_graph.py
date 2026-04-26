from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.services.langfuse_service import langfuse_service
from app.workflows.nodes import (
    call_mcp_tool,
    check_faithfulness,
    classify_question,
    generate_answer,
    grade_documents,
    retrieve_docs,
    rewrite_query,
)


class AgentGraphState(TypedDict, total=False):
    question: str
    route: str
    rewritten_query: str
    top_k: int
    retrieve_attempts: int
    should_retry: bool
    hits: list[dict]
    citations: list[dict]
    answer: str
    faithfulness: dict
    trace_steps: list[dict]
    trace_id: str
    prompt_logs: list[dict]
    retrieved_docs: list[dict]
    tool_calls: list[dict]
    token_usage: dict


class AgenticRAGGraph:
    def __init__(self) -> None:
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentGraphState)

        builder.add_node("classify_question", classify_question.run)
        builder.add_node("call_mcp_tool", call_mcp_tool.run)
        builder.add_node("rewrite_query", rewrite_query.run)
        builder.add_node("retrieve_docs", retrieve_docs.run)
        builder.add_node("grade_documents", grade_documents.run)
        builder.add_node("generate_answer", generate_answer.run)
        builder.add_node("check_faithfulness", check_faithfulness.run)

        builder.add_edge(START, "classify_question")

        builder.add_conditional_edges(
            "classify_question",
            self._route_after_classify,
            {
                "mcp_tool": "call_mcp_tool",
                "retrieval": "rewrite_query",
            },
        )

        builder.add_edge("rewrite_query", "retrieve_docs")
        builder.add_edge("retrieve_docs", "grade_documents")

        builder.add_conditional_edges(
            "grade_documents",
            self._route_after_grading,
            {
                "retry": "rewrite_query",
                "generate": "generate_answer",
            },
        )

        builder.add_edge("call_mcp_tool", "check_faithfulness")
        builder.add_edge("generate_answer", "check_faithfulness")
        builder.add_edge("check_faithfulness", END)

        return builder.compile()

    @staticmethod
    def _route_after_classify(state: AgentGraphState) -> str:
        if state.get("route") == "mcp_tool":
            return "mcp_tool"
        return "retrieval"

    @staticmethod
    def _route_after_grading(state: AgentGraphState) -> str:
        return "retry" if state.get("should_retry") else "generate"

    def run(self, question: str, top_k: int = 5) -> dict:
        trace_record = langfuse_service.create_trace(
            name="agentic_rag.ask",
            input_payload={"question": question, "top_k": top_k},
            metadata={"component": "agentic_rag_graph"},
        )
        trace_id = trace_record["trace_id"]

        t0 = time.perf_counter()
        initial_state: AgentGraphState = {
            "question": question,
            "top_k": top_k,
            "retrieve_attempts": 0,
            "should_retry": False,
            "hits": [],
            "citations": [],
            "trace_steps": [],
            "trace_id": trace_id,
            "prompt_logs": [],
            "retrieved_docs": [],
            "tool_calls": [],
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "token_cost": 0.0},
        }

        result: AgentGraphState = self.graph.invoke(initial_state)
        total_latency = round((time.perf_counter() - t0) * 1000, 2)

        output_payload = {
            "question": question,
            "route": result.get("route", "hybrid_rag"),
            "answer": result.get("answer", ""),
            "trace_id": trace_id,
            "citations": result.get("citations", []),
            "faithfulness": result.get(
                "faithfulness",
                {
                    "score": 0.0,
                    "supported_claim_ratio": 0.0,
                    "reason": "not_available",
                },
            ),
            "trace": {
                "trace_id": trace_id,
                "steps": result.get("trace_steps", []),
                "prompts": result.get("prompt_logs", []),
                "retrieved_docs": result.get("retrieved_docs", []),
                "tool_calls": result.get("tool_calls", []),
                "latency_ms": total_latency,
                "token_usage": result.get("token_usage", {}),
            },
        }

        for prompt_item in output_payload["trace"].get("prompts", []):
            langfuse_service.add_event(
                trace_id=trace_id,
                name="agentic_rag.prompt",
                input_payload={"node": prompt_item.get("node"), "input": prompt_item.get("input")},
                output_payload={"output": prompt_item.get("output"), "model": prompt_item.get("model")},
                metadata={"source": prompt_item.get("source")},
            )

        for retrieval_item in output_payload["trace"].get("retrieved_docs", []):
            langfuse_service.add_event(
                trace_id=trace_id,
                name="agentic_rag.retrieval",
                metadata={
                    "attempt": retrieval_item.get("attempt"),
                    "query": retrieval_item.get("query"),
                    "latency_ms": retrieval_item.get("latency_ms"),
                    "hits": retrieval_item.get("hits", []),
                },
            )

        for tool_item in output_payload["trace"].get("tool_calls", []):
            langfuse_service.add_event(
                trace_id=trace_id,
                name="agentic_rag.tool_call",
                metadata=tool_item,
            )

        langfuse_service.add_event(
            trace_id=trace_id,
            name="agentic_rag.trace",
            metadata={
                "route": output_payload["route"],
                "latency_ms": total_latency,
                "token_usage": output_payload["trace"].get("token_usage", {}),
            },
            output_payload={
                "steps": len(output_payload["trace"].get("steps", [])),
                "retrieval_attempts": result.get("retrieve_attempts", 0),
            },
        )
        langfuse_service.finalize_trace(
            trace_id=trace_id,
            output_payload={
                "answer": output_payload["answer"],
                "faithfulness": output_payload["faithfulness"],
            },
            metadata={
                "token_usage": output_payload["trace"].get("token_usage", {}),
                "latency_ms": total_latency,
            },
        )

        return output_payload


agentic_rag_graph = AgenticRAGGraph()


def run_agentic_rag(question: str, top_k: int = 5) -> dict:
    return agentic_rag_graph.run(question=question, top_k=top_k)
