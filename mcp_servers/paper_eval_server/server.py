"""paper-eval-mcp-server: lightweight tool dispatcher used by workflow nodes.

This file intentionally keeps a minimal runtime surface for local development.
"""

from __future__ import annotations

from mcp_servers.paper_eval_server.tools.get_metadata import get_paper_metadata
from mcp_servers.paper_eval_server.tools.query_eval_report import query_eval_report
from mcp_servers.paper_eval_server.tools.query_failure_cases import query_failure_cases
from mcp_servers.paper_eval_server.tools.search_paper import search_paper

TOOLS = {
    "search_paper": search_paper,
    "get_paper_metadata": get_paper_metadata,
    "query_eval_report": query_eval_report,
    "query_failure_cases": query_failure_cases,
}


def call_tool(tool_name: str, **kwargs):
    if tool_name not in TOOLS:
        return {"ok": False, "error": f"unknown tool: {tool_name}"}
    try:
        return {"ok": True, "result": TOOLS[tool_name](**kwargs)}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc)}


def run_server() -> None:
    print("paper-eval-mcp-server ready with tools:")
    for name in TOOLS:
        print(f"- {name}")


if __name__ == "__main__":
    run_server()
