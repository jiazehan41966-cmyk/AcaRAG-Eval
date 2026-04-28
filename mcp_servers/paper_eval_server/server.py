"""paper-eval-mcp-server powered by FastMCP.

This module exposes a standard MCP server plus a backwards-compatible helper
for local synchronous tool calls used by internal services.
"""

from __future__ import annotations

import argparse
import asyncio
from functools import lru_cache
from typing import Any

from fastmcp import Client, FastMCP

from mcp_servers.paper_eval_server.tools.get_citation_info import get_citation_info as get_citation_info_impl
from mcp_servers.paper_eval_server.tools.get_metadata import get_paper_metadata as get_paper_metadata_impl
from mcp_servers.paper_eval_server.tools.query_eval_report import query_eval_report as query_eval_report_impl
from mcp_servers.paper_eval_server.tools.query_failure_cases import query_failure_cases as query_failure_cases_impl
from mcp_servers.paper_eval_server.tools.search_paper import search_paper as search_paper_impl


def create_server() -> FastMCP:
    server = FastMCP(
        name="paper-eval-mcp-server",
        instructions=(
            "Academic paper metadata and evaluation report MCP tools. "
            "Use these tools for paper search, metadata lookup, evaluation run query, "
            "and failure case retrieval."
        ),
    )

    @server.tool(
        name="search_paper",
        description="Search uploaded papers by keyword and return candidate document metadata.",
    )
    def search_paper(keyword: str, top_k: int = 5) -> list[dict]:
        return search_paper_impl(keyword=keyword, top_k=top_k)

    @server.tool(
        name="get_paper_metadata",
        description="Get metadata by paper_id/doc_id.",
    )
    def get_paper_metadata(paper_id: str) -> dict:
        return get_paper_metadata_impl(paper_id=paper_id)

    @server.tool(
        name="query_eval_report",
        description="Query one evaluation run report by run_id.",
    )
    def query_eval_report(run_id: str) -> dict:
        return query_eval_report_impl(run_id=run_id)

    @server.tool(
        name="query_failure_cases",
        description="Query failure cases by type and score threshold.",
    )
    def query_failure_cases(failure_type: str, threshold: float = 0.0) -> list[dict]:
        return query_failure_cases_impl(failure_type=failure_type, threshold=threshold)

    @server.tool(
        name="get_citation_info",
        description="Get citation relationships and context for a paper by paper_id/doc_id.",
    )
    def get_citation_info(paper_id: str) -> list[dict]:
        return get_citation_info_impl(paper_id=paper_id)

    return server


@lru_cache(maxsize=1)
def get_server() -> FastMCP:
    return create_server()


async def _call_tool_async(tool_name: str, kwargs: dict[str, Any]) -> Any:
    async with Client(get_server()) as client:
        result = await client.call_tool(tool_name, kwargs)
        if result.is_error:
            raise RuntimeError(f"tool call error: {tool_name}")
        return result.data


def call_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
    try:
        result = asyncio.run(_call_tool_async(tool_name=tool_name, kwargs=kwargs))
        return {"ok": True, "result": result}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc)}


def run_server(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765, path: str = "/mcp") -> None:
    server = get_server()
    if transport == "stdio":
        server.run(transport="stdio")
        return

    server.run(
        transport=transport,
        host=host,
        port=port,
        path=path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-eval FastMCP server")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "http", "sse", "streamable-http"],
        help="MCP transport type",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for HTTP-like transports")
    parser.add_argument("--port", type=int, default=8765, help="Bind port for HTTP-like transports")
    parser.add_argument("--path", default="/mcp", help="MCP HTTP path for HTTP-like transports")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_server(transport=args.transport, host=args.host, port=args.port, path=args.path)

