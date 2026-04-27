from __future__ import annotations

import asyncio
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from fastmcp import Client

from app.core.config import get_settings


class MCPClientService:
    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        self.settings = get_settings()
        self._transport = self._resolve_transport()

    def _resolve_transport(self) -> Any:
        if self.settings.mcp_mode.lower() == "http":
            return self.settings.mcp_server_url
        module = importlib.import_module("mcp_servers.paper_eval_server.server")
        return module.get_server()

    async def _call_tool_async(self, tool_name: str, args: dict[str, Any]) -> Any:
        async with Client(self._transport, timeout=self.settings.mcp_timeout_seconds) as client:
            response = await client.call_tool(tool_name, args)
            if response.is_error:
                raise RuntimeError(f"MCP tool call failed: {tool_name}")

            if response.data is not None:
                return response.data

            text_fragments = [getattr(item, "text", "") for item in (response.content or [])]
            merged = "\n".join(fragment for fragment in text_fragments if fragment).strip()
            if not merged:
                return None

            try:
                return json.loads(merged)
            except json.JSONDecodeError:
                return {"text": merged}

    def _call_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        return asyncio.run(self._call_tool_async(tool_name=tool_name, args=args))

    def search_paper(self, keyword: str, top_k: int = 5) -> list[dict]:
        result = self._call_tool("search_paper", {"keyword": keyword, "top_k": top_k})
        return result if isinstance(result, list) else []

    def get_paper_metadata(self, paper_id: str) -> dict:
        result = self._call_tool("get_paper_metadata", {"paper_id": paper_id})
        return result if isinstance(result, dict) else {"paper_id": paper_id, "found": False}

    def query_eval_report(self, run_id: str) -> dict:
        result = self._call_tool("query_eval_report", {"run_id": run_id})
        return result if isinstance(result, dict) else {"run_id": run_id, "found": False}

    def query_failure_cases(self, failure_type: str, threshold: float = 0.0) -> list[dict]:
        result = self._call_tool(
            "query_failure_cases",
            {"failure_type": failure_type, "threshold": threshold},
        )
        return result if isinstance(result, list) else []

    def extract_run_id(self, text: str) -> str | None:
        match = re.search(r"\b[a-f0-9]{32}\b", text or "")
        return match.group(0) if match else None

    def health(self) -> dict[str, Any]:
        return {
            "mcp_mode": self.settings.mcp_mode,
            "mcp_server_url": self.settings.mcp_server_url if self.settings.mcp_mode.lower() == "http" else "inprocess",
        }


mcp_client_service = MCPClientService()

