from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any, Callable


class MCPClientService:
    def __init__(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[3]
        root_str = str(self.repo_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    def _load_tool(self, module_name: str, fn_name: str) -> Callable[..., Any]:
        module = importlib.import_module(module_name)
        fn = getattr(module, fn_name)
        return fn

    def search_paper(self, keyword: str, top_k: int = 5) -> list[dict]:
        fn = self._load_tool("mcp_servers.paper_eval_server.tools.search_paper", "search_paper")
        return fn(keyword=keyword, top_k=top_k)

    def get_paper_metadata(self, paper_id: str) -> dict:
        fn = self._load_tool("mcp_servers.paper_eval_server.tools.get_metadata", "get_paper_metadata")
        return fn(paper_id=paper_id)

    def query_eval_report(self, run_id: str) -> dict:
        fn = self._load_tool("mcp_servers.paper_eval_server.tools.query_eval_report", "query_eval_report")
        return fn(run_id=run_id)

    def query_failure_cases(self, failure_type: str, threshold: float = 0.0) -> list[dict]:
        fn = self._load_tool("mcp_servers.paper_eval_server.tools.query_failure_cases", "query_failure_cases")
        return fn(failure_type=failure_type, threshold=threshold)

    def extract_run_id(self, text: str) -> str | None:
        match = re.search(r"\b[a-f0-9]{32}\b", text or "")
        return match.group(0) if match else None


mcp_client_service = MCPClientService()
