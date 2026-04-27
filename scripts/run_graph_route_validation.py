#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run graph route validation on comparison/multi-hop cases")
    parser.add_argument("--golden-set", default="data/golden_set/graph_eval_set.jsonl")
    parser.add_argument("--run-name", default="graph_route_validation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=32)
    args = parser.parse_args()

    settings = get_settings()
    cases = _load_jsonl(Path(args.golden_set))
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit(f"no graph eval cases found: {args.golden_set}")

    client = TestClient(app)
    run_name = f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    response = client.post(
        "/api/eval/run",
        json={
            "run_name": run_name,
            "question_set": cases,
            "top_k": args.top_k,
            "run_ragas": False,
            "run_deepeval": False,
            "compare_with_baseline": False,
        },
    )
    if response.status_code != 200:
        raise SystemExit(f"eval run failed: status={response.status_code}, body={response.text}")

    run = client.get(f"/api/eval/runs/{response.json()['run_id']}").json()
    results = run.get("results", [])

    route_counts = Counter(item.get("route", "unknown") for item in results)
    by_type: dict[str, Counter] = defaultdict(Counter)
    for item in results:
        by_type[item.get("question_type", "unknown")][item.get("route", "unknown")] += 1

    graph_cases = [item for item in results if item.get("route") == "graph_rag"]
    graph_rate = len(graph_cases) / len(results) if results else 0.0
    faithfulness = _avg([float(item.get("faithfulness", 0.0)) for item in results])
    context_recall = _avg([float(item.get("context_recall", 0.0)) for item in results])
    answer_relevancy = _avg([float(item.get("answer_relevancy", 0.0)) for item in results])

    report_path = settings.eval_reports_dir / f"graph_route_validation_{run['run_id']}.md"
    lines = [
        "# Graph Route Validation",
        "",
        f"- Run ID: `{run['run_id']}`",
        f"- Run Name: `{run_name}`",
        f"- Cases: `{len(results)}`",
        f"- Graph Route Rate: `{graph_rate:.4f}`",
        f"- Faithfulness: `{faithfulness:.4f}`",
        f"- Answer Relevancy: `{answer_relevancy:.4f}`",
        f"- Context Recall: `{context_recall:.4f}`",
        f"- Eval Report: `{run.get('report_path')}`",
        "",
        "## Route Counts",
        "",
        "| Route | Count |",
        "| --- | ---: |",
    ]
    for route, count in route_counts.items():
        lines.append(f"| {route} | {count} |")

    lines.extend(["", "## By Question Type", "", "| Question Type | Route | Count |", "| --- | --- | ---: |"])
    for question_type, counter in by_type.items():
        for route, count in counter.items():
            lines.append(f"| {question_type} | {route} | {count} |")

    lines.extend(["", "## Sample Cases", ""])
    for item in results[:12]:
        lines.extend(
            [
                f"### Case {item.get('idx')}",
                "",
                f"- Type: `{item.get('question_type')}`",
                f"- Route: `{item.get('route')}`",
                f"- Failure: `{item.get('failure_type')}`",
                f"- Trace ID: `{item.get('trace_id')}`",
                f"- Question: {item.get('question')}",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_id": run["run_id"],
                "graph_route_rate": round(graph_rate, 6),
                "route_counts": dict(route_counts),
                "report_path": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
