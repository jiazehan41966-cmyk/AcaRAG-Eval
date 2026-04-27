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


METRICS = ("faithfulness", "answer_relevancy", "question_coverage", "context_recall", "citation_accuracy")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _forced_cases(cases: list[dict[str, Any]], route: str) -> list[dict[str, Any]]:
    forced: list[dict[str, Any]] = []
    for case in cases:
        item = dict(case)
        item["force_route"] = route
        forced.append(item)
    return forced


def _run_eval(
    client: TestClient,
    *,
    run_name: str,
    cases: list[dict[str, Any]],
    top_k: int,
    run_ragas: bool,
    run_deepeval: bool,
) -> dict[str, Any]:
    response = client.post(
        "/api/eval/run",
        json={
            "run_name": run_name,
            "question_set": cases,
            "top_k": top_k,
            "run_ragas": run_ragas,
            "run_deepeval": run_deepeval,
            "compare_with_baseline": False,
        },
    )
    if response.status_code != 200:
        raise SystemExit(f"eval run failed: status={response.status_code}, body={response.text}")
    run_id = response.json()["run_id"]
    detail_response = client.get(f"/api/eval/runs/{run_id}")
    if detail_response.status_code != 200:
        raise SystemExit(f"eval detail failed: status={detail_response.status_code}, body={detail_response.text}")
    return detail_response.json()


def _route_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(item.get("route", "unknown")) for item in results))


def _by_type_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[str(item.get("question_type", "unknown"))].append(item)

    output: dict[str, dict[str, float]] = {}
    for question_type, items in grouped.items():
        output[question_type] = {
            metric: _avg([float(item.get(metric, 0.0)) for item in items])
            for metric in METRICS
        }
    return output


def _metric_diff(graph: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, dict[str, float]]:
    diff: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        graph_value = float(graph.get("metrics", {}).get(metric, 0.0))
        hybrid_value = float(hybrid.get("metrics", {}).get(metric, 0.0))
        diff[metric] = {
            "hybrid": round(hybrid_value, 6),
            "graph": round(graph_value, 6),
            "delta": round(graph_value - hybrid_value, 6),
        }
    return diff


def _channel_diff(graph: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, float | None]:
    graph_overall = graph.get("channels", {}).get("blended_overall")
    hybrid_overall = hybrid.get("channels", {}).get("blended_overall")
    if graph_overall is None or hybrid_overall is None:
        return {"hybrid": hybrid_overall, "graph": graph_overall, "delta": None}
    return {
        "hybrid": round(float(hybrid_overall), 6),
        "graph": round(float(graph_overall), 6),
        "delta": round(float(graph_overall) - float(hybrid_overall), 6),
    }


def _write_report(
    report_path: Path,
    *,
    run_name: str,
    hybrid: dict[str, Any],
    graph: dict[str, Any],
    diff: dict[str, dict[str, float]],
    blended_diff: dict[str, float | None],
    top_k: int,
    cases: list[dict[str, Any]],
) -> None:
    hybrid_results = hybrid.get("results", [])
    graph_results = graph.get("results", [])
    hybrid_by_type = _by_type_metrics(hybrid_results)
    graph_by_type = _by_type_metrics(graph_results)

    lines = [
        "# Graph Retrieval Ablation",
        "",
        f"- Generated At (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- Run Name: `{run_name}`",
        f"- Cases: `{len(cases)}`",
        f"- Top K: `{top_k}`",
        f"- Hybrid Run ID: `{hybrid.get('run_id')}`",
        f"- Graph Run ID: `{graph.get('run_id')}`",
        f"- Hybrid Report: `{hybrid.get('report_path')}`",
        f"- Graph Report: `{graph.get('report_path')}`",
        "",
        "## Route Counts",
        "",
        f"- Hybrid forced route counts: `{json.dumps(_route_counts(hybrid_results), ensure_ascii=False)}`",
        f"- Graph forced route counts: `{json.dumps(_route_counts(graph_results), ensure_ascii=False)}`",
        "",
        "## Overall Metric Diff",
        "",
        "| Metric | Hybrid | Graph | Graph - Hybrid |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, values in diff.items():
        lines.append(f"| {metric} | {values['hybrid']:.4f} | {values['graph']:.4f} | {values['delta']:+.4f} |")

    lines.extend(
        [
            f"| blended_overall | {blended_diff['hybrid']} | {blended_diff['graph']} | {blended_diff['delta']} |",
            "",
            "## By Question Type",
            "",
            "| Question Type | Metric | Hybrid | Graph | Graph - Hybrid |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for question_type in sorted(set(hybrid_by_type) | set(graph_by_type)):
        for metric in METRICS:
            hybrid_value = float(hybrid_by_type.get(question_type, {}).get(metric, 0.0))
            graph_value = float(graph_by_type.get(question_type, {}).get(metric, 0.0))
            lines.append(
                f"| {question_type} | {metric} | {hybrid_value:.4f} | {graph_value:.4f} | "
                f"{graph_value - hybrid_value:+.4f} |"
            )

    lines.extend(["", "## Sample Case Diff", ""])
    for hybrid_item, graph_item in list(zip(hybrid_results, graph_results))[:12]:
        lines.extend(
            [
                f"### Case {hybrid_item.get('idx')}",
                "",
                f"- Type: `{hybrid_item.get('question_type')}`",
                f"- Question: {hybrid_item.get('question')}",
                f"- Hybrid failure: `{hybrid_item.get('failure_type')}`, trace: `{hybrid_item.get('trace_id')}`",
                f"- Graph failure: `{graph_item.get('failure_type')}`, trace: `{graph_item.get('trace_id')}`",
                f"- Context recall delta: `{float(graph_item.get('context_recall', 0.0)) - float(hybrid_item.get('context_recall', 0.0)):+.4f}`",
                f"- Faithfulness delta: `{float(graph_item.get('faithfulness', 0.0)) - float(hybrid_item.get('faithfulness', 0.0)):+.4f}`",
                "",
            ]
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare forced hybrid_rag vs forced graph_rag on graph eval cases")
    parser.add_argument("--golden-set", default="data/golden_set/graph_eval_set.jsonl")
    parser.add_argument("--run-name", default="graph_ablation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=32)
    parser.add_argument("--run-ragas", action="store_true")
    parser.add_argument("--run-deepeval", action="store_true")
    args = parser.parse_args()

    cases = _load_jsonl(Path(args.golden_set))
    cases = [
        case for case in cases
        if case.get("question_type") in {"comparison", "multi-hop"}
    ]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit(f"no comparison/multi-hop cases found: {args.golden_set}")

    client = TestClient(app)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.run_name}_{timestamp}"

    hybrid = _run_eval(
        client,
        run_name=f"{run_name}_hybrid",
        cases=_forced_cases(cases, "hybrid_rag"),
        top_k=args.top_k,
        run_ragas=args.run_ragas,
        run_deepeval=args.run_deepeval,
    )
    graph = _run_eval(
        client,
        run_name=f"{run_name}_graph",
        cases=_forced_cases(cases, "graph_rag"),
        top_k=args.top_k,
        run_ragas=args.run_ragas,
        run_deepeval=args.run_deepeval,
    )

    diff = _metric_diff(graph, hybrid)
    blended_diff = _channel_diff(graph, hybrid)
    settings = get_settings()
    report_path = settings.eval_reports_dir / f"graph_ablation_{hybrid['run_id']}_vs_{graph['run_id']}.md"
    _write_report(
        report_path,
        run_name=run_name,
        hybrid=hybrid,
        graph=graph,
        diff=diff,
        blended_diff=blended_diff,
        top_k=args.top_k,
        cases=cases,
    )

    print(
        json.dumps(
            {
                "hybrid_run_id": hybrid["run_id"],
                "graph_run_id": graph["run_id"],
                "cases": len(cases),
                "metric_diff": diff,
                "blended_overall_diff": blended_diff,
                "report_path": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
