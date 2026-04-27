#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_runs(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig").strip()
    return json.loads(text) if text else {}


def percentile(values: list[float], q: float, default: float) -> float:
    if not values:
        return default
    return float(np.percentile(np.array(values, dtype=float), q))


def bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate failure thresholds from eval runs")
    parser.add_argument("--runs", default="data/state/eval_runs.json", help="Path to eval_runs.json")
    parser.add_argument("--output", default="backend/config/failure_thresholds.json", help="Threshold file path")
    parser.add_argument("--min-samples-per-label", type=int, default=8, help="Minimum samples required to calibrate one label")
    parser.add_argument("--write", action="store_true", help="Write calibrated thresholds to output file")
    args = parser.parse_args()

    runs = load_runs(Path(args.runs))
    all_cases = []
    for run in runs.values():
        all_cases.extend(run.get("results", []))

    unsupported = [float(c.get("faithfulness", 0.0)) for c in all_cases if c.get("failure_type") == "unsupported_claim"]
    citation = [float(c.get("citation_accuracy", 0.0)) for c in all_cases if c.get("failure_type") == "citation_error"]
    retrieval = [float(c.get("context_recall", 0.0)) for c in all_cases if c.get("failure_type") == "retrieval_miss"]
    rerank = [float(c.get("avg_rerank_score", 0.0)) for c in all_cases if c.get("failure_type") == "rerank_error"]

    unsupported_raw = percentile(unsupported, 85, 0.45) if len(unsupported) >= args.min_samples_per_label else 0.45
    citation_raw = percentile(citation, 85, 0.50) if len(citation) >= args.min_samples_per_label else 0.50
    retrieval_raw = percentile(retrieval, 85, 0.35) if len(retrieval) >= args.min_samples_per_label else 0.35
    rerank_raw = percentile(rerank, 85, 0.18) if len(rerank) >= args.min_samples_per_label else 0.18

    calibrated = {
        "unsupported_claim_max_faithfulness": round(bounded(unsupported_raw, 0.20, 0.80), 4),
        "citation_error_min_accuracy": round(bounded(citation_raw, 0.30, 0.90), 4),
        "retrieval_miss_min_context_recall": round(bounded(retrieval_raw, 0.20, 0.80), 4),
        "rerank_error_min_avg_score": round(bounded(rerank_raw, 0.05, 0.90), 4),
        "missing_reference_default_score": 1.0,
    }

    print(json.dumps({"samples": len(all_cases), "calibrated": calibrated}, ensure_ascii=False, indent=2))

    if args.write:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(calibrated, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
