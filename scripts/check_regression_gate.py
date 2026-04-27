#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path, default):
    if not path.exists():
        return default
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return default
    return json.loads(text)


def fail(msg: str) -> int:
    print(f"[REGRESSION_GATE] FAIL: {msg}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Regression gate checker")
    parser.add_argument("--runs", default="data/state/eval_runs.json")
    parser.add_argument("--gate", default="backend/config/regression_gate.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()

    runs = load_json(Path(args.runs), {})
    if not runs:
        return fail("no eval runs found")

    if args.run_id:
        current = runs.get(args.run_id)
        if current is None:
            return fail(f"run_id not found: {args.run_id}")
    elif args.latest:
        items = sorted(runs.values(), key=lambda x: x.get("created_at", ""), reverse=True)
        current = next((item for item in items if item.get("baseline_comparison")), None)
        if current is None:
            return fail("no run with baseline_comparison found")
    else:
        return fail("please provide --run-id or --latest")

    comparison = current.get("baseline_comparison")
    if not comparison:
        return fail("current run has no baseline_comparison")
    baseline_run = runs.get(comparison.get("baseline_run_id")) if comparison.get("baseline_run_id") else None

    gate = load_json(Path(args.gate), {})
    metric_drop_thresholds = gate.get("metric_drop_thresholds", {})
    channel_drop_thresholds = gate.get("channel_drop_thresholds", {})
    max_failure_increase = gate.get("max_failure_increase", {})

    errors: list[str] = []

    for metric, threshold in metric_drop_thresholds.items():
        delta = comparison.get("metric_diff", {}).get(metric, {}).get("delta")
        if isinstance(delta, (int, float)) and delta < -float(threshold):
            errors.append(f"metric `{metric}` delta={delta:.6f} < -{threshold}")

    channel_diff = comparison.get("channel_diff", {})
    for channel, threshold in channel_drop_thresholds.items():
        if channel == "blended_overall":
            baseline = (
                baseline_run.get("channels", {}).get("blended_overall")
                if isinstance(baseline_run, dict)
                else None
            )
            if not isinstance(baseline, (int, float)):
                baseline = comparison.get("baseline_channels", {}).get("blended_overall")
            curr = current.get("channels", {}).get("blended_overall")
            if isinstance(curr, (int, float)) and isinstance(baseline, (int, float)):
                delta = curr - baseline
                if delta < -float(threshold):
                    errors.append(f"channel `{channel}` delta={delta:.6f} < -{threshold}")
            continue

        delta = channel_diff.get(channel, {}).get("delta")
        if isinstance(delta, (int, float)) and delta < -float(threshold):
            errors.append(f"channel `{channel}` delta={delta:.6f} < -{threshold}")

    for failure_type, max_delta in max_failure_increase.items():
        delta = comparison.get("failure_diff", {}).get(failure_type, {}).get("delta")
        if isinstance(delta, int) and delta > int(max_delta):
            errors.append(f"failure `{failure_type}` delta=+{delta} > {max_delta}")

    if errors:
        print("[REGRESSION_GATE] FAIL details:")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"[REGRESSION_GATE] PASS run_id={current.get('run_id')} baseline={comparison.get('baseline_run_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
