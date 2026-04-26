from __future__ import annotations

from ._state import load_state


def query_failure_cases(failure_type: str, threshold: float = 0.0) -> list[dict]:
    runs = load_state("eval_runs.json", {})
    failure_type = (failure_type or "").strip()

    cases: list[dict] = []
    for run_id, run in runs.items():
        for row in run.get("results", []):
            if row.get("failure_type") != failure_type:
                continue
            score = float(row.get("failure_score", 0.0) or 0.0)
            if score < threshold:
                continue
            cases.append(
                {
                    "run_id": run_id,
                    "run_name": run.get("run_name"),
                    "question": row.get("question"),
                    "failure_type": row.get("failure_type"),
                    "failure_score": score,
                    "route": row.get("route"),
                }
            )
    return cases
