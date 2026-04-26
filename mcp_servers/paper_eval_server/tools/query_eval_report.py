from __future__ import annotations

from ._state import load_state


def query_eval_report(run_id: str) -> dict:
    runs = load_state("eval_runs.json", {})
    payload = runs.get(run_id)
    if not payload:
        return {"run_id": run_id, "found": False, "message": "eval run not found"}

    return {
        "run_id": run_id,
        "found": True,
        "run_name": payload.get("run_name"),
        "status": payload.get("status"),
        "cases": payload.get("cases"),
        "metrics": payload.get("metrics", {}),
        "failure_summary": payload.get("failure_summary", {}),
        "report_path": payload.get("report_path"),
        "regression_report_path": payload.get("regression_report_path"),
        "created_at": payload.get("created_at"),
    }
