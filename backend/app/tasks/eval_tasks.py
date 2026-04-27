from __future__ import annotations

from typing import Any

from app.services.eval_service import eval_service


def run_eval_task(
    run_name: str,
    question_set: list[dict[str, Any]] | None = None,
    top_k: int = 5,
    golden_set_path: str | None = None,
    run_ragas: bool = False,
    run_deepeval: bool = False,
    baseline_name: str | None = None,
    set_as_baseline: bool = False,
    compare_with_baseline: bool = True,
) -> dict[str, Any]:
    """Batch evaluation task entrypoint.

    This function is intentionally framework-agnostic so it can be called:
    - directly in local development/tests
    - by a future Celery/RQ worker wrapper
    """

    return eval_service.run_eval(
        run_name=run_name,
        question_set=question_set or [],
        top_k=top_k,
        golden_set_path=golden_set_path,
        run_ragas=run_ragas,
        run_deepeval=run_deepeval,
        baseline_name=baseline_name,
        set_as_baseline=set_as_baseline,
        compare_with_baseline=compare_with_baseline,
    )
