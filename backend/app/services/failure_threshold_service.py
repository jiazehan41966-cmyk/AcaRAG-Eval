from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings


class FailureThresholdService:
    DEFAULTS = {
        "unsupported_claim_max_faithfulness": 0.45,
        "citation_error_min_accuracy": 0.5,
        "retrieval_miss_min_context_recall": 0.35,
        "rerank_error_min_avg_score": 0.18,
        "missing_reference_default_score": 1.0,
    }

    def __init__(self) -> None:
        self.settings = get_settings()
        self.path = Path(self.settings.failure_thresholds_path)
        self._ensure_file()

    def _ensure_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps(self.DEFAULTS, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> dict[str, Any]:
        self._ensure_file()
        try:
            content = self.path.read_text(encoding="utf-8-sig").strip()
            data = json.loads(content) if content else {}
            merged = {**self.DEFAULTS, **data}
            return merged
        except Exception:
            return dict(self.DEFAULTS)


failure_threshold_service = FailureThresholdService()
