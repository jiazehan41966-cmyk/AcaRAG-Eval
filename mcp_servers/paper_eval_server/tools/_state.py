from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _state_file(name: str) -> Path:
    return _repo_root() / "data" / "state" / name


def load_state(name: str, default: Any):
    path = _state_file(name)
    if not path.exists():
        return default
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return default
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return default
