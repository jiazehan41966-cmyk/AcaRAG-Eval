#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.storage_service import state_store  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def validate(path: Path) -> dict[str, Any]:
    chunks = state_store.list_chunks()
    real_chunk_ids = {chunk.id for chunk in chunks}
    real_citations = {
        f"{chunk.doc_id}:{chunk.page}"
        for chunk in chunks
        if chunk.doc_id and chunk.page is not None
    }
    rows = _load_jsonl(path)
    stats = {
        "supporting_evidence": {"empty": 0, "real": 0, "placeholder": 0},
        "expected_citation": {"empty": 0, "real": 0, "placeholder": 0},
    }
    examples = {"supporting_evidence": [], "expected_citation": []}

    for idx, row in enumerate(rows, start=1):
        supports = set(row.get("supporting_evidence") or [])
        citations = set(row.get("expected_citation") or [])

        if not supports:
            stats["supporting_evidence"]["empty"] += 1
        elif supports & real_chunk_ids:
            stats["supporting_evidence"]["real"] += 1
        else:
            stats["supporting_evidence"]["placeholder"] += 1
            if len(examples["supporting_evidence"]) < 5:
                examples["supporting_evidence"].append({"idx": idx, "values": sorted(supports)})

        if not citations:
            stats["expected_citation"]["empty"] += 1
        elif citations & real_citations:
            stats["expected_citation"]["real"] += 1
        else:
            stats["expected_citation"]["placeholder"] += 1
            if len(examples["expected_citation"]) < 5:
                examples["expected_citation"].append({"idx": idx, "values": sorted(citations)})

    return {
        "path": str(path),
        "cases": len(rows),
        "chunks": len(chunks),
        "real_chunk_ids": len(real_chunk_ids),
        "real_citations": len(real_citations),
        "stats": stats,
        "placeholder_examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate golden set references against parsed chunks")
    parser.add_argument("--input", default="data/golden_set/golden_set.jsonl")
    parser.add_argument("--fail-on-placeholder", action="store_true")
    args = parser.parse_args()

    summary = validate(Path(args.input))
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    placeholders = (
        summary["stats"]["supporting_evidence"]["placeholder"]
        + summary["stats"]["expected_citation"]["placeholder"]
    )
    if args.fail_on_placeholder and placeholders:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
