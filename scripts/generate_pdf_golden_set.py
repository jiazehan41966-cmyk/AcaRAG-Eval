#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.storage_service import state_store  # noqa: E402


def _sentence(text: str, max_chars: int = 380) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    selected = " ".join(parts[:2]).strip() or text
    return selected[:max_chars]


def _keyword(text: str) -> str:
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9\-]{4,}", text)
    for item in candidates:
        lowered = item.lower()
        if lowered not in {"table", "figure", "section", "paper", "model", "method", "results"}:
            return item
    return "the method"


def _query_phrase(text: str, max_chars: int = 140) -> str:
    phrase = _sentence(text, max_chars=max_chars)
    phrase = re.sub(r"[^A-Za-z0-9 ,;:\-()]", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase[:max_chars] or _keyword(text)


def _pdf_chunks() -> list[dict[str, Any]]:
    docs = [doc for doc in state_store.list_documents() if doc.file_ext == ".pdf" and doc.parse_status == "parsed"]
    rows: list[dict[str, Any]] = []
    for doc in docs:
        for chunk in state_store.get_chunks(doc.id):
            if chunk.page is None or len(chunk.text) < 180:
                continue
            rows.append(
                {
                    "doc_id": doc.id,
                    "title": doc.metadata.get("title") or doc.filename,
                    "chunk_id": chunk.id,
                    "page": chunk.page,
                    "section": chunk.section or f"Page {chunk.page}",
                    "text": chunk.text,
                }
            )
    return rows


def _fact_case(row: dict[str, Any], difficulty: str) -> dict[str, Any]:
    keyword = _keyword(row["text"])
    phrase = _query_phrase(row["text"])
    return {
        "question": f"According to '{row['title']}', what does the paper say about '{keyword}' in this evidence: \"{phrase}\"?",
        "ground_truth": _sentence(row["text"]),
        "supporting_evidence": [row["chunk_id"]],
        "expected_citation": [f"{row['doc_id']}:{row['page']}"],
        "difficulty": difficulty,
        "question_type": "fact",
        "failure_label": "",
    }


def _metadata_case(row: dict[str, Any], difficulty: str) -> dict[str, Any]:
    keyword = _keyword(row["text"])
    phrase = _query_phrase(row["text"])
    return {
        "question": f"Which paper and page contain supporting evidence about '{keyword}' with the phrase \"{phrase}\"?",
        "ground_truth": f"The evidence appears in '{row['title']}' on page {row['page']}: {_sentence(row['text'], 240)}",
        "supporting_evidence": [row["chunk_id"]],
        "expected_citation": [f"{row['doc_id']}:{row['page']}"],
        "difficulty": difficulty,
        "question_type": "metadata",
        "failure_label": "",
    }


def _comparison_case(left: dict[str, Any], right: dict[str, Any], difficulty: str) -> dict[str, Any]:
    return {
        "question": f"Compare evidence from '{left['title']}' about \"{_query_phrase(left['text'], 90)}\" with evidence from '{right['title']}' about \"{_query_phrase(right['text'], 90)}\".",
        "ground_truth": f"{_sentence(left['text'], 220)} In comparison, {_sentence(right['text'], 220)}",
        "supporting_evidence": [left["chunk_id"], right["chunk_id"]],
        "expected_citation": [f"{left['doc_id']}:{left['page']}", f"{right['doc_id']}:{right['page']}"],
        "difficulty": difficulty,
        "question_type": "comparison",
        "failure_label": "",
    }


def _multihop_case(first: dict[str, Any], second: dict[str, Any], difficulty: str) -> dict[str, Any]:
    return {
        "question": f"Use evidence from '{first['title']}' mentioning \"{_query_phrase(first['text'], 90)}\" and '{second['title']}' mentioning \"{_query_phrase(second['text'], 90)}\" to answer the question.",
        "ground_truth": f"{_sentence(first['text'], 220)} This connects with {_sentence(second['text'], 220)}",
        "supporting_evidence": [first["chunk_id"], second["chunk_id"]],
        "expected_citation": [f"{first['doc_id']}:{first['page']}", f"{second['doc_id']}:{second['page']}"],
        "difficulty": difficulty,
        "question_type": "multi-hop",
        "failure_label": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Golden Set rows grounded in parsed PDF chunk/page IDs")
    parser.add_argument("--output", default="data/golden_set/pdf_golden_set.jsonl")
    parser.add_argument("--max-cases", type=int, default=80)
    args = parser.parse_args()

    rows = _pdf_chunks()
    if len(rows) < 8:
        raise SystemExit("not enough parsed PDF chunks with pages; run scripts/seed_real_papers.py first")

    cases: list[dict[str, Any]] = []
    for idx, row in enumerate(rows[: max(args.max_cases // 2, 1)]):
        cases.append(_fact_case(row, "easy" if idx % 3 == 0 else "medium"))
        if len(cases) < args.max_cases:
            cases.append(_metadata_case(row, "easy"))
        if len(cases) >= args.max_cases:
            break

    left_right = list(zip(rows[::2], rows[1::2]))
    for idx, (left, right) in enumerate(left_right):
        if left["doc_id"] == right["doc_id"]:
            continue
        if len(cases) < args.max_cases:
            cases.append(_comparison_case(left, right, "medium" if idx % 2 == 0 else "hard"))
        if len(cases) < args.max_cases:
            cases.append(_multihop_case(left, right, "hard"))
        if len(cases) >= args.max_cases:
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in cases[: args.max_cases]) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": len(cases[: args.max_cases])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
