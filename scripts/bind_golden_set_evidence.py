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


CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
TOKEN_RE = re.compile(r"[a-zA-Z0-9_\-]+|[\u4e00-\u9fff]+")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "what",
    "how",
    "why",
    "which",
    "论文",
    "系统",
    "什么",
    "如何",
    "以及",
    "这个",
    "该",
    "的",
    "了",
    "和",
    "是",
    "在",
    "中",
    "与",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _tokens(text: str) -> set[str]:
    output: set[str] = set()
    for token in TOKEN_RE.findall((text or "").lower()):
        if token in STOPWORDS:
            continue
        if CJK_RE.fullmatch(token):
            if len(token) <= 2:
                if token not in STOPWORDS:
                    output.add(token)
                continue
            for i in range(len(token) - 1):
                gram = token[i : i + 2]
                if gram not in STOPWORDS:
                    output.add(gram)
            continue
        if len(token) >= 2 and token not in STOPWORDS:
            output.add(token)
    return output


def _score(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0
    overlap = len(query_tokens & chunk_tokens)
    recall = overlap / len(query_tokens)
    precision = overlap / len(chunk_tokens)
    return (0.75 * recall) + (0.25 * precision)


def _reference_stats(rows: list[dict[str, Any]], real_chunk_ids: set[str], real_citations: set[str]) -> dict[str, dict]:
    support = {"empty": 0, "real": 0, "placeholder": 0}
    citation = {"empty": 0, "real": 0, "placeholder": 0}
    for row in rows:
        supports = set(row.get("supporting_evidence") or [])
        citations = set(row.get("expected_citation") or [])

        if not supports:
            support["empty"] += 1
        elif supports & real_chunk_ids:
            support["real"] += 1
        else:
            support["placeholder"] += 1

        if not citations:
            citation["empty"] += 1
        elif citations & real_citations:
            citation["real"] += 1
        else:
            citation["placeholder"] += 1

    return {"supporting_evidence": support, "expected_citation": citation}


def _load_chunk_payloads() -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for chunk in state_store.list_chunks():
        row = chunk.model_dump(mode="json")
        row["_tokens"] = _tokens(" ".join([row.get("section") or "", row.get("text") or ""]))
        payloads.append(row)
    return payloads


def bind_cases(
    rows: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    max_evidence: int,
    min_score: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    real_chunk_ids = {chunk["id"] for chunk in chunks}
    real_citations = {
        f"{chunk.get('doc_id')}:{chunk.get('page')}"
        for chunk in chunks
        if chunk.get("doc_id") and chunk.get("page") is not None
    }

    before = _reference_stats(rows, real_chunk_ids, real_citations)
    bound = 0
    no_match = 0
    no_page = 0
    output: list[dict[str, Any]] = []

    for row in rows:
        query = " ".join([str(row.get("question") or ""), str(row.get("ground_truth") or "")])
        query_tokens = _tokens(query)
        scored = []
        for chunk in chunks:
            score = _score(query_tokens=query_tokens, chunk_tokens=chunk.get("_tokens", set()))
            if score >= min_score:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [chunk for _, chunk in scored[:max_evidence]]

        new_row = dict(row)
        if selected:
            new_row["supporting_evidence"] = [chunk["id"] for chunk in selected]
            citations = [
                f"{chunk.get('doc_id')}:{chunk.get('page')}"
                for chunk in selected
                if chunk.get("doc_id") and chunk.get("page") is not None
            ]
            new_row["expected_citation"] = citations[:1]
            if not citations:
                no_page += 1
            bound += 1
        else:
            new_row["supporting_evidence"] = []
            new_row["expected_citation"] = []
            no_match += 1

        output.append(new_row)

    after = _reference_stats(output, real_chunk_ids, real_citations)
    summary = {
        "cases": len(rows),
        "chunks": len(chunks),
        "real_chunk_ids": len(real_chunk_ids),
        "real_citations": len(real_citations),
        "bound_cases": bound,
        "no_match_cases": no_match,
        "bound_without_page_citation": no_page,
        "before": before,
        "after": after,
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Bind golden set evidence fields to real parsed chunk ids")
    parser.add_argument("--input", default="data/golden_set/golden_set.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-evidence", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.08)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    rows = _load_jsonl(input_path)
    chunks = _load_chunk_payloads()
    if not rows:
        raise SystemExit(f"no rows found: {input_path}")
    if not chunks:
        raise SystemExit("no parsed chunks found; upload and parse documents before binding evidence")

    bound_rows, summary = bind_cases(
        rows=rows,
        chunks=chunks,
        max_evidence=max(1, args.max_evidence),
        min_score=max(0.0, args.min_score),
    )
    _write_jsonl(output_path, bound_rows)

    summary["input"] = str(input_path)
    summary["output"] = str(output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
