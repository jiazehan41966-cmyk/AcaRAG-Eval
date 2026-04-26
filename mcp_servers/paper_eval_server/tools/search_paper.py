from __future__ import annotations

from ._state import load_state


def search_paper(keyword: str, top_k: int = 5) -> list[dict]:
    keyword = (keyword or "").strip().lower()
    docs = load_state("documents.json", {})
    if not keyword:
        return []

    results: list[dict] = []
    for doc_id, payload in docs.items():
        metadata = payload.get("metadata", {})
        title = str(metadata.get("title") or payload.get("filename") or "")
        authors = str(metadata.get("authors") or "")
        haystack = f"{title} {authors} {payload.get('filename', '')}".lower()
        if keyword in haystack:
            results.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "authors": authors,
                    "uploaded_at": payload.get("uploaded_at"),
                    "parse_status": payload.get("parse_status"),
                }
            )

    return results[:top_k]
