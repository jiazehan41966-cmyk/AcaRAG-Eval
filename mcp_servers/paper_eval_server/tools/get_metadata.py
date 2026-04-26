from __future__ import annotations

from ._state import load_state


def get_paper_metadata(paper_id: str) -> dict:
    docs = load_state("documents.json", {})
    payload = docs.get(paper_id)
    if not payload:
        return {"paper_id": paper_id, "found": False, "message": "paper not found"}

    metadata = payload.get("metadata", {})
    return {
        "paper_id": paper_id,
        "found": True,
        "title": metadata.get("title") or payload.get("filename"),
        "authors": metadata.get("authors"),
        "pages": metadata.get("pages"),
        "chunk_count": metadata.get("chunk_count"),
        "uploaded_at": payload.get("uploaded_at"),
        "parse_status": payload.get("parse_status"),
        "metadata": metadata,
    }
