"""MCP tool: get_citation_info — retrieve citation context for a paper."""

from __future__ import annotations

from typing import Any


def get_citation_info(paper_id: str) -> list[dict[str, Any]]:
    """Return citation relationships for a given paper.

    Queries the CitationEntity table (SQL mode) or scans chunks (Local mode)
    and returns citation mentions with surrounding context.
    """
    try:
        from app.services.storage_service import state_store, SQLALCHEMY_AVAILABLE

        if SQLALCHEMY_AVAILABLE and hasattr(state_store, "session"):
            return _query_sql(state_store, paper_id)
        return _query_local(state_store, paper_id)
    except Exception as exc:
        return [{"error": str(exc), "paper_id": paper_id}]


def _query_sql(store: Any, paper_id: str) -> list[dict[str, Any]]:
    """Query citation info from PostgreSQL CitationEntity table."""
    from app.services.storage_service import CitationEntity, select

    with store.session() as db:
        rows = db.execute(
            select(CitationEntity).where(CitationEntity.paper_id == paper_id)
        ).scalars().all()
        return [
            {
                "cited_paper_id": row.cited_paper_id,
                "context": row.context[:500],
                "page": (row.payload or {}).get("page"),
                "section": (row.payload or {}).get("section"),
                "evidence_chunk_id": row.evidence_chunk_id,
            }
            for row in rows
        ]


def _query_local(store: Any, paper_id: str) -> list[dict[str, Any]]:
    """Fallback: scan chunks and extract citation mentions via regex."""
    import re

    chunks = store.get_chunks(paper_id)
    if not chunks:
        return []

    bracket_re = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")
    author_year_re = re.compile(r"\b([A-Z][A-Za-z]+(?:\s+et al\.)?,\s*(?:19|20)\d{2})\b")

    results: list[dict[str, Any]] = []
    for chunk in chunks:
        mentions: list[str] = []
        for match in bracket_re.findall(chunk.text):
            mentions.extend(item.strip() for item in match.split(",") if item.strip())
        mentions.extend(author_year_re.findall(chunk.text))

        for mention in mentions[:5]:
            results.append(
                {
                    "cited_paper_id": mention,
                    "context": chunk.text[:500],
                    "page": chunk.page,
                    "section": chunk.section,
                    "evidence_chunk_id": chunk.id,
                }
            )
    return results[:50]
