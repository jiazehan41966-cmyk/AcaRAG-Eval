from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from typing import Any

from fastapi import HTTPException

from app.models.document import ChunkRecord, DocumentRecord
from app.services.storage_service import state_store

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None

try:  # pragma: no cover - optional dependency
    from unstructured.partition.pdf import partition_pdf
except Exception:  # pragma: no cover
    partition_pdf = None


def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return "Untitled"


def _split_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current_words: list[str] = []
    current_length = 0

    for word in words:
        projected = current_length + len(word) + (1 if current_words else 0)
        if projected > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            if chunk_overlap > 0:
                overlap_words: list[str] = []
                overlap_len = 0
                for prev_word in reversed(current_words):
                    add_len = len(prev_word) + (1 if overlap_words else 0)
                    if overlap_len + add_len > chunk_overlap:
                        break
                    overlap_words.insert(0, prev_word)
                    overlap_len += add_len
                current_words = overlap_words
                current_length = len(" ".join(current_words)) if current_words else 0
            else:
                current_words = []
                current_length = 0

        if current_words:
            current_length += 1
        current_words.append(word)
        current_length += len(word)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def _extract_markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = "Introduction"
    buffer: list[str] = []

    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if heading:
            if buffer:
                sections.append((current_title, "\n".join(buffer).strip()))
                buffer = []
            current_title = heading.group(2).strip()
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_title, "\n".join(buffer).strip()))

    if not sections:
        return [("Introduction", text)]
    return sections


def _build_chunks_from_sections(
    doc_id: str,
    sections: Iterable[tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    idx = 0
    for section_title, section_text in sections:
        normalized_text = _clean_text(section_text)
        for snippet in _split_with_overlap(normalized_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            chunks.append(
                ChunkRecord(
                    id=f"{doc_id}_chunk_{idx}",
                    doc_id=doc_id,
                    text=snippet,
                    chunk_index=idx,
                    section=section_title,
                    metadata={"source_type": "section", "section_title": section_title},
                )
            )
            idx += 1
    return chunks


def _span_sizes(block: dict[str, Any]) -> list[float]:
    sizes: list[float] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            size = span.get("size")
            if isinstance(size, int | float):
                sizes.append(float(size))
    return sizes


def _block_text(block: dict[str, Any]) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        pieces = [span.get("text", "") for span in line.get("spans", [])]
        text = _clean_text(" ".join(piece for piece in pieces if piece))
        if text:
            lines.append(text)
    return _clean_text(" ".join(lines))


def _looks_like_heading(text: str, max_size: float, body_size: float) -> bool:
    if not text or len(text) > 160:
        return False
    if max_size < body_size + 1.0:
        return False
    if re.match(r"^(\d+(\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9 ,:/()&\-]{3,}$", text):
        return True
    if re.match(r"^(abstract|introduction|related work|method|methods|experiments|results|conclusion|references)$", text.lower()):
        return True
    return False


def _extract_page_tables(page: Any) -> list[dict[str, Any]]:
    if not hasattr(page, "find_tables"):
        return []
    try:
        finder = page.find_tables()
    except Exception:
        return []

    tables: list[dict[str, Any]] = []
    for idx, table in enumerate(getattr(finder, "tables", []) or []):
        try:
            rows = table.extract()
        except Exception:
            rows = []
        tables.append(
            {
                "table_index": idx,
                "bbox": [round(float(v), 2) for v in getattr(table, "bbox", [])],
                "rows": len(rows),
                "cols": max((len(row) for row in rows), default=0),
                "preview": rows[:3],
            }
        )
    return tables


def _extract_unstructured_preview(path: Path) -> dict[str, Any]:
    if partition_pdf is None:
        return {"enabled": False, "reason": "unstructured_not_installed"}
    try:
        elements = partition_pdf(filename=str(path), strategy="fast")
    except Exception as exc:  # pragma: no cover - optional runtime
        return {"enabled": True, "error": str(exc), "elements": 0}
    return {
        "enabled": True,
        "elements": len(elements),
        "types": sorted({type(item).__name__ for item in elements}),
    }


def _extract_pdf_layout(path: Path) -> dict[str, Any]:
    if fitz is None:
        raise HTTPException(
            status_code=501,
            detail="PDF parsing requires PyMuPDF. Please install dependency 'pymupdf'.",
        )

    pages: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    page_texts: list[str] = []
    image_count = 0
    current_section = "Introduction"

    with fitz.open(path) as doc:
        meta = doc.metadata or {}
        page_count = doc.page_count

        for page_num, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            raw_blocks = page_dict.get("blocks", [])
            text_blocks: list[dict[str, Any]] = []
            sizes: list[float] = []

            for raw_block in raw_blocks:
                if raw_block.get("type") == 1:
                    image_count += 1
                    continue
                text = _block_text(raw_block)
                if not text:
                    continue
                block_sizes = _span_sizes(raw_block)
                sizes.extend(block_sizes)
                text_blocks.append(
                    {
                        "page": page_num,
                        "block_id": f"p{page_num}_b{len(text_blocks)}",
                        "text": text,
                        "bbox": [round(float(v), 2) for v in raw_block.get("bbox", [])],
                        "max_font_size": max(block_sizes, default=0.0),
                        "avg_font_size": round(sum(block_sizes) / len(block_sizes), 2) if block_sizes else 0.0,
                    }
                )

            body_size = median(sizes) if sizes else 10.0
            page_texts.append("\n".join(item["text"] for item in text_blocks))
            page_tables = _extract_page_tables(page)
            tables.extend({"page": page_num, **item} for item in page_tables)

            for item in text_blocks:
                if _looks_like_heading(item["text"], float(item["max_font_size"]), float(body_size)):
                    current_section = item["text"][:120]
                    sections.append(
                        {
                            "title": current_section,
                            "page": page_num,
                            "level": 1,
                            "block_id": item["block_id"],
                            "bbox": item["bbox"],
                        }
                    )
                item["section"] = current_section
                blocks.append(item)

            pages.append(
                {
                    "page": page_num,
                    "text_blocks": len(text_blocks),
                    "tables": len(page_tables),
                    "images": sum(1 for block in raw_blocks if block.get("type") == 1),
                }
            )

    full_text = "\n".join(page_texts)
    if not sections:
        sections = [{"title": "Introduction", "page": 1, "level": 1, "block_id": None, "bbox": []}]

    return {
        "metadata": meta,
        "page_count": page_count,
        "full_text": full_text,
        "pages": pages,
        "blocks": blocks,
        "tables": tables,
        "image_count": image_count,
        "document_structure": {
            "type": "section_tree",
            "sections": sections,
            "pages": pages,
        },
        "unstructured": _extract_unstructured_preview(path),
    }


def _build_pdf_chunks(
    doc_id: str,
    blocks: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    idx = 0
    for block in blocks:
        for snippet in _split_with_overlap(block["text"], chunk_size, chunk_overlap):
            chunks.append(
                ChunkRecord(
                    id=f"{doc_id}_chunk_{idx}",
                    doc_id=doc_id,
                    text=snippet,
                    chunk_index=idx,
                    section=block.get("section") or f"Page {block['page']}",
                    page=int(block["page"]),
                    metadata={
                        "source_type": "pymupdf_block",
                        "block_id": block["block_id"],
                        "bbox": block["bbox"],
                        "avg_font_size": block["avg_font_size"],
                        "max_font_size": block["max_font_size"],
                    },
                )
            )
            idx += 1
    return chunks


class ParserService:
    def parse_document(
        self,
        record: DocumentRecord,
        chunk_size: int,
        chunk_overlap: int,
    ) -> tuple[DocumentRecord, list[ChunkRecord]]:
        path = Path(record.file_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Document file missing")

        if record.file_ext in {".txt", ".md"}:
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            sections = _extract_markdown_sections(raw_text) if record.file_ext == ".md" else [("Body", raw_text)]
            chunks = _build_chunks_from_sections(record.id, sections, chunk_size, chunk_overlap)
            title = _extract_title(raw_text)
            record.parse_status = "parsed"
            record.metadata.update(
                {
                    "title": title,
                    "sections": len(sections),
                    "document_structure": {
                        "type": "markdown_sections" if record.file_ext == ".md" else "plain_text",
                        "sections": [{"title": title, "level": 1, "index": idx} for idx, (title, _) in enumerate(sections)],
                    },
                    "chunk_count": len(chunks),
                }
            )
            state_store.save_chunks(record.id, chunks)
            state_store.save_document(record)
            return record, chunks

        if record.file_ext == ".pdf":
            layout = _extract_pdf_layout(path)
            chunks = _build_pdf_chunks(record.id, layout["blocks"], chunk_size, chunk_overlap)
            meta = layout["metadata"]
            record.parse_status = "parsed"
            record.metadata.update(
                {
                    "title": record.metadata.get("title") or meta.get("title") or _extract_title(layout["full_text"]),
                    "authors": meta.get("author", ""),
                    "pages": layout["page_count"],
                    "chunk_count": len(chunks),
                    "layout_block_count": len(layout["blocks"]),
                    "table_count": len(layout["tables"]),
                    "image_count": layout["image_count"],
                    "document_structure": layout["document_structure"],
                    "tables": layout["tables"][:20],
                    "unstructured": layout["unstructured"],
                }
            )
            state_store.save_chunks(record.id, chunks)
            state_store.save_document(record)
            return record, chunks

        raise HTTPException(status_code=400, detail=f"Unsupported file ext: {record.file_ext}")


parser_service = ParserService()
