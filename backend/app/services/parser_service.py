from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from fastapi import HTTPException

from app.models.document import ChunkRecord, DocumentRecord
from app.services.storage_service import state_store

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None


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
                    "chunk_count": len(chunks),
                }
            )
            state_store.save_chunks(record.id, chunks)
            state_store.save_document(record)
            return record, chunks

        if record.file_ext == ".pdf":
            if fitz is None:
                raise HTTPException(
                    status_code=501,
                    detail="PDF parsing requires PyMuPDF. Please install dependency 'pymupdf'.",
                )

            page_texts: list[str] = []
            chunks: list[ChunkRecord] = []
            idx = 0

            with fitz.open(path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text = _clean_text(page.get_text("text"))
                    if not text:
                        continue
                    page_texts.append(text)
                    for snippet in _split_with_overlap(text, chunk_size, chunk_overlap):
                        chunks.append(
                            ChunkRecord(
                                id=f"{record.id}_chunk_{idx}",
                                doc_id=record.id,
                                text=snippet,
                                chunk_index=idx,
                                section=f"Page {page_num}",
                                page=page_num,
                            )
                        )
                        idx += 1

                full_text = "\n".join(page_texts)
                meta = doc.metadata or {}
                page_count = doc.page_count

            record.parse_status = "parsed"
            record.metadata.update(
                {
                    "title": meta.get("title") or _extract_title(full_text),
                    "authors": meta.get("author", ""),
                    "pages": page_count,
                    "chunk_count": len(chunks),
                }
            )
            state_store.save_chunks(record.id, chunks)
            state_store.save_document(record)
            return record, chunks

        raise HTTPException(status_code=400, detail=f"Unsupported file ext: {record.file_ext}")


parser_service = ParserService()
