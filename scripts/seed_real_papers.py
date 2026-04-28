#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.models.document import DocumentRecord  # noqa: E402
from app.services.parser_service import parser_service  # noqa: E402
from app.services.storage_service import state_store  # noqa: E402


PAPERS = [
    {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762.pdf",
    },
    {
        "arxiv_id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "url": "https://arxiv.org/pdf/2005.11401.pdf",
    },
    {
        "arxiv_id": "2004.04906",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "url": "https://arxiv.org/pdf/2004.04906.pdf",
    },
    {
        "arxiv_id": "2210.03629",
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "url": "https://arxiv.org/pdf/2210.03629.pdf",
    },
    {
        "arxiv_id": "2310.11511",
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "url": "https://arxiv.org/pdf/2310.11511.pdf",
    },
    {
        "arxiv_id": "2302.04761",
        "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "url": "https://arxiv.org/pdf/2302.04761.pdf",
    },
]


def _doc_id(arxiv_id: str) -> str:
    return f"arxiv_{arxiv_id.replace('.', '_')}"


def _download(url: str, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "AcaRAG-Eval/1.0"})
    with urlopen(request, timeout=90) as response:
        content = response.read()
    output.write_bytes(content)
    return len(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and parse open-access arXiv PDFs for real evidence evals")
    parser.add_argument("--max-papers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=160)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    seeded = []
    for paper in PAPERS[: args.max_papers]:
        doc_id = _doc_id(paper["arxiv_id"])
        filename = f"{doc_id}.pdf"
        path = settings.raw_dir / filename
        size_bytes = path.stat().st_size if path.exists() else 0
        if not args.skip_download and not path.exists():
            size_bytes = _download(paper["url"], path)

        record = state_store.get_document(doc_id) or DocumentRecord(
            id=doc_id,
            filename=filename,
            file_path=str(path),
            file_ext=".pdf",
            uploaded_at=datetime.now(timezone.utc),
            parse_status="uploaded",
            metadata={
                "title": paper["title"],
                "source": "arxiv",
                "source_url": paper["url"],
                "arxiv_id": paper["arxiv_id"],
                "size_bytes": size_bytes,
            },
        )
        record.file_path = str(path)
        record.metadata.update(
            {
                "title": paper["title"],
                "source": "arxiv",
                "source_url": paper["url"],
                "arxiv_id": paper["arxiv_id"],
                "size_bytes": size_bytes or path.stat().st_size,
            }
        )
        state_store.save_document(record)

        record, chunks = parser_service.parse_document(
            record,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        seeded.append(
            {
                "doc_id": doc_id,
                "title": record.metadata.get("title"),
                "pages": record.metadata.get("pages"),
                "chunks": len(chunks),
                "tables": record.metadata.get("table_count", 0),
                "layout_blocks": record.metadata.get("layout_block_count", 0),
            }
        )

    print(json.dumps({"seeded": seeded, "count": len(seeded)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
