from __future__ import annotations

import re
from urllib.parse import quote
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from ._state import load_state


def _search_arxiv(keyword: str, top_k: int) -> list[dict]:
    if top_k <= 0:
        return []
    url = f"https://export.arxiv.org/api/query?search_query=all:{quote(keyword)}&start=0&max_results={top_k}"
    try:
        request = Request(url, headers={"User-Agent": "AcaRAG-Eval/1.0"})
        with urlopen(request, timeout=20) as response:
            payload = response.read()
    except Exception:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ElementTree.fromstring(payload)
    results: list[dict] = []
    for entry in root.findall("atom:entry", ns):
        arxiv_url = entry.findtext("atom:id", default="", namespaces=ns)
        arxiv_id_match = re.search(r"abs/([^/]+)$", arxiv_url)
        arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else arxiv_url.rsplit("/", 1)[-1]
        authors = [
            item.findtext("atom:name", default="", namespaces=ns)
            for item in entry.findall("atom:author", ns)
        ]
        results.append(
            {
                "doc_id": f"arxiv:{arxiv_id}",
                "title": (entry.findtext("atom:title", default="", namespaces=ns) or "").replace("\n", " ").strip(),
                "authors": ", ".join(author for author in authors if author),
                "uploaded_at": None,
                "parse_status": "external",
                "source": "arxiv",
                "source_url": arxiv_url,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None,
            }
        )
    return results


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
                    "source": "local",
                    "source_url": metadata.get("source_url"),
                }
            )

    if len(results) < top_k:
        results.extend(_search_arxiv(keyword, top_k - len(results)))

    return results[:top_k]
