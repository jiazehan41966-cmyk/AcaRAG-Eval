from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.main import app  # noqa: E402


client = TestClient(app)


def test_end_to_end_pipeline() -> None:
    sample = """# Paper Title

This paper introduces a retrieval augmented approach for academic QA.

## Method
We use hybrid retrieval with dense and sparse indexing.

## Experiments
Experiments on benchmark datasets show improved faithfulness.
"""

    files = {"file": ("sample.md", sample.encode("utf-8"), "text/markdown")}
    upload_resp = client.post("/api/documents/upload", files=files)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["doc_id"]

    parse_resp = client.post(
        f"/api/documents/{doc_id}/parse",
        json={"chunk_size": 120, "chunk_overlap": 20},
    )
    assert parse_resp.status_code == 200
    assert parse_resp.json()["chunks"] > 0

    index_resp = client.post("/api/index/build", json={})
    assert index_resp.status_code == 200

    search_resp = client.post("/api/search/hybrid", json={"query": "方法是什么", "top_k": 3})
    assert search_resp.status_code == 200
    assert len(search_resp.json()["hits"]) > 0

    qa_resp = client.post("/api/chat/ask", json={"question": "请总结论文的方法", "top_k": 3})
    assert qa_resp.status_code == 200
    assert qa_resp.json()["route"] in {"hybrid_rag", "graph_rag", "mcp_tool"}
