import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

os.environ["FORCE_MOCK_EMBEDDING"] = "true"
os.environ["QDRANT_URL"] = "http://127.0.0.1:6333"
os.environ["QDRANT_FALLBACK_LOCAL"] = "true"
os.environ["ENABLE_RAGAS"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.main import app  # noqa: E402


client = TestClient(app)


def _ensure_index_ready() -> None:
    ready_resp = client.post("/api/index/build", json={})
    if ready_resp.status_code == 200:
        return

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
    rebuilt = client.post("/api/index/build", json={})
    assert rebuilt.status_code == 200


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


def test_eval_pipeline() -> None:
    _ensure_index_ready()

    run_resp = client.post(
        "/api/eval/run",
        json={
            "run_name": "smoke_eval",
            "top_k": 3,
            "run_ragas": False,
            "question_set": [
                {
                    "question": "论文提出了什么方法？",
                    "ground_truth": "论文提出了混合检索方法。",
                    "supporting_evidence": [],
                    "expected_citation": [],
                }
            ],
        },
    )
    assert run_resp.status_code == 200
    run_id = run_resp.json()["run_id"]

    detail_resp = client.get(f"/api/eval/runs/{run_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["status"] == "completed"
