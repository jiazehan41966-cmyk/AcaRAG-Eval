import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

os.environ["FORCE_MOCK_EMBEDDING"] = "true"
os.environ["QDRANT_URL"] = "http://127.0.0.1:6333"
os.environ["QDRANT_FALLBACK_LOCAL"] = "true"
os.environ["ENABLE_RAGAS"] = "false"
os.environ["ENABLE_DEEPEVAL"] = "false"

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
    assert qa_resp.json().get("trace_id")

    trace_resp = client.get(f"/api/traces/{qa_resp.json()['trace_id']}")
    assert trace_resp.status_code == 200
    assert trace_resp.json().get("trace_id") == qa_resp.json()["trace_id"]


def test_eval_pipeline() -> None:
    _ensure_index_ready()

    baseline_resp = client.post(
        "/api/eval/run",
        json={
            "run_name": "smoke_eval_baseline",
            "top_k": 3,
            "run_ragas": False,
            "run_deepeval": False,
            "set_as_baseline": True,
            "baseline_name": "smoke_main",
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
    assert baseline_resp.status_code == 200

    run_resp = client.post(
        "/api/eval/run",
        json={
            "run_name": "smoke_eval_compare",
            "top_k": 3,
            "run_ragas": False,
            "run_deepeval": False,
            "baseline_name": "smoke_main",
            "compare_with_baseline": True,
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
    run_payload = run_resp.json()
    run_id = run_payload["run_id"]
    assert run_payload.get("regression_report_path") is not None

    detail_resp = client.get(f"/api/eval/runs/{run_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["status"] == "completed"
    assert "failure_summary" in detail_resp.json()

    list_resp = client.get("/api/eval/runs")
    assert list_resp.status_code == 200
    assert isinstance(list_resp.json(), list)
