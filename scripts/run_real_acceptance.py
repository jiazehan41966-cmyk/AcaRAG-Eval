#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _short(v: Any, default: str = "-") -> str:
    if v is None:
        return default
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _draw_snapshot(image_path: Path, lines: list[str]) -> None:
    width = 1500
    height = max(900, 36 * (len(lines) + 2))
    image = Image.new("RGB", (width, height), color=(244, 248, 245))
    draw = ImageDraw.Draw(image)

    y = 24
    draw.rectangle((16, 16, width - 16, height - 16), outline=(170, 193, 181), width=2)
    for i, line in enumerate(lines):
        fill = (20, 41, 31) if i > 0 else (18, 93, 65)
        draw.text((36, y), line, fill=fill)
        y += 32

    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real-environment acceptance for OPENAI + RAGAS + DeepEval + Langfuse")
    parser.add_argument("--run-name", default="real_acceptance")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--golden-set", default="data/golden_set/golden_set.jsonl")
    parser.add_argument("--max-cases", type=int, default=20)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    backend_dir = repo_root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    # If key exists but provider isn't set, default to openai for acceptance.
    if os.getenv("OPENAI_API_KEY") and not os.getenv("LLM_PROVIDER"):
        os.environ["LLM_PROVIDER"] = "openai"

    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.main import app
    from app.services.langfuse_service import langfuse_service

    settings = get_settings()
    client = TestClient(app)

    cases = _load_jsonl(Path(args.golden_set))
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    if not cases:
        raise SystemExit(f"no valid cases in golden set: {args.golden_set}")

    prerequisite = {
        "openai_api_key_present": bool(settings.openai_api_key),
        "llm_provider": settings.llm_provider,
        "langfuse_keys_present": bool(settings.langfuse_public_key and settings.langfuse_secret_key),
        "langfuse_enabled": bool(langfuse_service.health().get("enabled")),
        "golden_set_cases_used": len(cases),
    }

    run_payload = {
        "run_name": f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "top_k": args.top_k,
        "run_ragas": True,
        "run_deepeval": True,
        "question_set": cases,
    }

    started = time.perf_counter()
    run_resp = client.post("/api/eval/run", json=run_payload)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    run_data: dict[str, Any] = {}
    detail_data: dict[str, Any] = {}
    if run_resp.status_code == 200:
        run_data = run_resp.json()
        detail_resp = client.get(f"/api/eval/runs/{run_data['run_id']}")
        if detail_resp.status_code == 200:
            detail_data = detail_resp.json()

    created = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = settings.eval_reports_dir / f"acceptance_{created}.md"
    snapshot_path = settings.eval_reports_dir / f"acceptance_{created}.png"

    ragas_status = _short(detail_data.get("ragas", {}).get("status"), default="not_started")
    deepeval_status = _short(detail_data.get("deepeval", {}).get("status"), default="not_started")
    langfuse_health = langfuse_service.health()

    lines = [
        "# Real Environment Acceptance Report",
        "",
        f"- Generated At (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- Request Run Name: `{run_payload['run_name']}`",
        f"- Response Status Code: `{run_resp.status_code}`",
        f"- Total Latency(ms): `{elapsed_ms}`",
        "",
        "## Prerequisites",
        "",
        f"- OPENAI_API_KEY present: `{prerequisite['openai_api_key_present']}`",
        f"- LLM_PROVIDER: `{prerequisite['llm_provider']}`",
        f"- Langfuse keys present: `{prerequisite['langfuse_keys_present']}`",
        f"- Langfuse enabled: `{prerequisite['langfuse_enabled']}`",
        f"- Golden cases used: `{prerequisite['golden_set_cases_used']}`",
        "",
        "## Evaluation Result",
        "",
        f"- run_id: `{run_data.get('run_id')}`",
        f"- status: `{run_data.get('status')}`",
        f"- report_path: `{run_data.get('report_path')}`",
        f"- ragas_status: `{ragas_status}`",
        f"- deepeval_status: `{deepeval_status}`",
        f"- blended_overall: `{_short(detail_data.get('channels', {}).get('blended_overall'))}`",
        f"- faithfulness: `{_short(detail_data.get('metrics', {}).get('faithfulness'))}`",
        f"- answer_relevancy: `{_short(detail_data.get('metrics', {}).get('answer_relevancy'))}`",
        f"- context_recall: `{_short(detail_data.get('metrics', {}).get('context_recall'))}`",
        f"- citation_accuracy: `{_short(detail_data.get('metrics', {}).get('citation_accuracy'))}`",
        "",
        "## Langfuse",
        "",
        f"- host: `{langfuse_health.get('host')}`",
        f"- enabled: `{langfuse_health.get('enabled')}`",
        f"- error: `{langfuse_health.get('error')}`",
        "",
        "## Notes",
        "",
        "- Full-chain acceptance requires OPENAI + Langfuse keys and `LLM_PROVIDER=openai`.",
        "- If DeepEval is skipped, verify OPENAI key and model access first.",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    snapshot_lines = [
        "AcaRAG Real Acceptance Snapshot",
        f"run_id: {_short(run_data.get('run_id'))}",
        f"run_status: {_short(run_data.get('status'))}",
        f"response_status_code: {run_resp.status_code}",
        f"cases_used: {prerequisite['golden_set_cases_used']}",
        f"ragas_status: {ragas_status}",
        f"deepeval_status: {deepeval_status}",
        f"langfuse_enabled: {langfuse_health.get('enabled')}",
        f"blended_overall: {_short(detail_data.get('channels', {}).get('blended_overall'))}",
        f"faithfulness: {_short(detail_data.get('metrics', {}).get('faithfulness'))}",
        f"answer_relevancy: {_short(detail_data.get('metrics', {}).get('answer_relevancy'))}",
        f"context_recall: {_short(detail_data.get('metrics', {}).get('context_recall'))}",
        f"citation_accuracy: {_short(detail_data.get('metrics', {}).get('citation_accuracy'))}",
        f"report: {report_path.name}",
    ]
    _draw_snapshot(snapshot_path, snapshot_lines)

    print(
        json.dumps(
            {
                "acceptance_report": str(report_path),
                "snapshot": str(snapshot_path),
                "run_status_code": run_resp.status_code,
                "run_id": run_data.get("run_id"),
                "ragas_status": ragas_status,
                "deepeval_status": deepeval_status,
                "langfuse_enabled": langfuse_health.get("enabled"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # Exit non-zero when strict full-chain prerequisites are absent.
    if not prerequisite["openai_api_key_present"]:
        return 2
    if str(prerequisite["llm_provider"]).lower() != "openai":
        return 2
    if not prerequisite["langfuse_keys_present"]:
        return 2
    if ragas_status != "completed" or deepeval_status != "completed":
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
