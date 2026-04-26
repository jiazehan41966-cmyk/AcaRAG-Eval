from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from app.core.config import get_settings
from app.services.agent_service import agent_service
from app.services.embedding_service import embedding_service
from app.services.storage_service import state_store


class EvalService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def run_eval(
        self,
        run_name: str,
        question_set: list[dict],
        top_k: int,
        golden_set_path: str | None = None,
        run_ragas: bool = False,
    ) -> dict:
        run_id = uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()

        cases = question_set if question_set else self._load_golden_set(golden_set_path)
        if not cases:
            raise HTTPException(status_code=400, detail="No evaluation cases found")

        case_results: list[dict] = []
        ragas_rows: list[dict] = []

        for idx, case in enumerate(cases, start=1):
            question = str(case.get("question", "")).strip()
            if not question:
                continue

            result = agent_service.run(question=question, top_k=top_k)
            answer = result.get("answer", "")
            citations = list(result.get("citations", []))

            expected_support = set(case.get("supporting_evidence", []) or [])
            expected_citations = set(case.get("expected_citation", []) or [])

            retrieved_chunk_ids = {item.get("chunk_id") for item in citations if item.get("chunk_id")}
            predicted_citations = {
                f"{item.get('doc_id')}:{item.get('page')}"
                for item in citations
                if item.get("doc_id") and item.get("page") is not None
            }

            context_recall = (
                len(expected_support & retrieved_chunk_ids) / len(expected_support) if expected_support else 0.0
            )
            citation_accuracy = (
                len(expected_citations & predicted_citations) / len(expected_citations) if expected_citations else 0.0
            )
            answer_relevancy = self._token_overlap(answer, case.get("ground_truth", ""))
            question_coverage = self._token_overlap(answer, question)
            faithfulness = float(result.get("faithfulness", {}).get("score", 0.0))

            row = {
                "idx": idx,
                "question": question,
                "ground_truth": case.get("ground_truth", ""),
                "answer": answer,
                "route": result.get("route", "hybrid_rag"),
                "faithfulness": round(faithfulness, 6),
                "answer_relevancy": round(answer_relevancy, 6),
                "question_coverage": round(question_coverage, 6),
                "context_recall": round(context_recall, 6),
                "citation_accuracy": round(citation_accuracy, 6),
                "citations": citations,
                "trace": result.get("trace", {}),
                "difficulty": case.get("difficulty", ""),
                "question_type": case.get("question_type", ""),
            }
            case_results.append(row)

            ragas_rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "ground_truth": case.get("ground_truth", ""),
                    "contexts": [item.get("text", "") for item in citations if item.get("text")],
                }
            )

        if not case_results:
            raise HTTPException(status_code=400, detail="No valid evaluation cases with non-empty question")

        summary_metrics = {
            "faithfulness": self._avg([item["faithfulness"] for item in case_results]),
            "answer_relevancy": self._avg([item["answer_relevancy"] for item in case_results]),
            "question_coverage": self._avg([item["question_coverage"] for item in case_results]),
            "context_recall": self._avg([item["context_recall"] for item in case_results]),
            "citation_accuracy": self._avg([item["citation_accuracy"] for item in case_results]),
        }

        ragas = self._run_ragas(ragas_rows) if (run_ragas or self.settings.enable_ragas) else {
            "status": "skipped",
            "reason": "run_ragas=false",
            "scores": {},
        }

        report_path = self._write_markdown_report(
            run_id=run_id,
            run_name=run_name,
            created_at=created_at,
            summary_metrics=summary_metrics,
            ragas=ragas,
            cases=case_results,
        )

        payload = {
            "run_id": run_id,
            "run_name": run_name,
            "status": "completed",
            "cases": len(case_results),
            "metrics": summary_metrics,
            "ragas": ragas,
            "report_path": str(report_path),
            "created_at": created_at,
            "results": case_results,
        }
        state_store.save_eval_run(run_id, payload)
        return payload

    def get_eval_run(self, run_id: str) -> dict:
        payload = state_store.get_eval_run(run_id)
        if not payload:
            raise HTTPException(status_code=404, detail="Eval run not found")
        return payload

    def _load_golden_set(self, golden_set_path: str | None) -> list[dict]:
        path = Path(golden_set_path) if golden_set_path else self.settings.default_golden_set_path
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"Golden set not found: {path}")

        cases: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return cases

    def _token_overlap(self, text_a: str, text_b: str) -> float:
        tokens_a = set(embedding_service.tokenize(text_a or ""))
        tokens_b = set(embedding_service.tokenize(text_b or ""))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_b)

    def _avg(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values), 6)

    def _run_ragas(self, rows: list[dict]) -> dict:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics.collections import answer_relevancy, context_precision, context_recall, faithfulness
        except Exception as exc:
            return {
                "status": "skipped",
                "reason": f"ragas import failed: {exc}",
                "scores": {},
            }

        dataset = Dataset.from_dict(
            {
                "question": [row["question"] for row in rows],
                "answer": [row["answer"] for row in rows],
                "contexts": [row["contexts"] for row in rows],
                "ground_truth": [row["ground_truth"] for row in rows],
            }
        )

        try:
            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            )
            if hasattr(result, "to_dict"):
                scores = result.to_dict()
            elif hasattr(result, "__dict__"):
                scores = dict(result.__dict__)
            else:
                scores = {}
            return {
                "status": "completed",
                "reason": "ok",
                "scores": scores,
            }
        except Exception as exc:  # pragma: no cover - depends on model keys/runtime
            return {
                "status": "failed",
                "reason": str(exc),
                "scores": {},
            }

    def _write_markdown_report(
        self,
        run_id: str,
        run_name: str,
        created_at: str,
        summary_metrics: dict,
        ragas: dict,
        cases: list[dict],
    ) -> Path:
        report_path = self.settings.eval_reports_dir / f"{run_id}.md"

        lines = [
            f"# Eval Report: {run_name}",
            "",
            f"- Run ID: `{run_id}`",
            f"- Created At (UTC): `{created_at}`",
            f"- Cases: `{len(cases)}`",
            "",
            "## Summary Metrics",
            "",
            "| Metric | Score |",
            "| --- | ---: |",
            f"| faithfulness | {summary_metrics.get('faithfulness', 0.0):.4f} |",
            f"| answer_relevancy | {summary_metrics.get('answer_relevancy', 0.0):.4f} |",
            f"| question_coverage | {summary_metrics.get('question_coverage', 0.0):.4f} |",
            f"| context_recall | {summary_metrics.get('context_recall', 0.0):.4f} |",
            f"| citation_accuracy | {summary_metrics.get('citation_accuracy', 0.0):.4f} |",
            "",
            "## RAGAS",
            "",
            f"- Status: `{ragas.get('status', 'unknown')}`",
            f"- Reason: `{ragas.get('reason', '')}`",
            "",
        ]

        ragas_scores = ragas.get("scores", {}) if isinstance(ragas, dict) else {}
        if ragas_scores:
            lines.extend(
                [
                    "| RAGAS Metric | Score |",
                    "| --- | ---: |",
                ]
            )
            for key, value in ragas_scores.items():
                lines.append(f"| {key} | {value} |")
            lines.append("")

        lines.extend(["## Cases", ""])
        for case in cases:
            lines.extend(
                [
                    f"### Case {case['idx']}",
                    "",
                    f"- Question: {case['question']}",
                    f"- Route: `{case['route']}`",
                    f"- Faithfulness: `{case['faithfulness']:.4f}`",
                    f"- Answer Relevancy: `{case['answer_relevancy']:.4f}`",
                    f"- Context Recall: `{case['context_recall']:.4f}`",
                    f"- Citation Accuracy: `{case['citation_accuracy']:.4f}`",
                    "",
                    "Answer:",
                    "",
                    case["answer"],
                    "",
                ]
            )

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path


eval_service = EvalService()
