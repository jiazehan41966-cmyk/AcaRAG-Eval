from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from app.core.config import get_settings
from app.services.agent_service import agent_service
from app.services.embedding_service import embedding_service
from app.services.langfuse_service import langfuse_service
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
        run_deepeval: bool = False,
        baseline_name: str | None = None,
        set_as_baseline: bool = False,
        compare_with_baseline: bool = True,
    ) -> dict:
        run_id = uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        baseline_name = (baseline_name or run_name).strip()

        eval_trace = langfuse_service.create_trace(
            name="eval.run",
            input_payload={
                "run_name": run_name,
                "top_k": top_k,
                "run_ragas": run_ragas,
                "run_deepeval": run_deepeval,
            },
            metadata={"run_id": run_id, "baseline_name": baseline_name},
        )

        cases = question_set if question_set else self._load_golden_set(golden_set_path)
        if not cases:
            raise HTTPException(status_code=400, detail="No evaluation cases found")

        case_results: list[dict] = []
        ragas_rows: list[dict] = []
        deepeval_rows: list[dict] = []

        for idx, case in enumerate(cases, start=1):
            question = str(case.get("question", "")).strip()
            if not question:
                continue

            result = agent_service.run(question=question, top_k=top_k)
            answer = result.get("answer", "")
            citations = list(result.get("citations", []))
            trace = dict(result.get("trace", {}))

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

            retrieve_step = self._find_step(trace.get("steps", []), "retrieve_docs")
            grade_step = self._find_step(trace.get("steps", []), "grade_documents")
            hit_count = int(retrieve_step.get("hit_count", len(citations)) if retrieve_step else len(citations))
            avg_rerank = float(grade_step.get("avg_rerank_score", 0.0) if grade_step else 0.0)

            failure_type, failure_score = self._attribute_failure(
                faithfulness=faithfulness,
                citation_accuracy=citation_accuracy,
                context_recall=context_recall,
                hit_count=hit_count,
                avg_rerank_score=avg_rerank,
                has_expected_citations=bool(expected_citations),
            )

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
                "retrieve_hit_count": hit_count,
                "avg_rerank_score": round(avg_rerank, 6),
                "failure_type": failure_type,
                "failure_score": round(failure_score, 6),
                "citations": citations,
                "trace": trace,
                "trace_id": trace.get("trace_id"),
                "difficulty": case.get("difficulty", ""),
                "question_type": case.get("question_type", ""),
            }
            case_results.append(row)

            contexts = [item.get("text", "") for item in citations if item.get("text")]
            ragas_rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "ground_truth": case.get("ground_truth", ""),
                    "contexts": contexts,
                }
            )
            deepeval_rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "ground_truth": case.get("ground_truth", ""),
                    "contexts": contexts,
                    "case_idx": idx,
                }
            )

            langfuse_service.add_event(
                trace_id=eval_trace["trace_id"],
                name="eval.case",
                input_payload={"idx": idx, "question": question},
                output_payload={
                    "failure_type": failure_type,
                    "metrics": {
                        "faithfulness": row["faithfulness"],
                        "answer_relevancy": row["answer_relevancy"],
                        "context_recall": row["context_recall"],
                        "citation_accuracy": row["citation_accuracy"],
                    },
                },
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
            "overall_score": None,
        }

        deepeval = self._run_deepeval(deepeval_rows) if (run_deepeval or self.settings.enable_deepeval) else {
            "status": "skipped",
            "reason": "run_deepeval=false",
            "overall_score": None,
            "component_scores": {},
            "cases": [],
        }

        channels = self._build_channel_scores(summary_metrics, ragas, deepeval)
        failure_summary = self._summarize_failures(case_results)

        report_path = self._write_markdown_report(
            run_id=run_id,
            run_name=run_name,
            created_at=created_at,
            summary_metrics=summary_metrics,
            channels=channels,
            ragas=ragas,
            deepeval=deepeval,
            failure_summary=failure_summary,
            cases=case_results,
        )

        payload = {
            "run_id": run_id,
            "run_name": run_name,
            "status": "completed",
            "cases": len(case_results),
            "metrics": summary_metrics,
            "channels": channels,
            "ragas": ragas,
            "deepeval": deepeval,
            "failure_summary": failure_summary,
            "report_path": str(report_path),
            "regression_report_path": None,
            "baseline_name": baseline_name,
            "baseline_comparison": None,
            "created_at": created_at,
            "results": case_results,
        }

        if compare_with_baseline and baseline_name:
            baseline_run_id = state_store.get_baseline(baseline_name)
            baseline_run = state_store.get_eval_run(baseline_run_id) if baseline_run_id else None
            if baseline_run:
                comparison = self._compare_with_baseline(current=payload, baseline=baseline_run)
                regression_path = self._write_regression_report(
                    baseline_name=baseline_name,
                    current=payload,
                    baseline=baseline_run,
                    comparison=comparison,
                )
                payload["baseline_comparison"] = comparison
                payload["regression_report_path"] = str(regression_path)

        if set_as_baseline and baseline_name:
            state_store.save_baseline(baseline_name=baseline_name, run_id=run_id)

        state_store.save_eval_run(run_id, payload)

        langfuse_service.finalize_trace(
            trace_id=eval_trace["trace_id"],
            output_payload={
                "run_id": run_id,
                "channels": channels,
                "failure_summary": failure_summary,
            },
            metadata={
                "cases": payload["cases"],
                "report_path": payload["report_path"],
                "regression_report_path": payload["regression_report_path"],
            },
        )

        return payload

    def get_eval_run(self, run_id: str) -> dict:
        payload = state_store.get_eval_run(run_id)
        if not payload:
            raise HTTPException(status_code=404, detail="Eval run not found")
        return payload

    def list_eval_runs(self) -> list[dict]:
        return state_store.list_eval_runs()

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

    def _find_step(self, steps: list[dict], node_name: str) -> dict:
        for step in reversed(steps or []):
            if step.get("node") == node_name:
                return step
        return {}

    def _attribute_failure(
        self,
        faithfulness: float,
        citation_accuracy: float,
        context_recall: float,
        hit_count: int,
        avg_rerank_score: float,
        has_expected_citations: bool,
    ) -> tuple[str, float]:
        if faithfulness < 0.45:
            return "unsupported_claim", 1 - faithfulness

        if has_expected_citations and citation_accuracy < 0.5:
            return "citation_error", 1 - citation_accuracy

        if context_recall < 0.35:
            if hit_count == 0:
                return "retrieval_miss", 1 - context_recall
            if avg_rerank_score < 0.18:
                return "rerank_error", 1 - avg_rerank_score
            return "retrieval_miss", 1 - context_recall

        if hit_count > 0 and avg_rerank_score < 0.18:
            return "rerank_error", 1 - avg_rerank_score

        return "none", 0.0

    def _summarize_failures(self, case_results: list[dict]) -> dict:
        summary = {
            "none": 0,
            "retrieval_miss": 0,
            "rerank_error": 0,
            "citation_error": 0,
            "unsupported_claim": 0,
        }
        for row in case_results:
            key = row.get("failure_type", "none")
            summary[key] = summary.get(key, 0) + 1
        summary["failure_rate"] = round((len(case_results) - summary.get("none", 0)) / len(case_results), 6)
        return summary

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
                "overall_score": None,
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

            numeric_values = [float(v) for v in scores.values() if isinstance(v, (int, float))]
            overall = self._avg(numeric_values) if numeric_values else None
            return {
                "status": "completed",
                "reason": "ok",
                "scores": scores,
                "overall_score": overall,
            }
        except Exception as exc:  # pragma: no cover
            return {
                "status": "failed",
                "reason": str(exc),
                "scores": {},
                "overall_score": None,
            }

    def _run_deepeval(self, rows: list[dict]) -> dict:
        if not self.settings.openai_api_key:
            return {
                "status": "skipped",
                "reason": "OPENAI_API_KEY missing",
                "overall_score": None,
                "component_scores": {},
                "cases": [],
            }

        try:
            from deepeval import evaluate
            from deepeval.evaluate.configs import DisplayConfig
            from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
            from deepeval.test_case import LLMTestCase
        except Exception as exc:
            return {
                "status": "skipped",
                "reason": f"deepeval import failed: {exc}",
                "overall_score": None,
                "component_scores": {},
                "cases": [],
            }

        test_cases = [
            LLMTestCase(
                input=row["question"],
                actual_output=row["answer"],
                expected_output=row["ground_truth"],
                retrieval_context=row["contexts"],
                name=f"case_{row['case_idx']}",
            )
            for row in rows
        ]

        metrics = [
            AnswerRelevancyMetric(model=self.settings.llm_model_name, async_mode=False, include_reason=True),
            FaithfulnessMetric(model=self.settings.llm_model_name, async_mode=False, include_reason=True),
        ]

        try:
            evaluation = evaluate(
                test_cases=test_cases,
                metrics=metrics,
                display_config=DisplayConfig(show_indicator=False, print_results=False),
            )

            per_metric: defaultdict[str, list[float]] = defaultdict(list)
            case_outputs: list[dict] = []

            for test_result in evaluation.test_results:
                metric_items = []
                scores = []
                for metric_data in test_result.metrics_data or []:
                    score = float(metric_data.score) if metric_data.score is not None else None
                    if score is not None:
                        per_metric[metric_data.name].append(score)
                        scores.append(score)
                    metric_items.append(
                        {
                            "name": metric_data.name,
                            "score": score,
                            "success": bool(metric_data.success),
                            "reason": metric_data.reason,
                            "error": metric_data.error,
                        }
                    )

                case_outputs.append(
                    {
                        "name": test_result.name,
                        "success": bool(test_result.success),
                        "score": self._avg(scores) if scores else None,
                        "metrics": metric_items,
                    }
                )

            component_scores = {name: self._avg(vals) for name, vals in per_metric.items()}
            numeric_values = list(component_scores.values())
            overall = self._avg(numeric_values) if numeric_values else None

            return {
                "status": "completed",
                "reason": "ok",
                "overall_score": overall,
                "component_scores": component_scores,
                "cases": case_outputs,
                "test_run_id": getattr(evaluation, "test_run_id", None),
            }
        except Exception as exc:  # pragma: no cover
            return {
                "status": "failed",
                "reason": str(exc),
                "overall_score": None,
                "component_scores": {},
                "cases": [],
            }

    def _build_channel_scores(self, summary_metrics: dict, ragas: dict, deepeval: dict) -> dict:
        heuristic_overall = self._avg(
            [
                summary_metrics.get("faithfulness", 0.0),
                summary_metrics.get("answer_relevancy", 0.0),
                summary_metrics.get("context_recall", 0.0),
                summary_metrics.get("citation_accuracy", 0.0),
            ]
        )

        channels = {
            "heuristic": {
                "overall_score": heuristic_overall,
                "component_scores": summary_metrics,
            },
            "ragas": {
                "overall_score": ragas.get("overall_score"),
                "component_scores": ragas.get("scores", {}),
                "status": ragas.get("status"),
            },
            "deepeval": {
                "overall_score": deepeval.get("overall_score"),
                "component_scores": deepeval.get("component_scores", {}),
                "status": deepeval.get("status"),
            },
        }

        available = [
            channels["heuristic"].get("overall_score"),
            channels["ragas"].get("overall_score"),
            channels["deepeval"].get("overall_score"),
        ]
        valid = [float(v) for v in available if isinstance(v, (int, float))]
        channels["blended_overall"] = self._avg(valid) if valid else None
        return channels

    def _compare_with_baseline(self, current: dict, baseline: dict) -> dict:
        metric_diff = {}
        for key in ("faithfulness", "answer_relevancy", "question_coverage", "context_recall", "citation_accuracy"):
            current_val = float(current.get("metrics", {}).get(key, 0.0))
            baseline_val = float(baseline.get("metrics", {}).get(key, 0.0))
            metric_diff[key] = {
                "baseline": baseline_val,
                "current": current_val,
                "delta": round(current_val - baseline_val, 6),
            }

        channel_diff = {}
        for channel in ("heuristic", "ragas", "deepeval"):
            current_val = current.get("channels", {}).get(channel, {}).get("overall_score")
            baseline_val = baseline.get("channels", {}).get(channel, {}).get("overall_score")
            if isinstance(current_val, (int, float)) and isinstance(baseline_val, (int, float)):
                channel_diff[channel] = {
                    "baseline": float(baseline_val),
                    "current": float(current_val),
                    "delta": round(float(current_val) - float(baseline_val), 6),
                }

        failure_types = ["retrieval_miss", "rerank_error", "citation_error", "unsupported_claim"]
        failure_diff = {}
        for ft in failure_types:
            current_count = int(current.get("failure_summary", {}).get(ft, 0))
            baseline_count = int(baseline.get("failure_summary", {}).get(ft, 0))
            failure_diff[ft] = {
                "baseline": baseline_count,
                "current": current_count,
                "delta": current_count - baseline_count,
            }

        regressed_metrics = [k for k, v in metric_diff.items() if v["delta"] < 0]
        regressed_failures = [k for k, v in failure_diff.items() if v["delta"] > 0]

        return {
            "baseline_run_id": baseline.get("run_id"),
            "baseline_run_name": baseline.get("run_name"),
            "metric_diff": metric_diff,
            "channel_diff": channel_diff,
            "failure_diff": failure_diff,
            "regressed_metrics": regressed_metrics,
            "regressed_failures": regressed_failures,
        }

    def _write_regression_report(self, baseline_name: str, current: dict, baseline: dict, comparison: dict) -> Path:
        path = self.settings.eval_reports_dir / f"regression_{baseline.get('run_id')}_to_{current.get('run_id')}.md"
        lines = [
            f"# Regression Report: {baseline_name}",
            "",
            f"- Baseline Run: `{baseline.get('run_id')}` ({baseline.get('run_name')})",
            f"- Current Run: `{current.get('run_id')}` ({current.get('run_name')})",
            "",
            "## Metric Diff",
            "",
            "| Metric | Baseline | Current | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]

        for key, item in comparison.get("metric_diff", {}).items():
            lines.append(f"| {key} | {item['baseline']:.4f} | {item['current']:.4f} | {item['delta']:+.4f} |")

        lines.extend(
            [
                "",
                "## Channel Diff",
                "",
                "| Channel | Baseline | Current | Delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for key, item in comparison.get("channel_diff", {}).items():
            lines.append(f"| {key} | {item['baseline']:.4f} | {item['current']:.4f} | {item['delta']:+.4f} |")

        lines.extend(
            [
                "",
                "## Failure Diff",
                "",
                "| Failure Type | Baseline | Current | Delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for key, item in comparison.get("failure_diff", {}).items():
            lines.append(f"| {key} | {item['baseline']} | {item['current']} | {item['delta']:+d} |")

        lines.extend(
            [
                "",
                f"- Regressed Metrics: {comparison.get('regressed_metrics', [])}",
                f"- Regressed Failures: {comparison.get('regressed_failures', [])}",
                "",
            ]
        )

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_markdown_report(
        self,
        run_id: str,
        run_name: str,
        created_at: str,
        summary_metrics: dict,
        channels: dict,
        ragas: dict,
        deepeval: dict,
        failure_summary: dict,
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
            "## Channel Scores",
            "",
            f"- Heuristic Overall: `{channels.get('heuristic', {}).get('overall_score')}`",
            f"- RAGAS Overall: `{channels.get('ragas', {}).get('overall_score')}`",
            f"- DeepEval Overall: `{channels.get('deepeval', {}).get('overall_score')}`",
            f"- Blended Overall: `{channels.get('blended_overall')}`",
            "",
            "## Failure Summary",
            "",
            "| Type | Count |",
            "| --- | ---: |",
            f"| retrieval_miss | {failure_summary.get('retrieval_miss', 0)} |",
            f"| rerank_error | {failure_summary.get('rerank_error', 0)} |",
            f"| citation_error | {failure_summary.get('citation_error', 0)} |",
            f"| unsupported_claim | {failure_summary.get('unsupported_claim', 0)} |",
            f"| none | {failure_summary.get('none', 0)} |",
            f"| failure_rate | {failure_summary.get('failure_rate', 0.0)} |",
            "",
            "## RAGAS",
            "",
            f"- Status: `{ragas.get('status', 'unknown')}`",
            f"- Reason: `{ragas.get('reason', '')}`",
            "",
            "## DeepEval",
            "",
            f"- Status: `{deepeval.get('status', 'unknown')}`",
            f"- Reason: `{deepeval.get('reason', '')}`",
            f"- Overall Score: `{deepeval.get('overall_score')}`",
            "",
        ]

        ragas_scores = ragas.get("scores", {}) if isinstance(ragas, dict) else {}
        if ragas_scores:
            lines.extend(["| RAGAS Metric | Score |", "| --- | ---: |"])
            for key, value in ragas_scores.items():
                lines.append(f"| {key} | {value} |")
            lines.append("")

        deepeval_scores = deepeval.get("component_scores", {}) if isinstance(deepeval, dict) else {}
        if deepeval_scores:
            lines.extend(["| DeepEval Metric | Score |", "| --- | ---: |"])
            for key, value in deepeval_scores.items():
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
                    f"- Failure Type: `{case.get('failure_type')}`",
                    f"- Failure Score: `{case.get('failure_score')}`",
                    f"- Trace ID: `{case.get('trace_id')}`",
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
