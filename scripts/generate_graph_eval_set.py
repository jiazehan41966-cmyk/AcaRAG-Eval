#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_cases() -> list[dict]:
    comparison_templates = [
        (
            "Compare method A versus method B on dataset usage and metric improvements.",
            "Method A and method B should be compared by datasets, metrics, empirical gains, and cited evidence.",
        ),
        (
            "请对比方法 A 和方法 B 在数据集选择、指标提升和证据来源上的差异。",
            "回答应覆盖方法差异、数据集选择、指标对比，并给出可追溯证据。",
        ),
        (
            "What is the difference between baseline retrieval and graph-enhanced retrieval for cross-paper questions?",
            "Graph-enhanced retrieval should better connect cross-entity and cross-paper evidence than plain retrieval.",
        ),
        (
            "相比普通 RAG，Graph-enhanced 检索在多论文对比任务上有什么优势？",
            "Graph-enhanced 检索更适合跨论文实体关系问题，普通 RAG 更偏单文档事实问答。",
        ),
        (
            "Please compare the BM25 + vector + rerank pipeline with dense-only retrieval in this project.",
            "Hybrid retrieval with rerank should improve evidence ranking and reduce retrieval misses.",
        ),
        (
            "请比较向量检索、BM25 检索和混合检索在论文问答场景中的不同作用。",
            "回答应说明向量检索负责语义召回，BM25 负责关键词匹配，混合检索负责融合两类证据。",
        ),
        (
            "Compare graph evidence paths with flat retrieved chunks for explaining multi-paper claims.",
            "Graph evidence paths should expose entities and relations, while flat chunks mainly provide local text evidence.",
        ),
        (
            "请比较结构化实体关系索引和普通 chunk 索引在可解释证据上的区别。",
            "结构化索引应突出实体、关系和证据链，chunk 索引应突出原文片段。",
        ),
    ]

    multihop_templates = [
        (
            "Trace the chain from question routing to query rewrite to retrieval to faithfulness check.",
            "The answer should describe router, rewrite, retrieval, grading, generation, and faithfulness validation.",
        ),
        (
            "请说明系统如何从问题路由、证据召回、重排到答案一致性校验形成闭环。",
            "回答应说明路由、检索、重排、生成、faithfulness 校验和回归评测闭环。",
        ),
        (
            "Explain why router and graph retrieval are both needed for comparison and multi-hop questions.",
            "Router selects the strategy, while graph retrieval provides structured cross-entity evidence for reasoning.",
        ),
        (
            "为什么 comparison 和 multi-hop 问题需要同时依赖 Router 与 Graph 检索？",
            "Router 负责选择流程，Graph 检索负责补充跨实体、跨论文的结构化证据。",
        ),
        (
            "How can failure attribution help improve graph-enhanced retrieval quality over iterations?",
            "Failure attribution should identify retrieval miss, rerank error, citation error, and unsupported claims.",
        ),
        (
            "从 failure case 归因出发，系统如何定位 graph 检索中的召回缺失或重排错误？",
            "回答应说明如何用 retrieval_miss、rerank_error、citation_error 等标签指导优化。",
        ),
        (
            "Starting from a comparison question, explain how the graph branch improves interpretable evidence.",
            "The graph branch should output matched entities, relations, and clearer cross-paper evidence paths.",
        ),
        (
            "请从一个对比问题出发，说明 graph 分支如何提升 evidence 的可解释性。",
            "Graph 分支应通过实体匹配、关系路径和证据片段增强答案解释性。",
        ),
    ]

    cases: list[dict] = []
    for i in range(16):
        question, ground_truth = comparison_templates[i % len(comparison_templates)]
        cases.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "supporting_evidence": [],
                "expected_citation": [],
                "difficulty": "medium" if i < 8 else "hard",
                "question_type": "comparison",
                "failure_label": "",
            }
        )

    for i in range(16):
        question, ground_truth = multihop_templates[i % len(multihop_templates)]
        cases.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "supporting_evidence": [],
                "expected_citation": [],
                "difficulty": "medium" if i < 8 else "hard",
                "question_type": "multi-hop",
                "failure_label": "",
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate graph-focused comparison and multi-hop eval set")
    parser.add_argument("--output", default="data/golden_set/graph_eval_set.jsonl")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = _build_cases()
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in cases) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(output),
                "cases": len(cases),
                "by_type": {"comparison": 16, "multi-hop": 16},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
