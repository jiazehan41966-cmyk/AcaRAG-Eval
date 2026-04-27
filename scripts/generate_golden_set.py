#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_cases() -> list[dict]:
    templates = {
        "fact": [
            ("该论文的核心方法是什么？", "论文提出了混合检索与重排结合的 RAG 流程，用于提升学术问答可靠性。"),
            ("作者声称系统主要改进点是什么？", "主要改进点是引入 query rewrite、文档评分和 faithfulness 检查的自纠错链路。"),
            ("论文在检索阶段采用了哪些策略？", "检索阶段结合向量召回、BM25 稀疏召回与 rerank 重排序。"),
            ("该工作如何处理论文章节结构？", "系统会抽取章节结构并把章节信息写入 chunk 元数据。"),
            ("评测层的核心目标是什么？", "评测层用于量化回答质量并支持回归分析与失败归因。"),
        ],
        "comparison": [
            ("与普通 RAG 相比，该系统的差异是什么？", "该系统增加了 Router、重写、二次检索和一致性检查，具备动态决策能力。"),
            ("BM25+向量与仅向量检索的差异是什么？", "混合检索可提升关键词命中并降低召回遗漏。"),
            ("加入 reranker 后结果通常如何变化？", "加入 reranker 后相关证据排序更靠前，回答可解释性更好。"),
            ("Graph-enhanced 检索与普通检索在多论文问题上的差异？", "图增强检索更适合跨论文实体关系问题，普通检索更适合单文档事实问答。"),
            ("RAGAS 与 DeepEval 的评测关注点有何差异？", "RAGAS偏向RAG链路指标，DeepEval补充整体与组件级 LLM 评测。"),
        ],
        "multi-hop": [
            ("从检索到生成再到校验，这个系统是如何形成闭环的？", "系统先检索候选证据，再生成答案，最后做 faithfulness 检查并沉淀失败归因用于回归优化。"),
            ("为什么需要 Router 与 Query Rewrite 共同存在？", "Router 负责分流问题类型，Rewrite 负责在选定路径内提高召回质量，二者协同减少误检。"),
            ("如何通过评测结果反推检索策略问题？", "可结合 context_recall、citation_accuracy 与 failure_type 判断是召回不足、重排问题还是引用错误。"),
            ("如果出现 unsupported_claim，系统应该如何定位原因？", "先检查证据覆盖，再检查生成与引用链路，必要时调低生成温度并强化约束提示词。"),
            ("回归测试中如何判断新版本是否可发布？", "对比 baseline 与当前 run 的核心指标和失败类型增量，若超过门限则阻断发布。"),
        ],
        "metadata": [
            ("这篇论文的作者与年份信息是什么？", "应从论文元数据中返回作者列表与年份字段。"),
            ("给定 doc_id 后可以查询哪些结构化信息？", "可查询标题、作者、页数、chunk 数量、解析状态与上传时间。"),
            ("如何查询某次评测 run 的报告元数据？", "可通过 run_id 查询状态、样本量、核心指标与失败摘要。"),
            ("如何列出某类失败样例？", "可按 failure_type 与阈值筛选失败 case，并返回问题与失败分数。"),
            ("MCP 工具层在该项目中的作用是什么？", "MCP 工具层用于标准化访问论文元数据和评测结果等结构化数据。"),
        ],
    }

    counts = {
        "fact": 24,
        "comparison": 20,
        "multi-hop": 20,
        "metadata": 16,
    }

    cases: list[dict] = []
    for question_type, n in counts.items():
        pool = templates[question_type]
        for i in range(n):
            q, a = pool[i % len(pool)]
            idx = len(cases) + 1
            evidence = [f"evidence_{question_type}_{idx}"] if idx % 3 == 0 else []
            citations = [f"paper_{(idx % 7) + 1}:{(idx % 9) + 1}"] if idx % 4 == 0 else []
            difficulty = "easy" if i < n * 0.35 else ("medium" if i < n * 0.75 else "hard")
            cases.append(
                {
                    "question": q,
                    "ground_truth": a,
                    "supporting_evidence": evidence,
                    "expected_citation": citations,
                    "difficulty": difficulty,
                    "question_type": question_type,
                    "failure_label": "",
                }
            )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate expanded golden set JSONL")
    parser.add_argument("--output", default="data/golden_set/golden_set.jsonl")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = build_cases()
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    output.write_text(text, encoding="utf-8")

    by_type: dict[str, int] = {}
    for row in rows:
        qt = row["question_type"]
        by_type[qt] = by_type.get(qt, 0) + 1

    print(json.dumps({"output": str(output), "cases": len(rows), "by_type": by_type}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

