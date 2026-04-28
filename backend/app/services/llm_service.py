from __future__ import annotations

import json
import re
from typing import Any

from app.core.config import get_settings
from app.services.embedding_service import embedding_service


MODEL_PRICING = {
    # USD per 1K tokens (rough reference)
    "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
}


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        self._client_error: str | None = None

    def _init_openai_client(self):
        if self._client is not None or self._client_error is not None:
            return self._client
        if self.settings.llm_provider.lower() != "openai":
            self._client_error = "llm_provider != openai"
            return None
        if not self.settings.openai_api_key:
            self._client_error = "OPENAI_API_KEY missing"
            return None

        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
            return self._client
        except Exception as exc:  # pragma: no cover
            self._client_error = str(exc)
            return None

    def is_enabled(self) -> bool:
        return self._init_openai_client() is not None

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(embedding_service.tokenize(text)))

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        price = MODEL_PRICING.get(model, MODEL_PRICING.get("gpt-4o-mini"))
        input_cost = (prompt_tokens / 1000) * price["input"]
        output_cost = (completion_tokens / 1000) * price["output"]
        return round(input_cost + output_cost, 8)

    def _chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> dict[str, Any] | None:
        client = self._init_openai_client()
        if client is None:
            return None

        try:
            response = client.chat.completions.create(
                model=self.settings.llm_model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            if prompt_tokens == 0:
                prompt_tokens = sum(self._estimate_tokens(msg.get("content", "")) for msg in messages)
            if completion_tokens == 0:
                completion_tokens = self._estimate_tokens(content)

            token_cost = self._estimate_cost(self.settings.llm_model_name, prompt_tokens, completion_tokens)

            return {
                "content": content,
                "model": self.settings.llm_model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "token_cost": token_cost,
            }
        except Exception as exc:  # pragma: no cover
            self._client_error = str(exc)
            return None

    @staticmethod
    def _extract_json_block(text: str) -> Any:
        text = (text or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            try:
                return json.loads(candidate)
            except Exception:
                pass

        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            try:
                return json.loads(candidate)
            except Exception:
                pass
        return None

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def rewrite_query(self, question: str, route: str) -> tuple[str, dict[str, Any]]:
        prompt = (
            "你是学术论文检索查询重写器。请把用户问题改写成便于检索的简洁查询，"
            "保留关键词（方法/数据集/指标/年份/作者），只输出一行改写结果。"
        )
        user_input = f"route={route}; question={question}"
        llm_result = self._chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ],
            max_tokens=128,
            temperature=0.0,
        )
        if llm_result and llm_result.get("content"):
            return str(llm_result["content"]), {
                "prompt": prompt,
                "input": user_input,
                **llm_result,
                "source": "llm",
            }

        if route == "graph_rag":
            rewritten = f"{question} 方法 对比 数据集 指标 引用"
        elif route == "mcp_tool":
            rewritten = f"{question} 论文 元数据 作者 年份 venue"
        else:
            rewritten = f"{question} 核心方法 实验结果 结论"

        prompt_tokens = self._estimate_tokens(prompt + " " + user_input)
        completion_tokens = self._estimate_tokens(rewritten)

        return rewritten, {
            "prompt": prompt,
            "input": user_input,
            "model": "heuristic",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "token_cost": 0.0,
            "source": "heuristic",
        }

    def generate_answer(self, question: str, route: str, contexts: list[dict]) -> tuple[str, dict[str, Any]]:
        if not contexts:
            empty_answer = "没有检索到可用证据，建议先上传并解析文档，再构建索引。"
            tokens = self._estimate_tokens(empty_answer)
            return empty_answer, {
                "prompt": "",
                "input": question,
                "model": "heuristic",
                "prompt_tokens": 0,
                "completion_tokens": tokens,
                "total_tokens": tokens,
                "token_cost": 0.0,
                "source": "heuristic",
            }

        evidence = []
        for idx, hit in enumerate(contexts[:5], start=1):
            section = hit.get("section") or "Unknown Section"
            text = (hit.get("text") or "").strip()
            evidence.append(f"[{idx}] ({section}) {text[:320]}")

        context_block = "\n".join(evidence)

        prompt = (
            "你是学术论文问答助手。必须仅根据给定证据回答，"
            "并给出要点化结论；如果证据不足要明确指出。"
        )
        user_input = (
            f"route={route}\n"
            f"question={question}\n"
            f"evidence:\n{context_block}\n"
            "请输出：1) 简短结论 2) 关键证据点。"
        )
        llm_result = self._chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        if llm_result and llm_result.get("content"):
            return str(llm_result["content"]), {
                "prompt": prompt,
                "input": user_input,
                **llm_result,
                "source": "llm",
            }

        answer = (
            f"问题：{question}\n"
            "基于检索证据的保守回答：\n"
            f"{context_block}\n"
            "说明：当前为非 LLM 兜底生成，建议配置 OpenAI 后获得更自然的综合回答。"
        )

        prompt_tokens = self._estimate_tokens(prompt + " " + user_input)
        completion_tokens = self._estimate_tokens(answer)

        return answer, {
            "prompt": prompt,
            "input": user_input,
            "model": "heuristic",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "token_cost": 0.0,
            "source": "heuristic",
        }

    def grade_documents(self, question: str, hits: list[dict]) -> tuple[list[dict], dict[str, Any]]:
        if not hits:
            return [], {
                "prompt": "",
                "input": question,
                "output": "[]",
                "model": "heuristic",
                "source": "heuristic",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "token_cost": 0.0,
            }

        snippet_lines = []
        for idx, hit in enumerate(hits[:12], start=1):
            text = str(hit.get("text", "")).replace("\n", " ").strip()[:280]
            snippet_lines.append(
                f"{idx}. score={hit.get('rerank_score', hit.get('score', 0.0)):.4f}; "
                f"section={hit.get('section')}; text={text}"
            )
        snippet_block = "\n".join(snippet_lines)

        prompt = (
            "你是文档相关性判别器。根据用户问题判断每个候选证据是否相关。"
            "请返回 JSON 数组，每项包含 index(从1开始)、relevant(bool)、score(0-1)、reason。"
        )
        user_input = f"question={question}\n\ncandidates:\n{snippet_block}"
        llm_result = self._chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ],
            max_tokens=700,
            temperature=0.0,
        )

        if llm_result and llm_result.get("content"):
            parsed = self._extract_json_block(str(llm_result.get("content")))
            if isinstance(parsed, list):
                by_index = {}
                for row in parsed:
                    if not isinstance(row, dict):
                        continue
                    idx = int(self._to_float(row.get("index"), -1))
                    if idx < 1:
                        continue
                    by_index[idx] = {
                        "relevant": bool(row.get("relevant", False)),
                        "llm_relevance_score": self._to_float(row.get("score"), 0.0),
                        "llm_relevance_reason": str(row.get("reason", ""))[:280],
                    }

                graded: list[dict] = []
                for idx, hit in enumerate(hits, start=1):
                    info = by_index.get(idx, {})
                    score = self._to_float(info.get("llm_relevance_score"), 0.0)
                    relevant = bool(info.get("relevant", score >= 0.5))
                    graded.append(
                        {
                            **hit,
                            "llm_relevant": relevant,
                            "llm_relevance_score": round(score, 6),
                            "llm_relevance_reason": info.get("llm_relevance_reason", ""),
                        }
                    )
                return graded, {
                    "prompt": prompt,
                    "input": user_input,
                    "output": llm_result.get("content", ""),
                    **llm_result,
                    "source": "llm",
                }

        query_tokens = set(embedding_service.tokenize(question))
        graded = []
        for hit in hits:
            text_tokens = set(embedding_service.tokenize(str(hit.get("text", ""))))
            overlap = len(query_tokens & text_tokens)
            union = len(query_tokens | text_tokens) or 1
            score = overlap / union
            graded.append(
                {
                    **hit,
                    "llm_relevant": score >= 0.08,
                    "llm_relevance_score": round(score, 6),
                    "llm_relevance_reason": "heuristic_token_overlap",
                }
            )
        prompt_tokens = self._estimate_tokens(prompt + " " + user_input)
        completion_tokens = self._estimate_tokens(" ".join(str(item["llm_relevance_score"]) for item in graded))
        return graded, {
            "prompt": prompt,
            "input": user_input,
            "output": str(
                [
                    {
                        "index": i + 1,
                        "relevant": item["llm_relevant"],
                        "score": item["llm_relevance_score"],
                    }
                    for i, item in enumerate(graded)
                ]
            ),
            "model": "heuristic",
            "source": "heuristic",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "token_cost": 0.0,
        }

    def check_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if not contexts:
            return None

        evidence = []
        for idx, text in enumerate(contexts[:8], start=1):
            evidence.append(f"[{idx}] {str(text).replace(chr(10), ' ')[:320]}")
        evidence_block = "\n".join(evidence)

        prompt = (
            "你是事实一致性评估器。请判断 answer 是否被 evidence 支持。"
            "输出 JSON：score(0-1), supported_claim_ratio(0-1), reason。"
        )
        user_input = f"question={question}\nanswer={answer}\nevidence:\n{evidence_block}"
        llm_result = self._chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ],
            max_tokens=220,
            temperature=0.0,
        )
        if not llm_result or not llm_result.get("content"):
            return None

        parsed = self._extract_json_block(str(llm_result.get("content")))
        if not isinstance(parsed, dict):
            return None

        score = max(0.0, min(1.0, self._to_float(parsed.get("score"), 0.0)))
        ratio = max(0.0, min(1.0, self._to_float(parsed.get("supported_claim_ratio"), score)))
        reason = str(parsed.get("reason", ""))[:400]
        result = {
            "score": round(score, 4),
            "supported_claim_ratio": round(ratio, 4),
            "reason": reason or "llm_judge",
            "source": "llm",
        }
        meta = {
            "prompt": prompt,
            "input": user_input,
            "output": llm_result.get("content", ""),
            **llm_result,
            "source": "llm",
        }
        return result, meta

    def extract_entities_structured(self, text: str, max_entities: int = 10) -> list[dict[str, str]] | None:
        """Use LLM structured output to extract typed entities from academic text.

        Returns a list of {"name": ..., "type": ...} dicts, or None if LLM unavailable.
        Types: method, dataset, metric, model, concept.
        """
        if not self.is_enabled():
            return None

        system_prompt = (
            "You are an academic NLP entity extractor. Given a passage from a research paper, "
            "extract the most important named entities. For each entity, classify its type as one of: "
            "method, dataset, metric, model, concept.\n\n"
            "Return a JSON array of objects with keys 'name' and 'type'. "
            f"Return at most {max_entities} entities. Example:\n"
            '[{"name": "BERT", "type": "model"}, {"name": "SQuAD", "type": "dataset"}]'
        )
        result = self._chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text[:2000]},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        if not result:
            return None

        parsed = self._extract_json_block(result.get("content", ""))
        if not isinstance(parsed, list):
            return None

        valid_types = {"method", "dataset", "metric", "model", "concept"}
        entities: list[dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            etype = str(item.get("type", "concept")).strip().lower()
            if not name or len(name) < 2:
                continue
            if etype not in valid_types:
                etype = "concept"
            entities.append({"name": name, "type": etype})
            if len(entities) >= max_entities:
                break
        return entities if entities else None

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.settings.llm_provider,
            "model": self.settings.llm_model_name,
            "enabled": self.is_enabled(),
            "error": self._client_error,
        }


llm_service = LLMService()
