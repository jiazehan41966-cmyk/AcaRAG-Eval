from __future__ import annotations

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

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.settings.llm_provider,
            "model": self.settings.llm_model_name,
            "enabled": self.is_enabled(),
            "error": self._client_error,
        }


llm_service = LLMService()
