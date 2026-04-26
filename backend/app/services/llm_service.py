from __future__ import annotations

from typing import Any

from app.core.config import get_settings


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
        except Exception as exc:  # pragma: no cover - environment dependent
            self._client_error = str(exc)
            return None

    def is_enabled(self) -> bool:
        return self._init_openai_client() is not None

    def _chat(self, messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.1) -> str | None:
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
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # pragma: no cover - runtime dependent
            self._client_error = str(exc)
            return None

    def rewrite_query(self, question: str, route: str) -> str:
        prompt = (
            "你是学术论文检索查询重写器。请把用户问题改写成便于检索的简洁查询，"
            "保留关键词（方法/数据集/指标/年份/作者），只输出一行改写结果。"
        )
        rewritten = self._chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"route={route}; question={question}"},
            ],
            max_tokens=128,
            temperature=0.0,
        )
        if rewritten:
            return rewritten

        # Heuristic fallback.
        if route == "graph_rag":
            return f"{question} 方法 对比 数据集 指标 引用"
        if route == "mcp_tool":
            return f"{question} 论文 元数据 作者 年份 venue"
        return f"{question} 核心方法 实验结果 结论"

    def generate_answer(self, question: str, route: str, contexts: list[dict]) -> str:
        if not contexts:
            return "没有检索到可用证据，建议先上传并解析文档，再构建索引。"

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
        generated = self._chat(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"route={route}\n"
                        f"question={question}\n"
                        f"evidence:\n{context_block}\n"
                        "请输出：1) 简短结论 2) 关键证据点。"
                    ),
                },
            ],
            max_tokens=512,
            temperature=0.2,
        )
        if generated:
            return generated

        # Heuristic fallback.
        return (
            f"问题：{question}\n"
            "基于检索证据的保守回答：\n"
            f"{context_block}\n"
            "说明：当前为非 LLM 兜底生成，建议配置 OpenAI 后获得更自然的综合回答。"
        )

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.settings.llm_provider,
            "model": self.settings.llm_model_name,
            "enabled": self.is_enabled(),
            "error": self._client_error,
        }


llm_service = LLMService()
