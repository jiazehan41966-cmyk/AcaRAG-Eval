from __future__ import annotations

import re
from typing import Any

from app.services.llm_service import llm_service


class FaithfulnessService:
    CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
    TOKEN_RE = re.compile(r"[a-zA-Z0-9_\-]+|[\u4e00-\u9fff]+")
    STOPWORDS = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "的",
        "了",
        "是",
        "在",
        "和",
        "与",
        "中",
        "对",
        "及",
        "这",
        "该",
        "一个",
        "一种",
        "可以",
    }

    def _content_tokens(self, text: str) -> set[str]:
        tokens: set[str] = set()
        for token in self.TOKEN_RE.findall((text or "").lower()):
            if token in self.STOPWORDS:
                continue
            if self.CJK_RE.fullmatch(token):
                filtered = "".join(ch for ch in token if ch not in self.STOPWORDS)
                if len(filtered) == 1:
                    continue
                if len(filtered) == 2:
                    tokens.add(filtered)
                    continue
                for idx in range(len(filtered) - 1):
                    gram = filtered[idx : idx + 2]
                    if gram not in self.STOPWORDS:
                        tokens.add(gram)
                continue
            if len(token) >= 3 and token not in self.STOPWORDS:
                tokens.add(token)
        return tokens

    def _heuristic(self, answer: str, contexts: list[str]) -> dict[str, Any]:
        if not contexts:
            return {
                "score": 0.0,
                "supported_claim_ratio": 0.0,
                "reason": "No retrieved context",
                "source": "heuristic",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "token_cost": 0.0,
            }

        context_blob = " ".join(contexts)
        context_tokens = self._content_tokens(context_blob)

        sentences = [s.strip() for s in re.split(r"[。！？!?\n]", answer) if s.strip()]
        if not sentences:
            return {
                "score": 0.0,
                "supported_claim_ratio": 0.0,
                "reason": "Empty answer",
                "source": "heuristic",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "token_cost": 0.0,
            }

        supported = 0
        for sentence in sentences:
            sentence_tokens = self._content_tokens(sentence)
            if not sentence_tokens:
                continue
            overlap = len(sentence_tokens & context_tokens)
            ratio = overlap / len(sentence_tokens)
            min_overlap = 1 if len(sentence_tokens) <= 3 else 2
            if overlap >= min_overlap and ratio >= 0.35:
                supported += 1

        score = supported / len(sentences)
        return {
            "score": round(score, 4),
            "supported_claim_ratio": round(score, 4),
            "reason": "heuristic_content_bigram_overlap_check",
            "source": "heuristic",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "token_cost": 0.0,
        }

    def check(self, question: str, answer: str, contexts: list[str]) -> dict[str, Any]:
        llm_checked = llm_service.check_faithfulness(
            question=question,
            answer=answer,
            contexts=contexts,
        )
        if llm_checked is not None:
            result, meta = llm_checked
            return {
                **result,
                "prompt_tokens": int(meta.get("prompt_tokens", 0)),
                "completion_tokens": int(meta.get("completion_tokens", 0)),
                "total_tokens": int(meta.get("total_tokens", 0)),
                "token_cost": float(meta.get("token_cost", 0.0)),
            }

        return self._heuristic(answer=answer, contexts=contexts)


faithfulness_service = FaithfulnessService()
