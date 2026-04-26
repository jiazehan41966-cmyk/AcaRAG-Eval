import re

from app.services.embedding_service import embedding_service


class FaithfulnessService:
    def check(self, answer: str, contexts: list[str]) -> dict:
        if not contexts:
            return {
                "score": 0.0,
                "supported_claim_ratio": 0.0,
                "reason": "No retrieved context",
            }

        context_blob = " ".join(contexts)
        context_tokens = set(embedding_service.tokenize(context_blob))

        sentences = [s.strip() for s in re.split(r"[。！？!?\n]", answer) if s.strip()]
        if not sentences:
            return {"score": 0.0, "supported_claim_ratio": 0.0, "reason": "Empty answer"}

        supported = 0
        for sentence in sentences:
            sentence_tokens = set(embedding_service.tokenize(sentence))
            if not sentence_tokens:
                continue
            overlap = len(sentence_tokens & context_tokens)
            ratio = overlap / len(sentence_tokens)
            if ratio >= 0.25:
                supported += 1

        score = supported / len(sentences)
        return {
            "score": round(score, 4),
            "supported_claim_ratio": round(score, 4),
            "reason": "heuristic_overlap_check",
        }


faithfulness_service = FaithfulnessService()
