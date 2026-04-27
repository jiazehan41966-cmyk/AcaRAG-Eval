from __future__ import annotations

import re
from collections import defaultdict


class GraphService:
    STOPWORDS = {
        "的",
        "了",
        "和",
        "是",
        "在",
        "中",
        "对",
        "与",
        "如何",
        "什么",
        "为什么",
        "for",
        "the",
        "and",
        "with",
        "from",
        "this",
        "that",
        "method",
        "paper",
    }
    TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]+|[0-9]+|[\u4e00-\u9fff]+")

    def _candidate_tokens(self, text: str) -> list[str]:
        tokens: list[str] = []
        for raw in self.TOKEN_PATTERN.findall(text.lower()):
            if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
                if 2 <= len(raw) <= 8:
                    tokens.append(raw)
                if len(raw) > 1:
                    tokens.extend(raw[i : i + 2] for i in range(len(raw) - 1))
                if len(raw) > 2:
                    tokens.extend(raw[i : i + 3] for i in range(len(raw) - 2))
            else:
                tokens.append(raw)
        return tokens

    def extract_entities(self, text: str, max_entities: int = 12) -> list[str]:
        entities: list[str] = []
        seen = set()
        for token in self._candidate_tokens(text):
            token = token.strip().lower()
            if not token or token in self.STOPWORDS:
                continue
            if token.isdigit():
                continue
            if len(token) < 2:
                continue
            if token in seen:
                continue
            seen.add(token)
            entities.append(token)
            if len(entities) >= max_entities:
                break
        return entities

    def build_graph_index(self, chunk_payloads: dict[str, dict]) -> dict:
        entity_to_chunks: dict[str, list[str]] = defaultdict(list)
        chunk_to_entities: dict[str, list[str]] = {}
        entity_freq: dict[str, int] = defaultdict(int)

        for chunk_id, payload in chunk_payloads.items():
            text = str(payload.get("text", ""))
            entities = self.extract_entities(text=text, max_entities=14)
            if not entities:
                continue
            chunk_to_entities[chunk_id] = entities
            for entity in entities:
                entity_to_chunks[entity].append(chunk_id)
                entity_freq[entity] += 1

        return {
            "version": 2,
            "entity_to_chunks": dict(entity_to_chunks),
            "chunk_to_entities": chunk_to_entities,
            "entity_freq": dict(entity_freq),
        }

    def graph_search(self, query: str, top_k: int, chunk_payloads: dict[str, dict], graph_index: dict) -> list[dict]:
        entity_to_chunks = graph_index.get("entity_to_chunks", {})
        chunk_to_entities = graph_index.get("chunk_to_entities", {})
        query_entities = self.extract_entities(query, max_entities=16)

        if not query_entities:
            return []

        chunk_scores: dict[str, float] = defaultdict(float)
        matched_entities_by_chunk: dict[str, set[str]] = defaultdict(set)

        for entity in query_entities:
            for chunk_id in entity_to_chunks.get(entity, []):
                chunk_scores[chunk_id] += 1.0
                matched_entities_by_chunk[chunk_id].add(entity)

        hits = []
        for chunk_id, score in chunk_scores.items():
            payload = chunk_payloads.get(chunk_id)
            if not payload:
                continue
            entity_count = len(chunk_to_entities.get(chunk_id, [])) or 1
            norm_score = score / entity_count
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": payload.get("doc_id"),
                    "score": round(norm_score, 6),
                    "text": payload.get("text", ""),
                    "section": payload.get("section"),
                    "page": payload.get("page"),
                    "metadata": {
                        **(payload.get("metadata", {}) or {}),
                        "matched_entities": sorted(list(matched_entities_by_chunk.get(chunk_id, set()))),
                    },
                }
            )

        hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return hits[:top_k]


graph_service = GraphService()
