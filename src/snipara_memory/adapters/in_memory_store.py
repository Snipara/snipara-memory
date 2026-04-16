"""In-memory adapter for local testing of the standalone memory engine."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..domain import Contradiction, GraveyardEntry, Memory, MemoryStatus, MemoryTier, MemoryType, RecallMatch, RecallQuery


class InMemoryMemoryStore:
    """Simple in-memory store that implements the memory service contract."""

    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}
        self._graveyard: dict[str, GraveyardEntry] = {}
        self._contradictions: dict[str, Contradiction] = {}
        self._embeddings: dict[str, list[float]] = {}

    async def create_memory(
        self,
        memory: Memory,
        *,
        embedding: Sequence[float] | None = None,
    ) -> Memory:
        self._memories[memory.id] = memory
        if embedding is not None:
            self._embeddings[memory.id] = list(embedding)
        return memory

    async def create_memories(
        self,
        memories: Sequence[Memory],
        *,
        embeddings: Sequence[Sequence[float] | None] | None = None,
    ) -> list[Memory]:
        created: list[Memory] = []
        embeddings = embeddings or [None] * len(memories)

        for memory, embedding in zip(memories, embeddings, strict=False):
            created.append(await self.create_memory(memory, embedding=embedding))

        return created

    async def search(
        self,
        query: RecallQuery,
        *,
        query_embedding: Sequence[float] | None = None,
    ) -> list[RecallMatch]:
        candidates = self._filter_memories(
            query.namespace_id,
            include_archived=query.include_archived,
            types=query.types,
            tiers=query.tiers,
            tags=query.tags,
        )
        matches = [
            RecallMatch(memory=memory, score=self._score(memory, query.query, query_embedding))
            for memory in candidates
        ]
        matches = [match for match in matches if match.score > 0]
        matches.sort(
            key=lambda match: (
                match.score,
                match.memory.confidence,
                match.memory.last_accessed_at or match.memory.created_at,
            ),
            reverse=True,
        )
        return matches[: query.limit]

    async def list_memories(
        self,
        namespace_id: str,
        *,
        statuses: Sequence[MemoryStatus] | None = None,
        tiers: Sequence[MemoryTier] | None = None,
        types: Sequence[MemoryType] | None = None,
        limit: int | None = None,
    ) -> list[Memory]:
        memories = [
            memory
            for memory in self._memories.values()
            if memory.namespace_id == namespace_id
            and (not statuses or memory.status in statuses)
            and (not tiers or memory.tier in tiers)
            and (not types or memory.type in types)
        ]
        memories.sort(
            key=lambda memory: memory.last_accessed_at or memory.created_at,
            reverse=True,
        )
        if limit is not None:
            return memories[:limit]
        return memories

    async def get_memory(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    async def update_memory(self, memory: Memory) -> Memory:
        self._memories[memory.id] = memory
        return memory

    async def delete_memory(self, memory_id: str) -> None:
        self._memories.pop(memory_id, None)
        self._embeddings.pop(memory_id, None)

    async def create_graveyard_entry(self, entry: GraveyardEntry) -> GraveyardEntry:
        self._graveyard[entry.id] = entry
        return entry

    async def get_graveyard_entry(self, entry_id: str) -> GraveyardEntry | None:
        return self._graveyard.get(entry_id)

    async def save_contradiction(self, contradiction: Contradiction) -> Contradiction:
        self._contradictions[contradiction.id] = contradiction
        return contradiction

    async def get_contradiction(self, contradiction_id: str) -> Contradiction | None:
        return self._contradictions.get(contradiction_id)

    async def list_contradiction_candidates(
        self,
        namespace_id: str,
        *,
        similarity_threshold: float,
    ) -> list[tuple[Memory, Memory, float]]:
        memories = self._filter_memories(namespace_id, include_archived=False)
        candidates: list[tuple[Memory, Memory, float]] = []

        for index, memory_a in enumerate(memories):
            for memory_b in memories[index + 1 :]:
                if memory_a.content_hash == memory_b.content_hash:
                    continue
                similarity = self._pair_similarity(memory_a, memory_b)
                if similarity >= similarity_threshold:
                    candidates.append((memory_a, memory_b, similarity))

        return candidates

    def _filter_memories(
        self,
        namespace_id: str,
        *,
        include_archived: bool,
        types: Sequence[MemoryType] | None = None,
        tiers: Sequence[MemoryTier] | None = None,
        tags: Sequence[str] | None = None,
    ) -> list[Memory]:
        allowed_statuses = {MemoryStatus.ACTIVE}
        if include_archived:
            allowed_statuses.add(MemoryStatus.ARCHIVED)

        return [
            memory
            for memory in self._memories.values()
            if memory.namespace_id == namespace_id
            and memory.status in allowed_statuses
            and memory.status is not MemoryStatus.GRAVEYARD
            and (not types or memory.type in types)
            and (not tiers or memory.tier in tiers)
            and (not tags or set(tags).issubset(set(memory.tags)))
        ]

    def _score(
        self,
        memory: Memory,
        query: str,
        query_embedding: Sequence[float] | None,
    ) -> float:
        if query_embedding is not None and memory.id in self._embeddings:
            return self._cosine_similarity(query_embedding, self._embeddings[memory.id])

        query_terms = self._tokenize(query)
        memory_terms = self._tokenize(memory.content)
        if not query_terms or not memory_terms:
            return 0.0

        overlap = len(query_terms & memory_terms)
        return overlap / max(len(query_terms), 1)

    def _pair_similarity(self, memory_a: Memory, memory_b: Memory) -> float:
        embedding_a = self._embeddings.get(memory_a.id)
        embedding_b = self._embeddings.get(memory_b.id)
        if embedding_a is not None and embedding_b is not None:
            return self._cosine_similarity(embedding_a, embedding_b)

        terms_a = self._tokenize(memory_a.content)
        terms_b = self._tokenize(memory_b.content)
        union = terms_a | terms_b
        if not union:
            return 0.0
        return len(terms_a & terms_b) / len(union)

    def _cosine_similarity(
        self,
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        if len(left) != len(right) or not left:
            return 0.0

        numerator = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in text.lower().split() if token}
