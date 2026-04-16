"""Persistence contract for the standalone memory engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..domain.models import Contradiction, GraveyardEntry, Memory, MemoryStatus, MemoryTier, MemoryType, RecallMatch, RecallQuery


class MemoryStore(Protocol):
    """Storage abstraction used by the domain service."""

    async def create_memory(
        self,
        memory: Memory,
        *,
        embedding: Sequence[float] | None = None,
    ) -> Memory: ...

    async def create_memories(
        self,
        memories: Sequence[Memory],
        *,
        embeddings: Sequence[Sequence[float] | None] | None = None,
    ) -> list[Memory]: ...

    async def search(
        self,
        query: RecallQuery,
        *,
        query_embedding: Sequence[float] | None = None,
    ) -> list[RecallMatch]: ...

    async def list_memories(
        self,
        namespace_id: str,
        *,
        statuses: Sequence[MemoryStatus] | None = None,
        tiers: Sequence[MemoryTier] | None = None,
        types: Sequence[MemoryType] | None = None,
        limit: int | None = None,
    ) -> list[Memory]: ...

    async def get_memory(self, memory_id: str) -> Memory | None: ...

    async def update_memory(self, memory: Memory) -> Memory: ...

    async def delete_memory(self, memory_id: str) -> None: ...

    async def create_graveyard_entry(self, entry: GraveyardEntry) -> GraveyardEntry: ...

    async def get_graveyard_entry(self, entry_id: str) -> GraveyardEntry | None: ...

    async def save_contradiction(self, contradiction: Contradiction) -> Contradiction: ...

    async def get_contradiction(self, contradiction_id: str) -> Contradiction | None: ...

    async def list_contradiction_candidates(
        self,
        namespace_id: str,
        *,
        similarity_threshold: float,
    ) -> list[tuple[Memory, Memory, float]]: ...
