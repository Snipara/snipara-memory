from __future__ import annotations

from snipara_memory import (
    ContradictionResolution,
    GraveyardReason,
    InMemoryMemoryStore,
    MemoryService,
    RecallQuery,
    ResolveContradictionRequest,
    StoreMemoryRequest,
)


async def test_store_and_recall_memory() -> None:
    service = MemoryService(store=InMemoryMemoryStore())

    await service.store_memory(
        StoreMemoryRequest(
            namespace_id="demo",
            title="JWT convention",
            content="JWT auth uses RS256 token pairs and refresh tokens.",
        )
    )

    matches = await service.semantic_recall(
        RecallQuery(namespace_id="demo", query="How do we handle JWT auth?")
    )

    assert len(matches) == 1
    assert matches[0].memory.title == "JWT convention"


async def test_detect_and_resolve_contradiction() -> None:
    service = MemoryService(store=InMemoryMemoryStore())

    newer = await service.store_memory(
        StoreMemoryRequest(
            namespace_id="demo",
            content="Rate limiting uses Redis and a sliding window.",
            confidence=0.9,
        )
    )
    older = await service.store_memory(
        StoreMemoryRequest(
            namespace_id="demo",
            content="Rate limiting uses Redis with a sliding window implementation.",
            confidence=0.6,
        )
    )

    contradictions = await service.detect_contradictions("demo", similarity_threshold=0.4)
    assert len(contradictions) == 1

    resolved = await service.resolve_contradiction(
        ResolveContradictionRequest(
            contradiction_id=contradictions[0].id,
            resolution=ContradictionResolution.HIGHER_CONFIDENCE,
            resolved_by="test",
        )
    )

    assert resolved.winner_memory_id == newer.id
    assert resolved.loser_memory_id == older.id

    loser = await service.list_memories("demo", limit=10)
    loser_statuses = {memory.id: memory.status for memory in loser}
    assert newer.id in loser_statuses


async def test_compaction_moves_duplicates_to_graveyard() -> None:
    service = MemoryService(store=InMemoryMemoryStore())

    await service.store_memory(
        StoreMemoryRequest(namespace_id="demo", content="Use RS256 for token signing.")
    )
    duplicate = await service.store_memory(
        StoreMemoryRequest(namespace_id="demo", content="Use RS256 for token signing.")
    )

    result = await service.compact_memories("demo", max_active_memories=1)

    assert result.duplicates_removed == 1

    duplicate_memory = await service._store.get_memory(duplicate.id)  # type: ignore[attr-defined]
    assert duplicate_memory is not None
    assert duplicate_memory.buried_reason == GraveyardReason.DUPLICATE
