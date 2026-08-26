from __future__ import annotations

from datetime import UTC, datetime

from snipara_memory import (
    ContradictionResolution,
    GraveyardReason,
    InMemoryMemoryStore,
    MemoryService,
    MemoryStatus,
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


async def test_explicit_memory_key_supersession_is_core_lifecycle_behavior() -> None:
    service = MemoryService(store=InMemoryMemoryStore())

    older = await service.store_memory(
        StoreMemoryRequest(
            namespace_id="demo",
            content="The deployment target is the staging cluster.",
            memory_key="deployment.target",
            provenance_key="handoff-1",
            observed_at=datetime(2026, 8, 20, tzinfo=UTC),
        )
    )
    newer = await service.store_memory(
        StoreMemoryRequest(
            namespace_id="demo",
            content="The deployment target is the production cluster.",
            memory_key="deployment.target",
            supersedes_memory_key="deployment.target",
            provenance_key="handoff-2",
            observed_at=datetime(2026, 8, 21, tzinfo=UTC),
        )
    )

    active = await service.list_memories("demo", statuses=[MemoryStatus.ACTIVE])
    assert [memory.id for memory in active] == [newer.id]
    buried = await service._store.get_memory(older.id)  # type: ignore[attr-defined]
    assert buried is not None
    assert buried.buried_reason == GraveyardReason.SUPERSEDED
    assert buried.superseded_by_id == newer.id


async def test_recall_can_deduplicate_and_diversify_provenance() -> None:
    service = MemoryService(store=InMemoryMemoryStore())
    for index, (source, content) in enumerate(
        (
            ("session-a", "The API decision uses a stable project token."),
            ("session-a", "The API decision uses a stable project token."),
            ("session-a", "The API decision uses a stable project token format."),
            ("session-b", "The API decision uses a rotating project token."),
        )
    ):
        await service.store_memory(
            StoreMemoryRequest(
                namespace_id="demo",
                content=content,
                provenance_key=source,
                memory_id=f"memory-{index}",
            )
        )

    matches = await service.semantic_recall(
        RecallQuery(
            namespace_id="demo",
            query="API project token decision",
            limit=3,
            candidate_limit=10,
            diversify_by_provenance=True,
            max_per_provenance=2,
            deduplicate_evidence=True,
        )
    )

    assert len(matches) == 3
    assert len({match.memory.provenance_key for match in matches}) == 2
    assert len({match.memory.content_hash for match in matches}) == 3


async def test_recall_can_include_sibling_provenance_context() -> None:
    service = MemoryService(store=InMemoryMemoryStore())
    await service.store_memories_bulk(
        [
            StoreMemoryRequest(
                namespace_id="demo",
                content="Account store: Target.",
                memory_key="store",
                provenance_key="session-a",
            ),
            StoreMemoryRequest(
                namespace_id="demo",
                content="Coupon redeemed.",
                memory_key="coupon",
                provenance_key="session-a",
            ),
            StoreMemoryRequest(
                namespace_id="demo",
                content="Account store: grocery.",
                memory_key="other",
                provenance_key="session-b",
            ),
        ]
    )

    matches = await service.semantic_recall(
        RecallQuery(
            namespace_id="demo",
            query="coupon redeemed",
            limit=2,
            include_provenance_context=True,
        )
    )

    assert [match.memory.memory_key for match in matches] == ["coupon", "store"]
    assert matches[1].reason == "provenance-context"
