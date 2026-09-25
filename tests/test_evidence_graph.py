from datetime import UTC, datetime
from decimal import Decimal

import pytest

from snipara_memory import (
    AnswerabilityStatus,
    EvidenceGraph,
    EvidenceRelation,
    InMemoryMemoryStore,
    Memory,
    MemoryService,
    RecallMatch,
    RecallQuery,
    extract_numeric_contributions,
    reason_over_contributions,
)


def _memory(
    memory_id: str,
    content: str,
    *,
    session: str,
    turn: int,
    entities: tuple[str, ...] = (),
    memory_key: str | None = None,
    supersedes: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Memory:
    merged_metadata = {
        "source_session_id": session,
        "source_turn_indices": [turn],
        "entity_keys": list(entities),
    }
    merged_metadata.update(metadata or {})
    return Memory(
        id=memory_id,
        namespace_id="demo",
        content=content,
        content_hash=f"hash-{memory_id}",
        metadata=merged_metadata,
        memory_key=memory_key,
        supersedes_memory_key=supersedes,
        provenance_key=session,
        observed_at=datetime(2024, 1, turn + 1, tzinfo=UTC),
        created_at=datetime(2024, 1, turn + 1, tzinfo=UTC),
    )


def test_graph_preserves_provenance_entities_and_supersession() -> None:
    older = _memory(
        "m1",
        "The user lives in Paris.",
        session="s1",
        turn=0,
        entities=("user", "home-city"),
        memory_key="user.city",
    )
    newer = _memory(
        "m2",
        "The user now lives in Zurich.",
        session="s2",
        turn=0,
        entities=("user", "home-city"),
        memory_key="user.city",
        supersedes="user.city",
    )

    graph = EvidenceGraph.from_memories([older, newer])
    relations = {(edge.relation, edge.source_id, edge.target_id) for edge in graph.edges}

    assert (EvidenceRelation.MENTIONS, "memory:m1", "entity:home-city") in relations
    assert (EvidenceRelation.SUPERSEDES, "memory:m2", "memory:m1") in relations
    assert any(edge.relation is EvidenceRelation.DERIVED_FROM for edge in graph.edges)


def test_graph_expansion_reaches_related_fact_through_bounded_pivot() -> None:
    seed = _memory(
        "m1",
        "The user bought a Sony camera.",
        session="s1",
        turn=0,
        entities=("user", "sony-camera"),
    )
    related = _memory(
        "m2",
        "The user needs a compact Sony-compatible lens.",
        session="s2",
        turn=0,
        entities=("user", "sony-camera"),
    )
    unrelated = _memory(
        "m3",
        "The user prefers pasta.",
        session="s3",
        turn=0,
        entities=("user", "food-preference"),
    )

    graph = EvidenceGraph.from_memories([seed, related, unrelated])
    matches = graph.expand_matches(
        [RecallMatch(memory=seed, score=1.0)],
        limit=5,
        max_hops=2,
        max_nodes=8,
        pivot_width=1,
    )

    ids = {match.memory.id for match in matches}
    assert {"m1", "m2"}.issubset(ids)
    assert "m3" not in ids
    assert any(
        match.memory.id == "m2"
        and match.reason is not None
        and match.reason.startswith("evidence-graph:hops=")
        and match.reason.endswith(":seed=m1")
        for match in matches
    )


def test_graph_rows_round_trip_without_losing_edge_contract() -> None:
    memories = [
        _memory(
            "m1",
            "A documented fact.",
            session="s1",
            turn=2,
            entities=("fact-a",),
        )
    ]
    graph = EvidenceGraph.from_memories(memories)
    rows = graph.to_rows()
    restored = EvidenceGraph.from_rows(rows["nodes"], rows["edges"], memories=memories)

    assert set(restored.nodes) == set(graph.nodes)
    assert {
        (edge.source_id, edge.target_id, edge.relation)
        for edge in restored.edges
    } == {
        (edge.source_id, edge.target_id, edge.relation)
        for edge in graph.edges
    }


def test_numeric_reasoning_is_exact_and_auditable() -> None:
    memories = [
        _memory(
            "m1",
            "The user paid $12.50 for the train.",
            session="s1",
            turn=0,
            metadata={"quantitative_values": ["12.50"], "unit": "USD"},
        ),
        _memory(
            "m2",
            "The user paid $7.50 for lunch.",
            session="s2",
            turn=0,
            metadata={"quantitative_values": ["7.50"], "unit": "USD"},
        ),
    ]

    result = reason_over_contributions("sum", extract_numeric_contributions(memories))

    assert result.status is AnswerabilityStatus.SUPPORTED
    assert result.value == Decimal("20.00")
    assert result.unit == "USD"
    assert len(result.contributions) == 2
    assert all(item.provenance_key in {"s1", "s2"} for item in result.contributions)


def test_numeric_reasoning_abstains_on_incompatible_units() -> None:
    memories = [
        _memory(
            "m1",
            "The package weighs 2 kg.",
            session="s1",
            turn=0,
            metadata={"quantitative_values": ["2"], "unit": "kg"},
        ),
        _memory(
            "m2",
            "The package weighs 4 lb.",
            session="s2",
            turn=0,
            metadata={"quantitative_values": ["4"], "unit": "lb"},
        ),
    ]

    result = reason_over_contributions("sum", extract_numeric_contributions(memories))

    assert result.status is AnswerabilityStatus.UNRESOLVED
    assert result.value is None
    assert result.reason == "incompatible_units"


def test_numeric_reasoning_flags_conflicting_repeated_contribution() -> None:
    memories = [
        _memory(
            "m1",
            "The current budget is $10.",
            session="s1",
            turn=0,
            metadata={
                "quantitative_values": ["10"],
                "unit": "USD",
                "contribution_id": "current-budget",
            },
        ),
        _memory(
            "m2",
            "The current budget is $12.",
            session="s2",
            turn=0,
            metadata={
                "quantitative_values": ["12"],
                "unit": "USD",
                "contribution_id": "current-budget",
            },
        ),
    ]

    result = reason_over_contributions("sum", extract_numeric_contributions(memories))

    assert result.status is AnswerabilityStatus.CONFLICTING
    assert result.value is None
    assert result.reason == "conflicting_values_for_same_contribution"


@pytest.mark.asyncio
async def test_service_graph_recall_is_opt_in_and_bounded() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store=store)
    seed = _memory(
        "m1",
        "The user owns a Sony camera.",
        session="s1",
        turn=0,
        entities=("sony-camera",),
    )
    related = _memory(
        "m2",
        "The user wants a compact lens for the Sony camera.",
        session="s2",
        turn=0,
        entities=("sony-camera",),
    )
    await store.create_memories([seed, related])

    matches = await service.graph_recall(
        RecallQuery(namespace_id="demo", query="Sony camera", limit=5),
        seeds=[RecallMatch(memory=seed, score=1.0)],
        max_nodes=8,
    )

    assert [match.memory.id for match in matches][:2] == ["m1", "m2"]
