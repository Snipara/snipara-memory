"""A bounded, provenance-first evidence graph for the memory engine.

The graph is deliberately storage-neutral.  Nodes and edges can be projected
to PostgreSQL rows or kept in memory for local retrieval; no graph database is
required.  The graph never invents entity merges: entity links are accepted
only from explicit extractor metadata.
"""

from __future__ import annotations

import heapq
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from itertools import pairwise
from typing import Any

from .domain.models import Memory, RecallMatch


class EvidenceNodeKind(StrEnum):
    SOURCE = "source"
    SESSION = "session"
    TURN = "turn"
    ENTITY = "entity"
    FACT = "fact"
    EVENT = "event"
    QUESTION = "question"
    ANSWER_CANDIDATE = "answer_candidate"


class EvidenceRelation(StrEnum):
    MENTIONS = "mentions"
    ASSERTS = "asserts"
    SAME_AS = "same_as"
    SUPERSEDES = "supersedes"
    CONTRADICTS = "contradicts"
    BEFORE = "before"
    AFTER = "after"
    CONTRIBUTES_TO = "contributes_to"
    DERIVED_FROM = "derived_from"


@dataclass(slots=True)
class EvidenceNode:
    id: str
    kind: EvidenceNodeKind
    label: str | None = None
    memory_id: str | None = None
    provenance_key: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceEdge:
    source_id: str
    target_id: str
    relation: EvidenceRelation
    weight: float = 1.0
    provenance_memory_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvidencePath:
    node_ids: tuple[str, ...]
    relations: tuple[EvidenceRelation, ...]
    score: float
    seed_memory_id: str


@dataclass(frozen=True, slots=True)
class NumericContribution:
    """One typed numeric contribution with an auditable source."""

    value: Decimal
    unit: str | None
    source_memory_id: str
    provenance_key: str
    content: str
    identity: str
    observed_at: datetime | None = None
    operation_scope: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": str(self.value),
            "unit": self.unit,
            "source_memory_id": self.source_memory_id,
            "provenance_key": self.provenance_key,
            "content": self.content,
            "identity": self.identity,
            "observed_at": (
                self.observed_at.isoformat() if self.observed_at is not None else None
            ),
            "operation_scope": self.operation_scope,
        }


class AnswerabilityStatus(StrEnum):
    SUPPORTED = "supported"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class ReasoningResult:
    status: AnswerabilityStatus
    operation: str
    value: Decimal | int | None
    unit: str | None
    contributions: tuple[NumericContribution, ...] = ()
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value: str | int | None
        if isinstance(self.value, Decimal):
            value = str(self.value)
        else:
            value = self.value
        return {
            "status": self.status.value,
            "operation": self.operation,
            "value": value,
            "unit": self.unit,
            "reason": self.reason,
            "contributions": [item.as_dict() for item in self.contributions],
        }


_NUMERIC_TOKEN = re.compile(
    r"(?P<currency>[$€£])?\s*(?P<number>\d+(?:[,.]\d{1,2})?)\s*(?P<unit>%|kg|g|km|mi|hours?|hrs?|minutes?|mins?|days?|weeks?|months?|years?)?",
    re.IGNORECASE,
)
_GENERIC_ENTITY_KEYS = frozenset({"user", "person", "thing", "item", "object"})
_EXPANSION_RELATIONS = frozenset(
    {
        EvidenceRelation.CONTRADICTS,
        EvidenceRelation.CONTRIBUTES_TO,
        EvidenceRelation.MENTIONS,
        EvidenceRelation.SAME_AS,
        EvidenceRelation.SUPERSEDES,
    }
)


def _as_json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (EvidenceNodeKind, EvidenceRelation, AnswerabilityStatus)):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _as_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json_value(item) for item in value]
    return value


class EvidenceGraph:
    """Small in-process graph with bounded pivot expansion.

    ``to_rows`` and ``from_rows`` make the same representation suitable for a
    PostgreSQL adapter.  The traversal is intentionally local and bounded:
    seeds come from hybrid retrieval, then only a limited frontier is
    expanded.  This keeps the graph additive to the existing engine.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, EvidenceNode] = {}
        self.edges: list[EvidenceEdge] = []
        self._edge_keys: set[tuple[str, str, EvidenceRelation]] = set()
        self.memories: dict[str, Memory] = {}

    def add_node(self, node: EvidenceNode) -> EvidenceNode:
        existing = self.nodes.get(node.id)
        if existing is None:
            self.nodes[node.id] = node
            return node
        if existing.label is None:
            existing.label = node.label
        if existing.provenance_key is None:
            existing.provenance_key = node.provenance_key
        existing.attributes.update(
            {key: value for key, value in node.attributes.items() if value is not None}
        )
        return existing

    def add_edge(self, edge: EvidenceEdge) -> EvidenceEdge:
        key = (edge.source_id, edge.target_id, edge.relation)
        if key not in self._edge_keys:
            self.edges.append(edge)
            self._edge_keys.add(key)
        return edge

    @classmethod
    def from_memories(cls, memories: Sequence[Memory]) -> EvidenceGraph:
        graph = cls()
        by_memory_key: dict[str, list[Memory]] = defaultdict(list)
        by_session: dict[str, list[Memory]] = defaultdict(list)
        by_entity_key: dict[str, list[str]] = defaultdict(list)

        for memory in memories:
            graph.memories[memory.id] = memory
            metadata = memory.metadata or {}
            session_id = str(
                metadata.get("source_session_id")
                or memory.provenance_key
                or "unknown"
            )
            source_ref = str(
                (memory.document_refs[0] if memory.document_refs else None)
                or memory.source
                or session_id
            )
            fact_kind = (
                EvidenceNodeKind.EVENT
                if metadata.get("evidence_kind") == "event"
                else EvidenceNodeKind.FACT
            )
            graph.add_node(
                EvidenceNode(
                    id=f"memory:{memory.id}",
                    kind=fact_kind,
                    label=memory.title or memory.content[:120],
                    memory_id=memory.id,
                    provenance_key=session_id,
                    attributes={
                        "memory_key": memory.memory_key,
                        "content_hash": memory.content_hash,
                        "observed_at": memory.observed_at,
                        "evidence_kind": metadata.get("evidence_kind"),
                    },
                )
            )
            graph.add_node(
                EvidenceNode(
                    id=f"session:{session_id}",
                    kind=EvidenceNodeKind.SESSION,
                    label=session_id,
                    provenance_key=session_id,
                )
            )
            graph.add_node(
                EvidenceNode(
                    id=f"source:{source_ref}",
                    kind=EvidenceNodeKind.SOURCE,
                    label=source_ref,
                    provenance_key=session_id,
                )
            )
            graph.add_edge(
                EvidenceEdge(
                    source_id=f"memory:{memory.id}",
                    target_id=f"session:{session_id}",
                    relation=EvidenceRelation.ASSERTS,
                    provenance_memory_id=memory.id,
                )
            )
            graph.add_edge(
                EvidenceEdge(
                    source_id=f"memory:{memory.id}",
                    target_id=f"source:{source_ref}",
                    relation=EvidenceRelation.DERIVED_FROM,
                    provenance_memory_id=memory.id,
                )
            )

            turn_indices = _metadata_ints(metadata.get("source_turn_indices"))
            for turn_index in turn_indices:
                turn_id = f"turn:{session_id}:{turn_index}"
                graph.add_node(
                    EvidenceNode(
                        id=turn_id,
                        kind=EvidenceNodeKind.TURN,
                        label=f"{session_id} turn {turn_index}",
                        provenance_key=session_id,
                        attributes={"turn_index": turn_index},
                    )
                )
                graph.add_edge(
                    EvidenceEdge(
                        source_id=f"memory:{memory.id}",
                        target_id=turn_id,
                        relation=EvidenceRelation.DERIVED_FROM,
                        provenance_memory_id=memory.id,
                    )
                )

            for entity_key in _explicit_entity_keys(metadata):
                entity_id = f"entity:{entity_key}"
                by_entity_key[entity_key].append(memory.id)
                graph.add_node(
                    EvidenceNode(
                        id=entity_id,
                        kind=EvidenceNodeKind.ENTITY,
                        label=entity_key,
                        provenance_key=session_id,
                    )
                )
                graph.add_edge(
                    EvidenceEdge(
                        source_id=f"memory:{memory.id}",
                        target_id=entity_id,
                        relation=EvidenceRelation.MENTIONS,
                        provenance_memory_id=memory.id,
                    )
                )

            for relation, metadata_key in (
                (EvidenceRelation.CONTRADICTS, "contradicts_memory_ids"),
                (EvidenceRelation.SAME_AS, "same_as_memory_ids"),
                (EvidenceRelation.CONTRIBUTES_TO, "contributes_to_memory_ids"),
            ):
                for target_memory_id in _metadata_refs(metadata.get(metadata_key)):
                    graph.add_edge(
                        EvidenceEdge(
                            source_id=f"memory:{memory.id}",
                            target_id=f"memory:{target_memory_id}",
                            relation=relation,
                            provenance_memory_id=memory.id,
                        )
                    )

            if memory.memory_key:
                by_memory_key[memory.memory_key].append(memory)
            by_session[session_id].append(memory)

        for entity_key, memory_ids in by_entity_key.items():
            if entity_key.lower() in _GENERIC_ENTITY_KEYS:
                continue
            for left, right in pairwise(memory_ids):
                graph.add_edge(
                    EvidenceEdge(
                        source_id=f"memory:{left}",
                        target_id=f"memory:{right}",
                        relation=EvidenceRelation.SAME_AS,
                        weight=0.9,
                        provenance_memory_id=left,
                        attributes={"entity_key": entity_key},
                    )
                )

        for memory in memories:
            if not memory.supersedes_memory_key:
                continue
            for previous in by_memory_key.get(memory.supersedes_memory_key, ()):
                if previous.id == memory.id:
                    continue
                graph.add_edge(
                    EvidenceEdge(
                        source_id=f"memory:{memory.id}",
                        target_id=f"memory:{previous.id}",
                        relation=EvidenceRelation.SUPERSEDES,
                        provenance_memory_id=memory.id,
                    )
                )

        # Temporal edges are created only when both facts carry explicit turn
        # indexes in the same session. This avoids inventing order from a
        # storage insertion timestamp.
        for session_memories in by_session.values():
            ordered = sorted(
                (
                    (min(_metadata_ints(memory.metadata.get("source_turn_indices"))), memory)
                    for memory in session_memories
                    if _metadata_ints(memory.metadata.get("source_turn_indices"))
                ),
                key=lambda item: (item[0], item[1].id),
            )
            for (_, earlier), (_, later) in pairwise(ordered):
                graph.add_edge(
                    EvidenceEdge(
                        source_id=f"memory:{earlier.id}",
                        target_id=f"memory:{later.id}",
                        relation=EvidenceRelation.BEFORE,
                        provenance_memory_id=earlier.id,
                    )
                )

        return graph

    def to_rows(self) -> dict[str, list[dict[str, Any]]]:
        """Return PostgreSQL-friendly node and edge rows without losing audit data."""

        return {
            "nodes": [
                {
                    "id": node.id,
                    "kind": node.kind.value,
                    "label": node.label,
                    "memory_id": node.memory_id,
                    "provenance_key": node.provenance_key,
                    "attributes": _as_json_value(node.attributes),
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation": edge.relation.value,
                    "weight": edge.weight,
                    "provenance_memory_id": edge.provenance_memory_id,
                    "attributes": _as_json_value(edge.attributes),
                }
                for edge in self.edges
            ],
        }

    @classmethod
    def from_rows(
        cls,
        nodes: Sequence[Mapping[str, Any]],
        edges: Sequence[Mapping[str, Any]],
        *,
        memories: Sequence[Memory] = (),
    ) -> EvidenceGraph:
        graph = cls()
        graph.memories = {memory.id: memory for memory in memories}
        for row in nodes:
            graph.add_node(
                EvidenceNode(
                    id=str(row["id"]),
                    kind=EvidenceNodeKind(str(row["kind"])),
                    label=row.get("label"),
                    memory_id=row.get("memory_id"),
                    provenance_key=row.get("provenance_key"),
                    attributes=dict(row.get("attributes") or {}),
                )
            )
        for row in edges:
            graph.add_edge(
                EvidenceEdge(
                    source_id=str(row["source_id"]),
                    target_id=str(row["target_id"]),
                    relation=EvidenceRelation(str(row["relation"])),
                    weight=float(row.get("weight", 1.0)),
                    provenance_memory_id=row.get("provenance_memory_id"),
                    attributes=dict(row.get("attributes") or {}),
                )
            )
        return graph

    def expand_matches(
        self,
        seeds: Sequence[RecallMatch],
        *,
        limit: int,
        max_hops: int = 2,
        max_nodes: int = 128,
        pivot_width: int = 16,
        hop_decay: float = 0.72,
    ) -> list[RecallMatch]:
        """Expand only a bounded frontier around hybrid-retrieval seeds.

        The frontier is processed in bounded pivot batches. A graph node is
        traversed in both directions so provenance/session/entity hubs can
        connect related facts, while only fact/event nodes are returned to the
        reader.
        """

        if limit <= 0 or max_hops < 0 or max_nodes <= 0 or pivot_width <= 0:
            return []
        adjacency: dict[str, list[tuple[str, EvidenceEdge]]] = defaultdict(list)
        for edge in self.edges:
            adjacency[edge.source_id].append((edge.target_id, edge))
            adjacency[edge.target_id].append((edge.source_id, edge))

        seed_scores = {
            f"memory:{match.memory.id}": max(0.0, float(match.score))
            for match in seeds
            if match.memory.id in self.memories
        }
        if not seed_scores:
            return []
        best: dict[
            str, tuple[float, int, str, tuple[str, ...], tuple[EvidenceRelation, ...]]
        ] = {}
        heap: list[
            tuple[
                float,
                int,
                int,
                str,
                str,
                tuple[str, ...],
                tuple[EvidenceRelation, ...],
            ]
        ] = []
        sequence = 0
        for node_id, score in seed_scores.items():
            path = (node_id,)
            relations: tuple[EvidenceRelation, ...] = ()
            seed_memory_id = node_id.removeprefix("memory:")
            best[node_id] = (score, 0, seed_memory_id, path, relations)
            heapq.heappush(
                heap,
                (-score, 0, sequence, node_id, seed_memory_id, path, relations),
            )

        expanded_nodes = 0
        while heap and expanded_nodes < max_nodes:
            batch: list[
                tuple[
                    float,
                    int,
                    int,
                    str,
                    str,
                    tuple[str, ...],
                    tuple[EvidenceRelation, ...],
                ]
            ] = []
            while heap and len(batch) < pivot_width:
                batch.append(heapq.heappop(heap))
            for negative_score, hops, _sequence, node_id, seed_id, path, relations in batch:
                current_score = -negative_score
                previous = best.get(node_id)
                if previous is not None and current_score + 1e-12 < previous[0]:
                    continue
                expanded_nodes += 1
                if hops >= max_hops:
                    continue
                current_node = self.nodes.get(node_id)
                if (
                    current_node is not None
                    and current_node.kind is EvidenceNodeKind.ENTITY
                    and (current_node.label or "").lower() in _GENERIC_ENTITY_KEYS
                ):
                    # Generic hubs such as ``user`` connect almost every
                    # memory and create noisy cross-topic paths. They remain
                    # in the graph for audit/export but are not traversal
                    # pivots.
                    continue
                if current_node is not None and current_node.kind in {
                    EvidenceNodeKind.SOURCE,
                    EvidenceNodeKind.TURN,
                }:
                    # Provenance nodes are retained for auditability, but
                    # traversing through them turns one source/session into a
                    # dense cross-topic hub. Evidence propagation must use an
                    # explicit entity or relation edge instead.
                    continue
                for neighbor_id, edge in adjacency.get(node_id, ()):
                    if edge.relation not in _EXPANSION_RELATIONS:
                        continue
                    if neighbor_id in path:
                        continue
                    next_hops = hops + 1
                    next_score = current_score * hop_decay * max(edge.weight, 0.01)
                    if neighbor_id in seed_scores:
                        next_score = max(next_score, seed_scores[neighbor_id])
                    old = best.get(neighbor_id)
                    if old is not None and next_score <= old[0] + 1e-12:
                        continue
                    next_path = (*path, neighbor_id)
                    next_relations = (*relations, edge.relation)
                    best[neighbor_id] = (
                        next_score,
                        next_hops,
                        seed_id,
                        next_path,
                        next_relations,
                    )
                    sequence += 1
                    heapq.heappush(
                        heap,
                        (
                            -next_score,
                            next_hops,
                            sequence,
                            neighbor_id,
                            seed_id,
                            next_path,
                            next_relations,
                        )
                    )

        selected: dict[str, RecallMatch] = {
            match.memory.id: match for match in seeds if match.memory.id in self.memories
        }
        for node_id, (score, hops, seed_id, _path, relations) in best.items():
            node = self.nodes.get(node_id)
            if node is None or node.kind not in {
                EvidenceNodeKind.FACT,
                EvidenceNodeKind.EVENT,
            } or node.memory_id is None:
                continue
            memory = self.memories.get(node.memory_id)
            if memory is None:
                continue
            existing = selected.get(memory.id)
            reason = f"evidence-graph:hops={hops}:seed={seed_id}"
            if existing is None or score > existing.score:
                selected[memory.id] = RecallMatch(
                    memory=memory,
                    score=score,
                    reason=reason,
                )
        return sorted(
            selected.values(),
            key=lambda match: (match.score, match.memory.confidence, match.memory.id),
            reverse=True,
        )[:limit]


def _metadata_ints(value: Any) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(sorted({int(item) for item in value if str(item).lstrip("-").isdigit()}))


def _metadata_refs(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item or "").strip()))


def _explicit_entity_keys(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    raw = metadata.get("entity_keys", metadata.get("entities", ()))
    if isinstance(raw, str):
        raw = (raw,)
    if not isinstance(raw, (list, tuple, set)):
        raw = (metadata.get("entity_key"),)
    return tuple(dict.fromkeys(str(item).strip() for item in raw if str(item or "").strip()))


def _parse_decimal(value: Any) -> Decimal | None:
    try:
        normalized = str(value).strip().replace(",", "")
        return Decimal(normalized)
    except (InvalidOperation, ValueError):
        return None


def extract_numeric_contributions(
    memories: Sequence[Memory],
    *,
    allow_text_parsing: bool = True,
) -> list[NumericContribution]:
    """Extract explicit typed values while retaining exact source lineage."""

    contributions: list[NumericContribution] = []
    for memory in memories:
        metadata = memory.metadata or {}
        raw_values = metadata.get("numeric_values", metadata.get("quantitative_values"))
        if raw_values is None and metadata.get("numeric_value") is not None:
            raw_values = [metadata["numeric_value"]]
        if isinstance(raw_values, (str, int, float, Decimal)):
            raw_values = [raw_values]
        if not isinstance(raw_values, (list, tuple)) and allow_text_parsing:
            raw_values = list(_NUMERIC_TOKEN.finditer(memory.content))
        if not isinstance(raw_values, (list, tuple)):
            continue
        for index, raw in enumerate(raw_values):
            unit = str(metadata.get("unit") or "").strip() or None
            value: Decimal | None
            if hasattr(raw, "group"):
                number = raw.group("number")
                value = _parse_decimal(number)
                unit = unit or raw.group("unit") or (
                    "currency" if raw.group("currency") else None
                )
            else:
                value = _parse_decimal(raw)
            if value is None:
                continue
            identity = str(
                metadata.get("contribution_id")
                or f"{memory.id}:{index}"
            )
            contributions.append(
                NumericContribution(
                    value=value,
                    unit=unit,
                    source_memory_id=memory.id,
                    provenance_key=(
                        memory.provenance_key
                        or str(metadata.get("source_session_id") or memory.id)
                    ),
                    content=memory.content,
                    identity=identity,
                    observed_at=memory.observed_at,
                    operation_scope=(
                        str(metadata["operation_scope"])
                        if metadata.get("operation_scope") is not None
                        else None
                    ),
                )
            )
    return contributions


def reason_over_contributions(
    operation: str,
    contributions: Sequence[NumericContribution],
    *,
    expected_count: int | None = None,
) -> ReasoningResult:
    """Run a deterministic operation with explicit answerability gates."""

    normalized_operation = operation.lower().strip()
    by_identity: dict[str, list[NumericContribution]] = defaultdict(list)
    for contribution in contributions:
        by_identity[contribution.identity].append(contribution)
    conflicting = [
        item
        for group in by_identity.values()
        if len({(item.value, item.unit) for item in group}) > 1
        for item in group
    ]
    if conflicting:
        return ReasoningResult(
            AnswerabilityStatus.CONFLICTING,
            normalized_operation,
            None,
            None,
            tuple(conflicting),
            reason="conflicting_values_for_same_contribution",
        )
    unique: list[NumericContribution] = []
    seen_identities: set[str] = set()
    for contribution in contributions:
        if contribution.identity in seen_identities:
            continue
        seen_identities.add(contribution.identity)
        unique.append(contribution)
    if not unique:
        return ReasoningResult(
            AnswerabilityStatus.INSUFFICIENT,
            normalized_operation,
            None,
            None,
            reason="no_numeric_contributions",
        )
    units = {item.unit for item in unique}
    if len(units) > 1:
        return ReasoningResult(
            AnswerabilityStatus.UNRESOLVED,
            normalized_operation,
            None,
            None,
            tuple(unique),
            reason="incompatible_units",
        )
    unit = next(iter(units))
    if normalized_operation == "count":
        identities = {item.identity for item in unique}
        if expected_count is not None and len(identities) < expected_count:
            return ReasoningResult(
                AnswerabilityStatus.INSUFFICIENT,
                normalized_operation,
                None,
                unit,
                tuple(unique),
                reason="contribution_count_below_expected",
            )
        return ReasoningResult(
            AnswerabilityStatus.SUPPORTED,
            normalized_operation,
            len(identities),
            unit,
            tuple(unique),
        )
    if normalized_operation == "sum":
        return ReasoningResult(
            AnswerabilityStatus.SUPPORTED,
            normalized_operation,
            sum((item.value for item in unique), Decimal(0)),
            unit,
            tuple(unique),
        )
    if normalized_operation == "difference":
        if len(unique) != 2:
            return ReasoningResult(
                AnswerabilityStatus.INSUFFICIENT,
                normalized_operation,
                None,
                unit,
                tuple(unique),
                reason="difference_requires_two_contributions",
            )
        return ReasoningResult(
            AnswerabilityStatus.SUPPORTED,
            normalized_operation,
            unique[0].value - unique[1].value,
            unit,
            tuple(unique),
        )
    return ReasoningResult(
        AnswerabilityStatus.UNRESOLVED,
        normalized_operation,
        None,
        unit,
        tuple(unique),
        reason="unsupported_operation",
    )
