"""Domain models for the standalone memory engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import StrEnum
from typing import Any


class NamespaceType(StrEnum):
    AGENT = "AGENT"
    PROJECT = "PROJECT"
    TEAM = "TEAM"
    USER = "USER"


class MemoryType(StrEnum):
    FACT = "FACT"
    DECISION = "DECISION"
    LEARNING = "LEARNING"
    PREFERENCE = "PREFERENCE"
    TODO = "TODO"
    CONTEXT = "CONTEXT"


class MemoryScope(StrEnum):
    AGENT = "AGENT"
    PROJECT = "PROJECT"
    TEAM = "TEAM"
    USER = "USER"


class MemoryTier(StrEnum):
    CRITICAL = "CRITICAL"
    DAILY = "DAILY"
    ARCHIVE = "ARCHIVE"


class MemoryStatus(StrEnum):
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    GRAVEYARD = "GRAVEYARD"


class GraveyardReason(StrEnum):
    DUPLICATE = "DUPLICATE"
    SUPERSEDED = "SUPERSEDED"
    CONTRADICTION_LOSER = "CONTRADICTION_LOSER"
    MERGED_INTO_OTHER = "MERGED_INTO_OTHER"
    EXPIRED = "EXPIRED"
    INVALID_REFERENCE = "INVALID_REFERENCE"
    MANUAL = "MANUAL"


class ContradictionStatus(StrEnum):
    DETECTED = "DETECTED"
    RESOLVED = "RESOLVED"
    IGNORED = "IGNORED"


class ContradictionResolution(StrEnum):
    NEWER = "NEWER"
    HIGHER_CONFIDENCE = "HIGHER_CONFIDENCE"
    MERGE = "MERGE"
    MANUAL = "MANUAL"


@dataclass(slots=True)
class Namespace:
    id: str
    slug: str
    name: str
    type: NamespaceType = NamespaceType.PROJECT
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class Memory:
    id: str
    namespace_id: str
    content: str
    content_hash: str
    title: str | None = None
    type: MemoryType = MemoryType.FACT
    scope: MemoryScope = MemoryScope.PROJECT
    category: str | None = None
    source: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Stable semantic identity for an evolving memory.  This is deliberately
    # separate from the storage id: a new observation can replace an older
    # memory while keeping the identity of the thing being remembered.
    memory_key: str | None = None
    supersedes_memory_key: str | None = None
    provenance_key: str | None = None
    observed_at: datetime | None = None
    confidence: float = 1.0
    relevance_boost: float = 1.0
    access_count: int = 0
    last_accessed_at: datetime | None = None
    tier: MemoryTier = MemoryTier.ARCHIVE
    status: MemoryStatus = MemoryStatus.ACTIVE
    related_memory_ids: list[str] = field(default_factory=list)
    document_refs: list[str] = field(default_factory=list)
    invalid_document_refs: list[str] = field(default_factory=list)
    expires_at: datetime | None = None
    journal_date: date | None = None
    contradiction_count: int = 0
    superseded_by_id: str | None = None
    restored_from_graveyard_id: str | None = None
    promoted_at: datetime | None = None
    promoted_by: str | None = None
    buried_at: datetime | None = None
    buried_reason: GraveyardReason | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def is_recallable(self) -> bool:
        return self.status is not MemoryStatus.GRAVEYARD


@dataclass(slots=True)
class GraveyardEntry:
    id: str
    namespace_id: str
    original_memory_id: str | None
    content: str
    content_hash: str
    reason: GraveyardReason
    buried_at: datetime
    replaced_by_id: str | None = None
    contradiction_id: str | None = None
    title: str | None = None
    type: MemoryType = MemoryType.FACT
    scope: MemoryScope = MemoryScope.PROJECT
    category: str | None = None
    source: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_key: str | None = None
    supersedes_memory_key: str | None = None
    provenance_key: str | None = None
    observed_at: datetime | None = None
    confidence: float = 1.0
    previous_tier: MemoryTier | None = None
    previous_status: MemoryStatus | None = None
    related_memory_ids: list[str] = field(default_factory=list)
    document_refs: list[str] = field(default_factory=list)
    invalid_document_refs: list[str] = field(default_factory=list)
    restore_hint: str | None = None
    snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Contradiction:
    id: str
    namespace_id: str
    memory_a_id: str
    memory_b_id: str
    similarity: float
    status: ContradictionStatus
    created_at: datetime
    memory_a_summary: str | None = None
    memory_b_summary: str | None = None
    contradiction_kind: str | None = None
    detected_by: str | None = None
    resolution: ContradictionResolution | None = None
    winner_memory_id: str | None = None
    loser_memory_id: str | None = None
    rationale: str | None = None
    resolved_by: str | None = None
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StoreMemoryRequest:
    namespace_id: str
    content: str
    title: str | None = None
    memory_type: MemoryType = MemoryType.FACT
    scope: MemoryScope = MemoryScope.PROJECT
    category: str | None = None
    source: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_key: str | None = None
    supersedes_memory_key: str | None = None
    provenance_key: str | None = None
    observed_at: datetime | None = None
    confidence: float = 0.7
    relevance_boost: float = 1.0
    tier: MemoryTier | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    related_memory_ids: list[str] = field(default_factory=list)
    document_refs: list[str] = field(default_factory=list)
    invalid_document_refs: list[str] = field(default_factory=list)
    expires_at: datetime | None = None
    journal_date: date | None = None
    embedding: list[float] | None = None
    memory_id: str | None = None


@dataclass(slots=True)
class RecallQuery:
    namespace_id: str
    query: str
    limit: int = 10
    min_confidence: float = 0.0
    include_archived: bool = False
    types: list[MemoryType] = field(default_factory=list)
    tiers: list[MemoryTier] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # Ask the store for more candidates than the final answer budget when a
    # caller wants provenance diversity or duplicate-evidence filtering.
    candidate_limit: int | None = None
    diversify_by_provenance: bool = False
    max_per_provenance: int | None = None
    deduplicate_evidence: bool = False
    # When enabled, include nearby evidence from a provenance group that
    # already produced a query match. This resolves cross-turn references
    # without forcing callers to store raw transcripts.
    include_provenance_context: bool = False
    provenance_context_limit: int | None = None
    # Bound how many provenance groups can contribute sibling context. This
    # prevents a broad candidate search from being flooded by weak groups.
    provenance_context_group_limit: int | None = None


@dataclass(slots=True)
class RecallMatch:
    memory: Memory
    score: float
    reason: str | None = None


@dataclass(slots=True)
class SessionMemoryBundle:
    namespace_id: str
    critical: list[Memory] = field(default_factory=list)
    daily: list[Memory] = field(default_factory=list)
    archive: list[Memory] = field(default_factory=list)

    def all_memories(self) -> list[Memory]:
        return [*self.critical, *self.daily, *self.archive]


@dataclass(slots=True)
class ResolveContradictionRequest:
    contradiction_id: str
    resolution: ContradictionResolution
    resolved_by: str | None = None
    rationale: str | None = None
    winner_memory_id: str | None = None
    merged_content: str | None = None


@dataclass(slots=True)
class CompactionResult:
    namespace_id: str
    initial_count: int
    final_count: int
    duplicates_removed: int = 0
    archived_count: int = 0
