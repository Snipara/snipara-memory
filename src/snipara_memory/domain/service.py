"""Core orchestration service for the standalone memory engine."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from .models import (
    CompactionResult,
    Contradiction,
    ContradictionResolution,
    ContradictionStatus,
    GraveyardEntry,
    GraveyardReason,
    Memory,
    MemoryScope,
    MemoryStatus,
    MemoryTier,
    MemoryType,
    RecallMatch,
    RecallQuery,
    ResolveContradictionRequest,
    SessionMemoryBundle,
    StoreMemoryRequest,
)
from ..ports.cache import CacheStore
from ..ports.embeddings import EmbeddingsProvider
from ..ports.store import MemoryStore

CONFIDENCE_DECAY_RATE = 0.01
MIN_CONFIDENCE = 0.1


def calculate_confidence_decay(
    initial_confidence: float,
    created_at: datetime,
    last_accessed_at: datetime | None = None,
) -> float:
    """Decay confidence over time while preserving a minimum floor."""
    now = datetime.now(UTC)
    reference_time = last_accessed_at or created_at

    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=UTC)

    days_since_reference = (now - reference_time).days
    decay_factor = (1 - CONFIDENCE_DECAY_RATE) ** days_since_reference
    return max(initial_confidence * decay_factor, MIN_CONFIDENCE)


def classify_memory_tier(
    memory_type: MemoryType,
    confidence: float,
    scope: MemoryScope,
) -> MemoryTier:
    """Classify a memory into the default retrieval tier."""
    if memory_type in {MemoryType.DECISION, MemoryType.PREFERENCE}:
        return MemoryTier.CRITICAL
    if scope in {MemoryScope.AGENT, MemoryScope.USER} and confidence >= 0.8:
        return MemoryTier.DAILY
    if memory_type in {MemoryType.TODO, MemoryType.CONTEXT}:
        return MemoryTier.DAILY
    return MemoryTier.ARCHIVE


class MemoryService:
    """Standalone domain service for storage, recall and lifecycle handling."""

    def __init__(
        self,
        store: MemoryStore,
        embeddings: EmbeddingsProvider | None = None,
        cache: CacheStore | None = None,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._cache = cache

    async def store_memory(self, request: StoreMemoryRequest) -> Memory:
        memory = self._build_memory(request)
        embedding = request.embedding

        if embedding is None and self._embeddings is not None:
            embedding = await self._embeddings.embed_text(request.content)

        created = await self._store.create_memory(memory, embedding=embedding)
        await self._invalidate_namespace_cache(created.namespace_id)
        return created

    async def store_memories_bulk(
        self,
        requests: list[StoreMemoryRequest],
    ) -> list[Memory]:
        if not requests:
            return []

        memories = [self._build_memory(request) for request in requests]
        embeddings = [request.embedding for request in requests]

        if self._embeddings is not None and any(embedding is None for embedding in embeddings):
            generated = await self._embeddings.embed_batch([request.content for request in requests])
            embeddings = [
                explicit if explicit is not None else generated[index]
                for index, explicit in enumerate(embeddings)
            ]

        created = await self._store.create_memories(memories, embeddings=embeddings)
        for namespace_id in {request.namespace_id for request in requests}:
            await self._invalidate_namespace_cache(namespace_id)
        return created

    async def semantic_recall(self, query: RecallQuery) -> list[RecallMatch]:
        query_embedding: list[float] | None = None
        if self._embeddings is not None:
            query_embedding = await self._embeddings.embed_text(query.query)

        matches = await self._store.search(query, query_embedding=query_embedding)
        now = datetime.now(UTC)
        filtered: list[RecallMatch] = []

        for match in matches:
            decayed_confidence = calculate_confidence_decay(
                match.memory.confidence,
                created_at=match.memory.created_at,
                last_accessed_at=match.memory.last_accessed_at,
            )
            if decayed_confidence < query.min_confidence:
                continue

            touched = replace(
                match.memory,
                confidence=decayed_confidence,
                access_count=match.memory.access_count + 1,
                last_accessed_at=now,
            )
            await self._store.update_memory(touched)
            filtered.append(
                replace(
                    match,
                    memory=touched,
                    score=max(match.score, decayed_confidence),
                )
            )

        return filtered[: query.limit]

    async def list_memories(
        self,
        namespace_id: str,
        *,
        statuses: list[MemoryStatus] | None = None,
        tiers: list[MemoryTier] | None = None,
        types: list[MemoryType] | None = None,
        limit: int | None = None,
    ) -> list[Memory]:
        return await self._store.list_memories(
            namespace_id,
            statuses=statuses,
            tiers=tiers,
            types=types,
            limit=limit,
        )

    async def get_session_memories(
        self,
        namespace_id: str,
        *,
        critical_limit: int = 12,
        daily_limit: int = 20,
        archive_limit: int = 20,
    ) -> SessionMemoryBundle:
        cache_key = (
            f"session:{namespace_id}:"
            f"{critical_limit}:{daily_limit}:{archive_limit}"
        )
        cached = await self._get_cached_bundle(cache_key)
        if cached is not None:
            return cached

        memories = await self._store.list_memories(
            namespace_id,
            statuses=[MemoryStatus.ACTIVE, MemoryStatus.ARCHIVED],
        )

        critical = self._rank_session_memories(
            [memory for memory in memories if memory.tier is MemoryTier.CRITICAL],
        )[:critical_limit]
        daily = self._rank_session_memories(
            [memory for memory in memories if memory.tier is MemoryTier.DAILY],
        )[:daily_limit]
        archive = self._rank_session_memories(
            [
                memory
                for memory in memories
                if memory.tier is MemoryTier.ARCHIVE
                and memory.status is not MemoryStatus.GRAVEYARD
            ],
        )[:archive_limit]

        bundle = SessionMemoryBundle(
            namespace_id=namespace_id,
            critical=critical,
            daily=daily,
            archive=archive,
        )
        if self._cache is not None:
            await self._cache.set(cache_key, bundle, ttl_seconds=300)
        return bundle

    async def move_to_graveyard(
        self,
        memory_id: str,
        *,
        reason: GraveyardReason,
        replaced_by_id: str | None = None,
        contradiction_id: str | None = None,
        restore_hint: str | None = None,
    ) -> GraveyardEntry:
        memory = await self._require_memory(memory_id)
        if memory.status is MemoryStatus.GRAVEYARD:
            raise ValueError(f"Memory is already in graveyard: {memory_id}")

        buried_at = datetime.now(UTC)
        entry = GraveyardEntry(
            id=self._generate_id(),
            namespace_id=memory.namespace_id,
            original_memory_id=memory.id,
            replaced_by_id=replaced_by_id,
            contradiction_id=contradiction_id,
            content=memory.content,
            content_hash=memory.content_hash,
            title=memory.title,
            type=memory.type,
            scope=memory.scope,
            category=memory.category,
            source=memory.source,
            tags=list(memory.tags),
            metadata=dict(memory.metadata),
            confidence=memory.confidence,
            previous_tier=memory.tier,
            previous_status=memory.status,
            related_memory_ids=list(memory.related_memory_ids),
            document_refs=list(memory.document_refs),
            invalid_document_refs=list(memory.invalid_document_refs),
            reason=reason,
            buried_at=buried_at,
            restore_hint=restore_hint,
            snapshot={"memory_id": memory.id, "status": memory.status.value},
        )
        stored_entry = await self._store.create_graveyard_entry(entry)
        await self._store.update_memory(
            replace(
                memory,
                status=MemoryStatus.GRAVEYARD,
                buried_at=buried_at,
                buried_reason=reason,
                superseded_by_id=replaced_by_id,
            )
        )
        await self._invalidate_namespace_cache(memory.namespace_id)
        return stored_entry

    async def restore_from_graveyard(
        self,
        graveyard_entry_id: str,
        *,
        restored_by: str | None = None,
    ) -> Memory:
        entry = await self._store.get_graveyard_entry(graveyard_entry_id)
        if entry is None:
            raise ValueError(f"Unknown graveyard entry: {graveyard_entry_id}")

        restored = Memory(
            id=self._generate_id(),
            namespace_id=entry.namespace_id,
            content=entry.content,
            content_hash=entry.content_hash,
            title=entry.title,
            type=entry.type,
            scope=entry.scope,
            category=entry.category,
            source=entry.source,
            tags=list(entry.tags),
            metadata=dict(entry.metadata),
            confidence=entry.confidence,
            tier=entry.previous_tier or MemoryTier.ARCHIVE,
            status=MemoryStatus.ACTIVE,
            related_memory_ids=list(entry.related_memory_ids),
            document_refs=list(entry.document_refs),
            invalid_document_refs=list(entry.invalid_document_refs),
            restored_from_graveyard_id=entry.id,
            promoted_at=datetime.now(UTC),
            promoted_by=restored_by,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        memory = await self._store.create_memory(restored)
        await self._invalidate_namespace_cache(entry.namespace_id)
        return memory

    async def detect_contradictions(
        self,
        namespace_id: str,
        *,
        similarity_threshold: float = 0.82,
        detected_by: str = "semantic-search",
    ) -> list[Contradiction]:
        candidates = await self._store.list_contradiction_candidates(
            namespace_id,
            similarity_threshold=similarity_threshold,
        )
        contradictions: list[Contradiction] = []

        for memory_a, memory_b, similarity in candidates:
            contradiction = Contradiction(
                id=self._generate_id(),
                namespace_id=namespace_id,
                memory_a_id=memory_a.id,
                memory_b_id=memory_b.id,
                memory_a_summary=self._summarize(memory_a.content),
                memory_b_summary=self._summarize(memory_b.content),
                similarity=similarity,
                contradiction_kind="semantic-overlap",
                detected_by=detected_by,
                status=ContradictionStatus.DETECTED,
                created_at=datetime.now(UTC),
            )
            contradictions.append(await self._store.save_contradiction(contradiction))

        return contradictions

    async def resolve_contradiction(
        self,
        request: ResolveContradictionRequest,
    ) -> Contradiction:
        contradiction = await self._store.get_contradiction(request.contradiction_id)
        if contradiction is None:
            raise ValueError(f"Unknown contradiction: {request.contradiction_id}")

        memory_a = await self._require_memory(contradiction.memory_a_id)
        memory_b = await self._require_memory(contradiction.memory_b_id)
        winner_id: str | None = None
        loser_id: str | None = None

        if request.resolution is ContradictionResolution.MERGE:
            merged_content = request.merged_content or self._merge_content(memory_a, memory_b)
            merged_memory = await self.store_memory(
                StoreMemoryRequest(
                    namespace_id=memory_a.namespace_id,
                    content=merged_content,
                    title=memory_a.title or memory_b.title,
                    memory_type=memory_a.type,
                    scope=memory_a.scope,
                    category=memory_a.category or memory_b.category,
                    source="contradiction-merge",
                    tags=sorted(set(memory_a.tags + memory_b.tags)),
                    metadata={"merged_from": [memory_a.id, memory_b.id]},
                    confidence=max(memory_a.confidence, memory_b.confidence),
                    tier=memory_a.tier,
                )
            )
            winner_id = merged_memory.id
            await self.move_to_graveyard(
                memory_a.id,
                reason=GraveyardReason.MERGED_INTO_OTHER,
                replaced_by_id=winner_id,
                contradiction_id=contradiction.id,
            )
            await self.move_to_graveyard(
                memory_b.id,
                reason=GraveyardReason.MERGED_INTO_OTHER,
                replaced_by_id=winner_id,
                contradiction_id=contradiction.id,
            )
        else:
            winner, loser = self._select_winner(memory_a, memory_b, request)
            winner_id = winner.id
            loser_id = loser.id
            await self.move_to_graveyard(
                loser.id,
                reason=GraveyardReason.CONTRADICTION_LOSER,
                replaced_by_id=winner.id,
                contradiction_id=contradiction.id,
            )

        resolved = replace(
            contradiction,
            status=ContradictionStatus.RESOLVED,
            resolution=request.resolution,
            winner_memory_id=winner_id,
            loser_memory_id=loser_id,
            rationale=request.rationale,
            resolved_by=request.resolved_by,
            resolved_at=datetime.now(UTC),
        )
        return await self._store.save_contradiction(resolved)

    async def compact_memories(
        self,
        namespace_id: str,
        *,
        max_active_memories: int = 500,
    ) -> CompactionResult:
        memories = await self._store.list_memories(
            namespace_id,
            statuses=[MemoryStatus.ACTIVE],
        )
        if len(memories) <= max_active_memories:
            return CompactionResult(
                namespace_id=namespace_id,
                initial_count=len(memories),
                final_count=len(memories),
            )

        duplicate_groups: dict[str, list[Memory]] = defaultdict(list)
        for memory in memories:
            duplicate_groups[memory.content_hash].append(memory)

        duplicates_removed = 0
        for group in duplicate_groups.values():
            if len(group) < 2:
                continue
            ranked = sorted(
                group,
                key=lambda memory: (
                    -memory.confidence,
                    -(memory.access_count),
                    (memory.last_accessed_at or memory.created_at).timestamp(),
                    memory.created_at.timestamp(),
                ),
            )
            for loser in ranked[1:]:
                await self.move_to_graveyard(loser.id, reason=GraveyardReason.DUPLICATE)
                duplicates_removed += 1

        remaining = await self._store.list_memories(
            namespace_id,
            statuses=[MemoryStatus.ACTIVE],
        )
        overflow = max(0, len(remaining) - max_active_memories)
        archived_count = 0

        if overflow > 0:
            archivable = sorted(
                [
                    memory
                    for memory in remaining
                    if memory.tier in {MemoryTier.ARCHIVE, MemoryTier.DAILY}
                ],
                key=lambda memory: (
                    memory.tier is MemoryTier.DAILY,
                    memory.last_accessed_at or memory.created_at,
                    memory.created_at,
                ),
            )
            for memory in archivable[:overflow]:
                await self._store.update_memory(
                    replace(memory, status=MemoryStatus.ARCHIVED)
                )
                archived_count += 1

        final_count = len(
            await self._store.list_memories(
                namespace_id,
                statuses=[MemoryStatus.ACTIVE],
            )
        )
        await self._invalidate_namespace_cache(namespace_id)
        return CompactionResult(
            namespace_id=namespace_id,
            initial_count=len(memories),
            final_count=final_count,
            duplicates_removed=duplicates_removed,
            archived_count=archived_count,
        )

    def _build_memory(self, request: StoreMemoryRequest) -> Memory:
        now = datetime.now(UTC)
        tier = request.tier or classify_memory_tier(
            request.memory_type,
            request.confidence,
            request.scope,
        )
        return Memory(
            id=request.memory_id or self._generate_id(),
            namespace_id=request.namespace_id,
            content=request.content,
            content_hash=self._hash_content(request.content),
            title=request.title,
            type=request.memory_type,
            scope=request.scope,
            category=request.category,
            source=request.source,
            tags=list(request.tags),
            metadata=dict(request.metadata),
            confidence=request.confidence,
            relevance_boost=request.relevance_boost,
            tier=tier,
            status=request.status,
            related_memory_ids=list(request.related_memory_ids),
            document_refs=list(request.document_refs),
            invalid_document_refs=list(request.invalid_document_refs),
            expires_at=request.expires_at,
            journal_date=request.journal_date,
            created_at=now,
            updated_at=now,
        )

    async def _require_memory(self, memory_id: str) -> Memory:
        memory = await self._store.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"Unknown memory: {memory_id}")
        return memory

    async def _get_cached_bundle(self, cache_key: str) -> SessionMemoryBundle | None:
        if self._cache is None:
            return None

        cached = await self._cache.get(cache_key)
        if isinstance(cached, SessionMemoryBundle):
            return cached
        return None

    async def _invalidate_namespace_cache(self, namespace_id: str) -> None:
        if self._cache is None:
            return
        await self._cache.delete_prefix(f"session:{namespace_id}:")

    def _rank_session_memories(self, memories: list[Memory]) -> list[Memory]:
        return sorted(
            memories,
            key=lambda memory: (
                memory.confidence,
                memory.relevance_boost,
                memory.last_accessed_at or memory.created_at,
                memory.created_at,
            ),
            reverse=True,
        )

    def _select_winner(
        self,
        memory_a: Memory,
        memory_b: Memory,
        request: ResolveContradictionRequest,
    ) -> tuple[Memory, Memory]:
        if request.resolution is ContradictionResolution.NEWER:
            ordered = sorted(
                [memory_a, memory_b],
                key=lambda memory: memory.created_at,
                reverse=True,
            )
            return ordered[0], ordered[1]

        if request.resolution is ContradictionResolution.HIGHER_CONFIDENCE:
            ordered = sorted(
                [memory_a, memory_b],
                key=lambda memory: (memory.confidence, memory.created_at),
                reverse=True,
            )
            return ordered[0], ordered[1]

        if request.resolution is ContradictionResolution.MANUAL:
            if request.winner_memory_id not in {memory_a.id, memory_b.id}:
                raise ValueError("Manual contradiction resolution requires a valid winner_memory_id")
            if request.winner_memory_id == memory_a.id:
                return memory_a, memory_b
            return memory_b, memory_a

        raise ValueError(f"Unsupported contradiction resolution: {request.resolution}")

    def _merge_content(self, memory_a: Memory, memory_b: Memory) -> str:
        if memory_a.content == memory_b.content:
            return memory_a.content
        return f"{memory_a.content}\n\n---\n\n{memory_b.content}"

    def _summarize(self, content: str, *, max_length: int = 160) -> str:
        if len(content) <= max_length:
            return content
        return f"{content[: max_length - 3].rstrip()}..."

    def _generate_id(self) -> str:
        return uuid4().hex

    def _hash_content(self, content: str) -> str:
        return hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()
