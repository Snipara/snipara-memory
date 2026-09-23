"""Core orchestration service for the standalone memory engine."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
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


def provenance_key_for_memory(memory: Memory) -> str:
    """Return the stable evidence group used by configurable recall diversity.

    Callers can provide an explicit ``provenance_key`` for a session, commit,
    document section, or other evidence bundle.  Older memories remain
    compatible through the existing metadata/source/document fields.
    """

    explicit = memory.provenance_key or memory.metadata.get("provenance_key")
    if explicit:
        return str(explicit)
    source_session = memory.metadata.get("source_session_id")
    if source_session:
        return str(source_session)
    source_id = memory.metadata.get("source_id")
    if source_id:
        return str(source_id)
    if memory.source:
        return memory.source
    if memory.document_refs:
        return memory.document_refs[0]
    return memory.id


def select_diverse_matches(
    matches: Sequence[RecallMatch],
    *,
    limit: int,
    max_per_provenance: int | None = None,
    deduplicate_evidence: bool = False,
) -> list[RecallMatch]:
    """Select ranked evidence while optionally preserving source diversity.

    The algorithm is intentionally domain-agnostic: it knows only about
    evidence identity and provenance, not about benchmark categories or query
    wording. Ranked matches are visited round-robin by provenance group, so a
    single verbose source cannot consume the whole context budget. When
    deduplication is enabled, identical evidence is collapsed within one
    provenance group while identical text from distinct source groups remains
    available for corroboration and multi-source questions.
    """

    if limit <= 0:
        return []
    if max_per_provenance is not None and max_per_provenance <= 0:
        raise ValueError("max_per_provenance must be positive when provided")

    grouped: dict[str, list[RecallMatch]] = {}
    seen_hashes_by_provenance: dict[str, set[str]] = defaultdict(set)
    for match in matches:
        provenance = provenance_key_for_memory(match.memory)
        if deduplicate_evidence:
            content_hash = match.memory.content_hash
            if content_hash in seen_hashes_by_provenance[provenance]:
                continue
            seen_hashes_by_provenance[provenance].add(content_hash)
        grouped.setdefault(provenance, []).append(match)

    if not grouped:
        return []
    if max_per_provenance is None:
        # Deduplication alone should not reorder ranked evidence.  Diversity
        # is opt-in through max_per_provenance.
        if not deduplicate_evidence:
            return list(matches)[:limit]
        selected: list[RecallMatch] = []
        seen_hashes_by_provenance: dict[str, set[str]] = defaultdict(set)
        for match in matches:
            provenance = provenance_key_for_memory(match.memory)
            if match.memory.content_hash in seen_hashes_by_provenance[provenance]:
                continue
            seen_hashes_by_provenance[provenance].add(match.memory.content_hash)
            selected.append(match)
            if len(selected) >= limit:
                break
        return selected

    selected: list[RecallMatch] = []
    group_keys = list(grouped)
    round_index = 0
    while len(selected) < limit:
        made_progress = False
        for group_key in group_keys:
            group = grouped[group_key]
            if round_index >= len(group):
                continue
            if max_per_provenance is not None and round_index >= max_per_provenance:
                continue
            selected.append(group[round_index])
            made_progress = True
            if len(selected) >= limit:
                break
        if not made_progress:
            break
        round_index += 1
    return selected[:limit]


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
        """Store a single memory object.

        Persists the memory to the backing store, generates an embedding if none
        was provided and an embeddings provider is available, and invalidates
        the namespace cache so session bundles refresh on next access.
        """
        memory = self._build_memory(request)
        embedding = request.embedding

        if embedding is None and self._embeddings is not None:
            embedding = await self._embeddings.embed_text(request.content)

        created = await self._store.create_memory(memory, embedding=embedding)
        await self._apply_explicit_supersession([created])
        await self._invalidate_namespace_cache(created.namespace_id)
        return created

    async def store_memories_bulk(
        self,
        requests: list[StoreMemoryRequest],
    ) -> list[Memory]:
        """Store multiple memory objects in a single batch.

        More efficient than individual store_memory calls: reuses embeddings
        computation and defers cache invalidation to one operation per namespace.
        """
        if not requests:
            return []

        memories = [self._build_memory(request) for request in requests]
        embeddings = [request.embedding for request in requests]

        if self._embeddings is not None and any(
            embedding is None for embedding in embeddings
        ):
            generated = await self._embeddings.embed_batch(
                [request.content for request in requests]
            )
            embeddings = [
                explicit if explicit is not None else generated[index]
                for index, explicit in enumerate(embeddings)
            ]

        created = await self._store.create_memories(memories, embeddings=embeddings)
        await self._apply_explicit_supersession(created)
        for namespace_id in {request.namespace_id for request in requests}:
            await self._invalidate_namespace_cache(namespace_id)
        return created

    async def semantic_recall(self, query: RecallQuery) -> list[RecallMatch]:
        """Recall memories matching a semantic query.

        Generates an embedding for the query (if an embeddings provider is
        available), searches the store, applies confidence decay to filter
        stale memories below the query's min_confidence threshold, updates
        access counts and timestamps for recall metrics, and returns the top-k
        matches sorted by relevance score.
        """
        query_embedding: list[float] | None = None
        if query.limit <= 0:
            raise ValueError("RecallQuery.limit must be positive")
        if query.candidate_limit is not None and query.candidate_limit < query.limit:
            raise ValueError("RecallQuery.candidate_limit must be at least limit")
        if query.max_per_provenance is not None and query.max_per_provenance <= 0:
            raise ValueError("RecallQuery.max_per_provenance must be positive")
        if (
            query.provenance_context_group_limit is not None
            and query.provenance_context_group_limit <= 0
        ):
            raise ValueError(
                "RecallQuery.provenance_context_group_limit must be positive"
            )
        if query.profile_context_limit is not None and query.profile_context_limit <= 0:
            raise ValueError("RecallQuery.profile_context_limit must be positive")
        if query.source_context_limit is not None and query.source_context_limit <= 0:
            raise ValueError("RecallQuery.source_context_limit must be positive")

        query_embedding_text = query.query
        if self._embeddings is not None:
            query_embedding = await self._embeddings.embed_text(query_embedding_text)

        candidate_limit = query.candidate_limit
        if candidate_limit is None:
            # Hybrid stores may contain source excerpts that are intentionally
            # excluded from direct recall. Over-fetch so those excerpts cannot
            # consume the store's candidate window and hide compact facts.
            candidate_limit = max(query.limit * 4, query.limit)
        search_query = replace(
            query,
            limit=max(query.limit, candidate_limit or query.limit),
        )
        matches = await self._store.search(
            search_query,
            query_embedding=query_embedding,
        )
        eligible: list[RecallMatch] = []
        matched_provenance_scores: dict[str, float] = {}

        direct_source_count = 0
        for match in matches:
            is_source_context = bool(match.memory.metadata.get("source_context"))
            if is_source_context:
                if not query.include_source_context:
                    continue
                if direct_source_count >= (query.source_context_limit or 4):
                    continue
                direct_source_count += 1
            decayed_confidence = calculate_confidence_decay(
                match.memory.confidence,
                created_at=match.memory.created_at,
                last_accessed_at=match.memory.last_accessed_at,
            )
            if decayed_confidence < query.min_confidence:
                continue

            scored = replace(
                match.memory,
                confidence=decayed_confidence,
            )
            scored_match = replace(
                # Confidence is an eligibility gate, not a relevance score.
                # Inflating every positive match by its confidence lets an
                # unrelated high-confidence memory outrank query evidence.
                match,
                memory=scored,
                score=match.score,
            )
            eligible.append(scored_match)
            if query.include_provenance_context:
                provenance = provenance_key_for_memory(scored)
                matched_provenance_scores[provenance] = max(
                    matched_provenance_scores.get(provenance, 0.0),
                    scored_match.score,
                )

        if query.include_provenance_context and matched_provenance_scores:
            existing_ids = {match.memory.id for match in eligible}
            context_limit = query.provenance_context_limit or max(query.limit, 4)
            context_group_limit = query.provenance_context_group_limit or max(
                4, min(16, query.limit // 8 or 1)
            )
            selected_provenances = {
                provenance
                for provenance, _score in sorted(
                    matched_provenance_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:context_group_limit]
            }
            # Keep direct-hit scores and reasons intact. A weak direct hit
            # must not be promoted to the anchor score merely because another
            # memory from the same evidence bundle matched the query; doing
            # so makes an unrelated sibling outrank concrete query evidence.
            # Siblings that were not direct hits are added below with the
            # explicit ``provenance-context`` reason.
            context_counts: dict[str, int] = defaultdict(int)
            active_memories = await self._store.list_memories(
                query.namespace_id,
                statuses=[MemoryStatus.ACTIVE],
            )
            for memory in active_memories:
                if memory.id in existing_ids:
                    continue
                provenance = provenance_key_for_memory(memory)
                if provenance not in selected_provenances:
                    continue
                anchor_score = matched_provenance_scores.get(provenance)
                if anchor_score is None or context_counts[provenance] >= context_limit:
                    continue
                decayed_confidence = calculate_confidence_decay(
                    memory.confidence,
                    created_at=memory.created_at,
                    last_accessed_at=memory.last_accessed_at,
                )
                if decayed_confidence < query.min_confidence:
                    continue
                context_memory = replace(memory, confidence=decayed_confidence)
                eligible.append(
                    RecallMatch(
                        memory=context_memory,
                        # Keep sibling evidence in the same relevance band as
                        # the anchor so a large candidate pool cannot hide it
                        # behind unrelated direct matches, without making a
                        # contextual sibling outrank the direct hit.
                        score=anchor_score,
                        reason="provenance-context",
                    )
                )
                context_counts[provenance] += 1

        if query.include_profile_context:
            existing_ids = {match.memory.id for match in eligible}
            active_memories = await self._store.list_memories(
                query.namespace_id,
                statuses=[MemoryStatus.ACTIVE],
            )
            profile_memories = [
                memory
                for memory in active_memories
                if memory.type is MemoryType.PREFERENCE
                or memory.metadata.get("evidence_kind") == "preference"
            ]
            profile_memories.sort(
                key=lambda memory: (
                    memory.confidence,
                    memory.relevance_boost,
                    self._observation_sort_key(memory),
                ),
                reverse=True,
            )
            for memory in profile_memories[: query.profile_context_limit or 8]:
                if memory.id in existing_ids:
                    continue
                eligible.append(
                    RecallMatch(
                        memory=memory,
                        score=0.25,
                        reason="profile-context",
                    )
                )
                existing_ids.add(memory.id)

        eligible.sort(
            key=lambda match: (
                match.score,
                match.memory.confidence,
                match.memory.relevance_boost,
                self._observation_sort_key(match.memory),
                match.memory.last_accessed_at or datetime.min.replace(tzinfo=UTC),
            ),
            reverse=True,
        )

        reserved_profile: list[RecallMatch] = []
        if query.include_profile_context:
            profile_limit = min(query.profile_context_limit or 8, query.limit)
            reserved_profile = [
                match
                for match in eligible
                if match.memory.type is MemoryType.PREFERENCE
                or match.memory.metadata.get("evidence_kind") == "preference"
            ][:profile_limit]
        reserved_ids = {match.memory.id for match in reserved_profile}
        selected = select_diverse_matches(
            [match for match in eligible if match.memory.id not in reserved_ids],
            limit=query.limit - len(reserved_profile),
            max_per_provenance=(
                (query.max_per_provenance or 1)
                if query.diversify_by_provenance
                else None
            ),
            deduplicate_evidence=query.deduplicate_evidence,
        )
        selected.extend(reserved_profile)
        rank = {match.memory.id: index for index, match in enumerate(eligible)}
        selected.sort(key=lambda match: rank[match.memory.id])
        now = datetime.now(UTC)
        touched_matches: list[RecallMatch] = []
        for match in selected:
            touched = replace(
                match.memory,
                access_count=match.memory.access_count + 1,
                last_accessed_at=now,
            )
            await self._store.update_memory(touched)
            touched_matches.append(replace(match, memory=touched))
        return touched_matches

    async def list_memories(
        self,
        namespace_id: str,
        *,
        statuses: list[MemoryStatus] | None = None,
        tiers: list[MemoryTier] | None = None,
        types: list[MemoryType] | None = None,
        limit: int | None = None,
    ) -> list[Memory]:
        """List memories in a namespace, with optional filters.

        Delegates to the backing store; filters are applied server-side.
        Returns unranked results; use get_session_memories for ranked bundles.
        """
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
        """Build a tiered session memory bundle for agent warm-up.

        Fetches all active and archived memories, ranks them by tier (CRITICAL
        first, then DAILY, then ARCHIVE), applies per-tier limits, caches the
        result, and returns a compact bundle ready for an agent session startup.
        Cache TTL and per-tier budgets are configurable.
        """
        cache_key = (
            f"session:{namespace_id}:{critical_limit}:{daily_limit}:{archive_limit}"
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
            memory_key=memory.memory_key,
            supersedes_memory_key=memory.supersedes_memory_key,
            provenance_key=memory.provenance_key,
            observed_at=memory.observed_at,
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
            memory_key=entry.memory_key,
            supersedes_memory_key=entry.supersedes_memory_key,
            provenance_key=entry.provenance_key,
            observed_at=entry.observed_at,
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
            merged_content = request.merged_content or self._merge_content(
                memory_a, memory_b
            )
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

    async def _apply_explicit_supersession(
        self,
        created: Sequence[Memory],
    ) -> tuple[str, ...]:
        """Apply explicit memory-key replacements through the graveyard.

        Supersession is an ingestion concern, not a benchmark concern.  A
        caller must opt in by naming the identity it replaces; unrelated facts
        and repeated evidence remain active.  The local map also handles two
        successive updates arriving in one bulk import.
        """

        if not any(
            memory.memory_key or memory.supersedes_memory_key for memory in created
        ):
            return ()

        superseded_ids: list[str] = []
        for namespace_id in {memory.namespace_id for memory in created}:
            active = await self._store.list_memories(
                namespace_id,
                statuses=[MemoryStatus.ACTIVE],
            )
            # Bulk-created records are already visible in the store, but must
            # enter this index in input order. Indexing them all up front can
            # leave a key pointing at a record buried earlier in this loop.
            created_ids = {memory.id for memory in created}
            current_by_key = {
                memory.memory_key: memory
                for memory in active
                if memory.memory_key and memory.id not in created_ids
            }
            for memory in (
                item for item in created if item.namespace_id == namespace_id
            ):
                supersedes_key = memory.supersedes_memory_key
                buried = False
                if supersedes_key:
                    previous = current_by_key.get(supersedes_key)
                    if previous is not None and previous.id != memory.id:
                        if self._observation_sort_key(
                            memory
                        ) >= self._observation_sort_key(previous):
                            await self.move_to_graveyard(
                                previous.id,
                                reason=GraveyardReason.SUPERSEDED,
                                replaced_by_id=memory.id,
                                restore_hint=(
                                    "Restore the graveyard entry if the newer observation "
                                    "is later determined to be incorrect."
                                ),
                            )
                            superseded_ids.append(previous.id)
                            # A replacement can be known through more than one
                            # key in a chain. Redirect every alias so no later
                            # update tries to bury the old record again.
                            for key, current in list(current_by_key.items()):
                                if current.id == previous.id:
                                    current_by_key[key] = memory
                            current_by_key[supersedes_key] = memory
                        else:
                            # Imports can arrive out of order. Keep the latest
                            # observed value active and bury the late, older
                            # replacement instead of letting arrival order
                            # rewrite the current identity.
                            await self.move_to_graveyard(
                                memory.id,
                                reason=GraveyardReason.SUPERSEDED,
                                replaced_by_id=previous.id,
                                restore_hint=(
                                    "This observation arrived late and is older than the "
                                    "active value; restore it only if its timestamp is corrected."
                                ),
                            )
                            superseded_ids.append(memory.id)
                            buried = True
                    else:
                        current_by_key[supersedes_key] = memory
                if memory.memory_key and not buried:
                    current = current_by_key.get(memory.memory_key)
                    if current is None or self._observation_sort_key(
                        memory
                    ) >= self._observation_sort_key(current):
                        current_by_key[memory.memory_key] = memory
        return tuple(superseded_ids)

    @staticmethod
    def _observation_sort_key(memory: Memory) -> tuple[datetime, datetime, str]:
        """Order observations by domain time, then creation time, then id."""

        observed_at = memory.observed_at or memory.created_at
        created_at = memory.created_at
        if observed_at is None:
            observed_at = datetime.min.replace(tzinfo=UTC)
        if created_at is None:
            created_at = datetime.min.replace(tzinfo=UTC)
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=UTC)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        return observed_at, created_at, memory.id

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
            memory_key=request.memory_key,
            supersedes_memory_key=request.supersedes_memory_key,
            provenance_key=request.provenance_key,
            observed_at=request.observed_at,
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
                key=self._observation_sort_key,
                reverse=True,
            )
            return ordered[0], ordered[1]

        if request.resolution is ContradictionResolution.HIGHER_CONFIDENCE:
            ordered = sorted(
                [memory_a, memory_b],
                key=lambda memory: (
                    memory.confidence,
                    *self._observation_sort_key(memory),
                ),
                reverse=True,
            )
            return ordered[0], ordered[1]

        if request.resolution is ContradictionResolution.MANUAL:
            if request.winner_memory_id not in {memory_a.id, memory_b.id}:
                raise ValueError(
                    "Manual contradiction resolution requires a valid winner_memory_id"
                )
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
