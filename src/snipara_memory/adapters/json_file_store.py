"""Persistent JSON-backed adapter for local standalone usage."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..domain import (
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
)
from .in_memory_store import InMemoryMemoryStore


DEFAULT_STORE_PATH = Path.home() / ".snipara-memory" / "store.json"


def get_default_store_path() -> Path:
    """Return the default persistent store path."""
    return DEFAULT_STORE_PATH


class JsonFileMemoryStore(InMemoryMemoryStore):
    """Persist the in-memory adapter state to a local JSON file."""

    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__()
        self._path = Path(path or DEFAULT_STORE_PATH).expanduser()
        self._load_state()

    async def create_memory(
        self,
        memory: Memory,
        *,
        embedding: list[float] | tuple[float, ...] | None = None,
    ) -> Memory:
        created = await super().create_memory(memory, embedding=embedding)
        self._write_state()
        return created

    async def create_memories(
        self,
        memories,
        *,
        embeddings=None,
    ) -> list[Memory]:
        created = await super().create_memories(memories, embeddings=embeddings)
        self._write_state()
        return created

    async def update_memory(self, memory: Memory) -> Memory:
        updated = await super().update_memory(memory)
        self._write_state()
        return updated

    async def delete_memory(self, memory_id: str) -> None:
        await super().delete_memory(memory_id)
        self._write_state()

    async def create_graveyard_entry(self, entry: GraveyardEntry) -> GraveyardEntry:
        stored = await super().create_graveyard_entry(entry)
        self._write_state()
        return stored

    async def save_contradiction(self, contradiction: Contradiction) -> Contradiction:
        stored = await super().save_contradiction(contradiction)
        self._write_state()
        return stored

    def _load_state(self) -> None:
        if not self._path.exists():
            return

        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._memories = {
            item["id"]: _memory_from_dict(item)
            for item in data.get("memories", [])
        }
        self._graveyard = {
            item["id"]: _graveyard_entry_from_dict(item)
            for item in data.get("graveyard", [])
        }
        self._contradictions = {
            item["id"]: _contradiction_from_dict(item)
            for item in data.get("contradictions", [])
        }
        self._embeddings = {
            memory_id: [float(value) for value in embedding]
            for memory_id, embedding in data.get("embeddings", {}).items()
        }

    def _write_state(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "memories": [_serialize(asdict(memory)) for memory in self._memories.values()],
            "graveyard": [_serialize(asdict(entry)) for entry in self._graveyard.values()],
            "contradictions": [
                _serialize(asdict(contradiction))
                for contradiction in self._contradictions.values()
            ],
            "embeddings": self._embeddings,
        }
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self._path)


def _serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, (MemoryType, MemoryScope, MemoryTier, MemoryStatus)):
        return value.value
    if isinstance(value, (GraveyardReason, ContradictionStatus, ContradictionResolution)):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(value)


def _memory_from_dict(data: dict[str, Any]) -> Memory:
    return Memory(
        id=data["id"],
        namespace_id=data["namespace_id"],
        content=data["content"],
        content_hash=data["content_hash"],
        title=data.get("title"),
        type=MemoryType(data.get("type", MemoryType.FACT.value)),
        scope=MemoryScope(data.get("scope", MemoryScope.PROJECT.value)),
        category=data.get("category"),
        source=data.get("source"),
        tags=list(data.get("tags", [])),
        metadata=dict(data.get("metadata", {})),
        confidence=float(data.get("confidence", 1.0)),
        relevance_boost=float(data.get("relevance_boost", 1.0)),
        access_count=int(data.get("access_count", 0)),
        last_accessed_at=_parse_datetime(data.get("last_accessed_at")),
        tier=MemoryTier(data.get("tier", MemoryTier.ARCHIVE.value)),
        status=MemoryStatus(data.get("status", MemoryStatus.ACTIVE.value)),
        related_memory_ids=list(data.get("related_memory_ids", [])),
        document_refs=list(data.get("document_refs", [])),
        invalid_document_refs=list(data.get("invalid_document_refs", [])),
        expires_at=_parse_datetime(data.get("expires_at")),
        journal_date=_parse_date(data.get("journal_date")),
        contradiction_count=int(data.get("contradiction_count", 0)),
        superseded_by_id=data.get("superseded_by_id"),
        restored_from_graveyard_id=data.get("restored_from_graveyard_id"),
        promoted_at=_parse_datetime(data.get("promoted_at")),
        promoted_by=data.get("promoted_by"),
        buried_at=_parse_datetime(data.get("buried_at")),
        buried_reason=(
            GraveyardReason(data["buried_reason"])
            if data.get("buried_reason") is not None
            else None
        ),
        created_at=_parse_datetime(data.get("created_at")),
        updated_at=_parse_datetime(data.get("updated_at")),
    )


def _graveyard_entry_from_dict(data: dict[str, Any]) -> GraveyardEntry:
    return GraveyardEntry(
        id=data["id"],
        namespace_id=data["namespace_id"],
        original_memory_id=data.get("original_memory_id"),
        content=data["content"],
        content_hash=data["content_hash"],
        reason=GraveyardReason(data["reason"]),
        buried_at=_parse_datetime(data["buried_at"]) or datetime.now(),
        replaced_by_id=data.get("replaced_by_id"),
        contradiction_id=data.get("contradiction_id"),
        title=data.get("title"),
        type=MemoryType(data.get("type", MemoryType.FACT.value)),
        scope=MemoryScope(data.get("scope", MemoryScope.PROJECT.value)),
        category=data.get("category"),
        source=data.get("source"),
        tags=list(data.get("tags", [])),
        metadata=dict(data.get("metadata", {})),
        confidence=float(data.get("confidence", 1.0)),
        previous_tier=(
            MemoryTier(data["previous_tier"])
            if data.get("previous_tier") is not None
            else None
        ),
        previous_status=(
            MemoryStatus(data["previous_status"])
            if data.get("previous_status") is not None
            else None
        ),
        related_memory_ids=list(data.get("related_memory_ids", [])),
        document_refs=list(data.get("document_refs", [])),
        invalid_document_refs=list(data.get("invalid_document_refs", [])),
        restore_hint=data.get("restore_hint"),
        snapshot=dict(data.get("snapshot", {})),
    )


def _contradiction_from_dict(data: dict[str, Any]) -> Contradiction:
    return Contradiction(
        id=data["id"],
        namespace_id=data["namespace_id"],
        memory_a_id=data["memory_a_id"],
        memory_b_id=data["memory_b_id"],
        similarity=float(data["similarity"]),
        status=ContradictionStatus(data["status"]),
        created_at=_parse_datetime(data["created_at"]) or datetime.now(),
        memory_a_summary=data.get("memory_a_summary"),
        memory_b_summary=data.get("memory_b_summary"),
        contradiction_kind=data.get("contradiction_kind"),
        detected_by=data.get("detected_by"),
        resolution=(
            ContradictionResolution(data["resolution"])
            if data.get("resolution") is not None
            else None
        ),
        winner_memory_id=data.get("winner_memory_id"),
        loser_memory_id=data.get("loser_memory_id"),
        rationale=data.get("rationale"),
        resolved_by=data.get("resolved_by"),
        resolved_at=_parse_datetime(data.get("resolved_at")),
        metadata=dict(data.get("metadata", {})),
    )
