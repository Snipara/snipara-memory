from __future__ import annotations

from pathlib import Path

from datetime import UTC, datetime

from snipara_memory import JsonFileMemoryStore, MemoryService, RecallQuery, StoreMemoryRequest


async def test_json_file_store_persists_across_reopen(tmp_path: Path) -> None:
    store_path = tmp_path / "memory-store.json"
    service = MemoryService(store=JsonFileMemoryStore(store_path))

    await service.store_memory(
        StoreMemoryRequest(
            namespace_id="demo",
            title="JWT convention",
            content="JWT auth uses RS256 token pairs and refresh tokens.",
            memory_key="auth.jwt",
            provenance_key="docs/auth.md",
            observed_at=datetime(2026, 8, 20, tzinfo=UTC),
        )
    )

    reopened = MemoryService(store=JsonFileMemoryStore(store_path))
    matches = await reopened.semantic_recall(
        RecallQuery(namespace_id="demo", query="How do we handle JWT auth?")
    )

    assert len(matches) == 1
    assert matches[0].memory.title == "JWT convention"
    assert matches[0].memory.memory_key == "auth.jwt"
    assert matches[0].memory.provenance_key == "docs/auth.md"
    assert matches[0].memory.observed_at == datetime(2026, 8, 20, tzinfo=UTC)
