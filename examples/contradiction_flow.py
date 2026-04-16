from __future__ import annotations

import asyncio

from snipara_memory import (
    ContradictionResolution,
    InMemoryMemoryStore,
    MemoryService,
    ResolveContradictionRequest,
    StoreMemoryRequest,
)


async def main() -> None:
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
    contradiction = contradictions[0]

    resolved = await service.resolve_contradiction(
        ResolveContradictionRequest(
            contradiction_id=contradiction.id,
            resolution=ContradictionResolution.HIGHER_CONFIDENCE,
            resolved_by="example",
        )
    )

    print("winner", resolved.winner_memory_id == newer.id)
    print("loser", resolved.loser_memory_id == older.id)


if __name__ == "__main__":
    asyncio.run(main())
