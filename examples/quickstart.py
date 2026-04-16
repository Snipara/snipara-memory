from __future__ import annotations

import asyncio

from snipara_memory import InMemoryMemoryStore, MemoryService, RecallQuery, StoreMemoryRequest


async def main() -> None:
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

    for match in matches:
        print(f"{match.score:.3f} | {match.memory.title} | {match.memory.content}")


if __name__ == "__main__":
    asyncio.run(main())
