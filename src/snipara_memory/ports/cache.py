"""Cache contract used for session bundles and recall accelerators."""

from __future__ import annotations

from typing import Any, Protocol


class CacheStore(Protocol):
    """Minimal cache interface for the standalone memory engine."""

    async def get(self, key: str) -> Any | None: ...

    async def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def delete_prefix(self, prefix: str) -> None: ...
