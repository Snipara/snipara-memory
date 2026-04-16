"""Embeddings provider contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingsProvider(Protocol):
    """Produces embeddings for memory storage and recall."""

    async def embed_text(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...
