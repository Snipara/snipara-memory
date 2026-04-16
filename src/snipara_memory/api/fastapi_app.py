"""Minimal FastAPI entrypoint for the standalone memory engine."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..domain import (
    MemoryScope,
    MemoryService,
    MemoryStatus,
    MemoryTier,
    MemoryType,
    RecallQuery,
    StoreMemoryRequest,
)


class StoreMemoryBody(BaseModel):
    content: str = Field(..., min_length=1)
    title: str | None = None
    memory_type: MemoryType = MemoryType.FACT
    scope: MemoryScope = MemoryScope.PROJECT
    category: str | None = None
    source: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    relevance_boost: float = Field(default=1.0, ge=0.0)
    tier: MemoryTier | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    related_memory_ids: list[str] = Field(default_factory=list)
    document_refs: list[str] = Field(default_factory=list)
    invalid_document_refs: list[str] = Field(default_factory=list)
    expires_at: datetime | None = None
    journal_date: date | None = None


class RecallBody(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    include_archived: bool = False
    types: list[MemoryType] = Field(default_factory=list)
    tiers: list[MemoryTier] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


def create_app(service: MemoryService) -> FastAPI:
    """Create a standalone API app around the memory service."""
    app = FastAPI(title="snipara-memory", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/namespaces/{namespace_id}/memories")
    async def store_memory(namespace_id: str, body: StoreMemoryBody) -> dict[str, Any]:
        memory = await service.store_memory(
            StoreMemoryRequest(namespace_id=namespace_id, **body.model_dump())
        )
        return asdict(memory)

    @app.post("/v1/namespaces/{namespace_id}/memories/recall")
    async def recall(namespace_id: str, body: RecallBody) -> list[dict[str, Any]]:
        matches = await service.semantic_recall(
            RecallQuery(namespace_id=namespace_id, **body.model_dump())
        )
        return [asdict(match) for match in matches]

    @app.get("/v1/namespaces/{namespace_id}/session-memories")
    async def session_memories(
        namespace_id: str,
        critical_limit: int = 12,
        daily_limit: int = 20,
        archive_limit: int = 20,
    ) -> dict[str, Any]:
        bundle = await service.get_session_memories(
            namespace_id,
            critical_limit=critical_limit,
            daily_limit=daily_limit,
            archive_limit=archive_limit,
        )
        return asdict(bundle)

    @app.exception_handler(ValueError)
    async def handle_value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app
