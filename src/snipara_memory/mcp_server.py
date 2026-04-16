"""Minimal MCP wrapper for the standalone memory engine."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import date, datetime
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .adapters import InMemoryMemoryStore, JsonFileMemoryStore, get_default_store_path
from .domain import (
    ContradictionResolution,
    MemoryScope,
    MemoryService,
    MemoryStatus,
    MemoryTier,
    MemoryType,
    RecallQuery,
    ResolveContradictionRequest,
    StoreMemoryRequest,
)
from .importers import import_project_documents, import_transcript


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the snipara-memory MCP server")
    parser.add_argument(
        "--store-path",
        default=str(get_default_store_path()),
        help="Persistent JSON store path",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use an ephemeral in-memory store instead of the JSON file store",
    )
    return parser


def create_server(service: MemoryService) -> Server:
    server = Server("snipara-memory")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="memory_store",
                description="Store a durable memory in a namespace.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace_id": {"type": "string"},
                        "content": {"type": "string"},
                        "title": {"type": "string"},
                        "memory_type": {"type": "string", "enum": [item.value for item in MemoryType]},
                        "scope": {"type": "string", "enum": [item.value for item in MemoryScope]},
                        "category": {"type": "string"},
                        "source": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number"},
                    },
                    "required": ["namespace_id", "content"],
                },
            ),
            Tool(
                name="memory_recall",
                description="Recall memories semantically from a namespace.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace_id": {"type": "string"},
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                        "min_confidence": {"type": "number", "default": 0.0},
                        "include_archived": {"type": "boolean", "default": False},
                        "types": {"type": "array", "items": {"type": "string"}},
                        "tiers": {"type": "array", "items": {"type": "string"}},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["namespace_id", "query"],
                },
            ),
            Tool(
                name="memory_session_bundle",
                description="Load the tiered session memory bundle for a namespace.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace_id": {"type": "string"},
                        "critical_limit": {"type": "integer", "default": 12},
                        "daily_limit": {"type": "integer", "default": 20},
                        "archive_limit": {"type": "integer", "default": 20},
                    },
                    "required": ["namespace_id"],
                },
            ),
            Tool(
                name="memory_list",
                description="List memories in a namespace with optional filters.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace_id": {"type": "string"},
                        "statuses": {"type": "array", "items": {"type": "string"}},
                        "tiers": {"type": "array", "items": {"type": "string"}},
                        "types": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer"},
                    },
                    "required": ["namespace_id"],
                },
            ),
            Tool(
                name="memory_detect_contradictions",
                description="Detect semantically overlapping memories that may need resolution.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace_id": {"type": "string"},
                        "similarity_threshold": {"type": "number", "default": 0.82},
                    },
                    "required": ["namespace_id"],
                },
            ),
            Tool(
                name="memory_resolve_contradiction",
                description="Resolve a contradiction by choosing a winner or merging content.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "contradiction_id": {"type": "string"},
                        "resolution": {
                            "type": "string",
                            "enum": [item.value for item in ContradictionResolution],
                        },
                        "resolved_by": {"type": "string"},
                        "rationale": {"type": "string"},
                        "winner_memory_id": {"type": "string"},
                        "merged_content": {"type": "string"},
                    },
                    "required": ["contradiction_id", "resolution"],
                },
            ),
            Tool(
                name="memory_import_transcript",
                description="Import durable memories from a local transcript file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace_id": {"type": "string"},
                        "path": {"type": "string"},
                        "source": {"type": "string"},
                        "max_items": {"type": "integer"},
                    },
                    "required": ["namespace_id", "path"],
                },
            ),
            Tool(
                name="memory_import_project",
                description="Import durable memory candidates from local project docs.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace_id": {"type": "string"},
                        "path": {"type": "string"},
                        "max_items": {"type": "integer"},
                    },
                    "required": ["namespace_id", "path"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "memory_store":
            memory = await service.store_memory(
                StoreMemoryRequest(
                    namespace_id=arguments["namespace_id"],
                    content=arguments["content"],
                    title=arguments.get("title"),
                    memory_type=MemoryType(arguments.get("memory_type", MemoryType.FACT.value)),
                    scope=MemoryScope(arguments.get("scope", MemoryScope.PROJECT.value)),
                    category=arguments.get("category"),
                    source=arguments.get("source"),
                    tags=list(arguments.get("tags", [])),
                    confidence=float(arguments.get("confidence", 0.7)),
                )
            )
            return [_json_result(asdict(memory))]

        if name == "memory_recall":
            matches = await service.semantic_recall(
                RecallQuery(
                    namespace_id=arguments["namespace_id"],
                    query=arguments["query"],
                    limit=int(arguments.get("limit", 10)),
                    min_confidence=float(arguments.get("min_confidence", 0.0)),
                    include_archived=bool(arguments.get("include_archived", False)),
                    types=[MemoryType(value) for value in arguments.get("types", [])],
                    tiers=[MemoryTier(value) for value in arguments.get("tiers", [])],
                    tags=list(arguments.get("tags", [])),
                )
            )
            return [_json_result([asdict(match) for match in matches])]

        if name == "memory_session_bundle":
            bundle = await service.get_session_memories(
                arguments["namespace_id"],
                critical_limit=int(arguments.get("critical_limit", 12)),
                daily_limit=int(arguments.get("daily_limit", 20)),
                archive_limit=int(arguments.get("archive_limit", 20)),
            )
            return [_json_result(asdict(bundle))]

        if name == "memory_list":
            memories = await service.list_memories(
                arguments["namespace_id"],
                statuses=[MemoryStatus(value) for value in arguments.get("statuses", [])] or None,
                tiers=[MemoryTier(value) for value in arguments.get("tiers", [])] or None,
                types=[MemoryType(value) for value in arguments.get("types", [])] or None,
                limit=arguments.get("limit"),
            )
            return [_json_result([asdict(memory) for memory in memories])]

        if name == "memory_detect_contradictions":
            contradictions = await service.detect_contradictions(
                arguments["namespace_id"],
                similarity_threshold=float(arguments.get("similarity_threshold", 0.82)),
            )
            return [_json_result([asdict(item) for item in contradictions])]

        if name == "memory_resolve_contradiction":
            resolved = await service.resolve_contradiction(
                ResolveContradictionRequest(
                    contradiction_id=arguments["contradiction_id"],
                    resolution=ContradictionResolution(arguments["resolution"]),
                    resolved_by=arguments.get("resolved_by"),
                    rationale=arguments.get("rationale"),
                    winner_memory_id=arguments.get("winner_memory_id"),
                    merged_content=arguments.get("merged_content"),
                )
            )
            return [_json_result(asdict(resolved))]

        if name == "memory_import_transcript":
            result = await import_transcript(
                service,
                arguments["path"],
                arguments["namespace_id"],
                source=arguments.get("source"),
                max_items=arguments.get("max_items"),
            )
            return [_json_result(asdict(result))]

        if name == "memory_import_project":
            result = await import_project_documents(
                service,
                arguments["path"],
                arguments["namespace_id"],
                max_items=arguments.get("max_items"),
            )
            return [_json_result(asdict(result))]

        raise ValueError(f"Unknown tool: {name}")

    return server


async def run_stdio_server(*, store_path: str | None = None, in_memory: bool = False) -> None:
    store = InMemoryMemoryStore() if in_memory else JsonFileMemoryStore(store_path)
    service = MemoryService(store=store)
    server = create_server(service)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def _json_result(payload: Any) -> TextContent:
    return TextContent(
        type="text",
        text=json.dumps(_jsonable(payload), indent=2, ensure_ascii=True, sort_keys=False),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return getattr(value, "value")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run_stdio_server(store_path=args.store_path, in_memory=args.in_memory))
