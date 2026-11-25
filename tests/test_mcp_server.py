from __future__ import annotations

import json

import pytest
from mcp import Client

from snipara_memory import InMemoryMemoryStore, MemoryService, __version__
from snipara_memory.mcp_server import create_server

TOOLS = {
    "memory_store",
    "memory_recall",
    "memory_session_bundle",
    "memory_list",
    "memory_detect_contradictions",
    "memory_resolve_contradiction",
    "memory_import_transcript",
    "memory_import_project",
}


def _client(mode: str) -> Client:
    return Client(create_server(MemoryService(store=InMemoryMemoryStore())), mode=mode)


@pytest.mark.parametrize(
    ("mode", "version"), [("legacy", "2025-11-25"), ("auto", "2026-07-28")]
)
class TestServedTools:
    """The tools as a client reaches them, with the handshake and with MCP 2026-07-28."""

    async def test_lists_and_calls_tools(self, mode: str, version: str) -> None:
        async with _client(mode) as client:
            assert client.protocol_version == version
            assert client.server_info.name == "snipara-memory"
            assert client.server_info.version == __version__
            listed = (await client.list_tools()).tools
            stored = await client.call_tool(
                "memory_store",
                {"namespace_id": "demo", "content": "JWT keys rotate every quarter."},
            )
            recalled = await client.call_tool(
                "memory_recall",
                {"namespace_id": "demo", "query": "When do JWT keys rotate?"},
            )

        assert {tool.name for tool in listed} == TOOLS
        assert not stored.is_error
        assert not recalled.is_error
        matches = json.loads(recalled.content[0].text)
        assert [match["memory"]["content"] for match in matches] == [
            "JWT keys rotate every quarter."
        ]

    @pytest.mark.parametrize(
        ("name", "arguments", "expected"),
        [
            (
                "memory_store",
                {"namespace_id": "demo", "content": 1},
                "Input validation error: 1 is not of type 'string'",
            ),
            (
                "memory_store",
                None,
                "Input validation error: 'namespace_id' is a required property",
            ),
            (
                "memory_recall",
                {"namespace_id": "demo", "query": "q", "types": ["unknown"]},
                "'unknown' is not a valid MemoryType",
            ),
            ("unknown", {}, "Unknown tool: unknown"),
        ],
    )
    async def test_reports_refused_and_failed_calls(
        self,
        mode: str,
        version: str,
        name: str,
        arguments: dict[str, object] | None,
        expected: str,
    ) -> None:
        async with _client(mode) as client:
            result = await client.call_tool(name, arguments)

        assert result.is_error
        assert result.content[0].text == expected
