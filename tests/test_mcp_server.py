from __future__ import annotations

from snipara_memory import InMemoryMemoryStore, MemoryService
from snipara_memory.mcp_server import create_server


def test_mcp_server_builds_with_the_declared_sdk() -> None:
    # mcp 2 removed the Server decorators: an unbounded install could not start.
    server = create_server(MemoryService(store=InMemoryMemoryStore()))

    assert server.name == "snipara-memory"
