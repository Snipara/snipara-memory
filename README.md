# snipara-memory

`snipara-memory` is an open source memory schema and local engine for
AI-assisted projects.

**Memory belongs to the project, not the model.**

Use it to model, store, recall, compact, archive, and review durable project
memory without depending on Snipara Cloud.

## Quickstart

```bash
# 1. Install
pip install snipara-memory

# 2. Use the local API
snipara-memory serve --port 8000

# 3. In another terminal, store and recall
curl -X POST http://127.0.0.1:8000/v1/namespaces/demo/memories \
  -H "content-type: application/json" \
  -d '{"title": "Auth convention", "content": "JWT auth uses RS256 token pairs."}'

curl -X POST http://127.0.0.1:8000/v1/namespaces/demo/memories/recall \
  -H "content-type: application/json" \
  -d '{"query": "How do we handle auth?"}'
```

Or use the Python API:

```python
import asyncio
from snipara_memory import InMemoryMemoryStore, MemoryService, RecallQuery, StoreMemoryRequest

async def main():
    store = InMemoryMemoryStore()
    service = MemoryService(store=store)
    
    await service.store_memory(StoreMemoryRequest(
        namespace_id="demo",
        content="JWT auth uses RS256 token pairs and refresh tokens.",
        title="Auth convention",
    ))
    
    matches = await service.semantic_recall(
        RecallQuery(namespace_id="demo", query="How do we handle JWT auth?")
    )
    for match in matches:
        print(f"{match.score:.2f}: {match.memory.title}")

asyncio.run(main())
```

Full docs below. Local continuity works out-of-the-box; import commands and MCP are optional.

## What It Is

`snipara-memory` provides project-scoped memory primitives:

- memory object types
- lifecycle states
- source provenance
- authority metadata
- semantic recall requests
- contradiction records
- session warm-up bundles
- local API and MCP wrappers

It is not a generic vector database. It is the shared memory language for
agents that need to remember what should keep mattering.

## The Problem

Most agent memory systems are either transcript stores or embedding caches.
They can retrieve old text, but they rarely answer the deeper workflow question:

What should a future agent trust, reuse, or revisit?

Durable project memory needs structure:

- decisions need authority and source context
- preferences need scope
- learnings need confidence
- stale memories need retirement
- conflicting memories need review
- session startup needs compact bundles

## The Solution

`snipara-memory` gives those concepts a small, inspectable implementation.

```text
Agent Session
     |
     v
Memory Extraction
     |
     v
Project-Scoped Memory Objects
     |
     +--> Recall
     +--> Session Bundle
     +--> Compaction
     +--> Contradiction Review
     +--> Archive / Graveyard
```

The package can run locally in tests, CLIs, prototypes, and MCP-compatible
developer tools. Hosted Snipara builds on the same domain concepts with managed
retrieval, review workflows, ranking, team controls, and production operations.

## Architecture

```text
Claude Code       Cursor          Codex          OpenAI Agents
    |               |              |                  |
    +---------------+--------------+------------------+
                    |
          Project Memory Interface
                    |
             snipara-memory
                    |
       Local Store / API / MCP Wrapper
                    |
          Durable Project Context
```

## Why This Is Different

Many tools stop at "store text, run semantic search".

`snipara-memory` focuses on the memory lifecycle:

- tiered retrieval: `CRITICAL`, `DAILY`, `ARCHIVE`
- lifecycle states: `ACTIVE`, `ARCHIVED`, `GRAVEYARD`
- scoped memory ownership
- contradiction detection and resolution
- graveyard restore instead of destructive deletes
- session bundles for agent warm-up
- importers for transcripts and project docs

## Transcript Store vs Durable Memory

| Need | Transcript-first memory | `snipara-memory` |
| --- | --- | --- |
| Keep the original conversation | Strong | Not the main goal |
| Preserve durable decisions | Usually ad hoc | First-class |
| Scope memory to projects | Often weak | Built-in |
| Handle contradictions | Rare | Built-in |
| Archive without hard delete | Rare | Built-in graveyard |
| Warm up a new session | Manual | Session bundles |
| Model memory as typed objects | Limited | Built-in |

If your main problem is "search my old chats", a transcript store may be
enough. If your main problem is "my agent should keep stable project memory",
this package is the right layer.

## Install

```bash
pip install snipara-memory
```

For local development:

```bash
pip install -e ".[dev]"
```

Main CLI:

```bash
snipara-memory version
```

Local store path by default:

```text
~/.snipara-memory/store.json
```

## Python Quickstart

```python
import asyncio

from snipara_memory import InMemoryMemoryStore, MemoryService, RecallQuery, StoreMemoryRequest


async def main() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store=store)

    await service.store_memory(
        StoreMemoryRequest(
            namespace_id="demo",
            content="JWT auth uses RS256 token pairs and refresh tokens.",
            title="Auth convention",
        )
    )

    matches = await service.semantic_recall(
        RecallQuery(namespace_id="demo", query="How do we handle JWT auth?")
    )

    for match in matches:
        print(match.score, match.memory.title, match.memory.content)


asyncio.run(main())
```

Runnable example:

```bash
python examples/quickstart.py
```

Import a transcript:

```bash
snipara-memory import-transcript examples/transcript.txt --namespace demo
```

Import project documents:

```bash
snipara-memory import-project docs --namespace demo
```

## Local API

Start the FastAPI server backed by the local JSON store:

```bash
snipara-memory serve --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Store a memory:

```bash
curl -X POST http://127.0.0.1:8000/v1/namespaces/demo/memories \
  -H "content-type: application/json" \
  -d '{
    "title": "Auth convention",
    "content": "JWT auth uses RS256 token pairs and refresh tokens."
  }'
```

Recall memory:

```bash
curl -X POST http://127.0.0.1:8000/v1/namespaces/demo/memories/recall \
  -H "content-type: application/json" \
  -d '{
    "query": "How do we handle JWT auth?"
  }'
```

## Local MCP Server

Run the stdio MCP wrapper:

```bash
snipara-memory mcp
```

With an explicit store file:

```bash
snipara-memory mcp --store-path ./.snipara-memory.json
```

Current MCP tools:

- `memory_store`
- `memory_recall`
- `memory_session_bundle`
- `memory_list`
- `memory_detect_contradictions`
- `memory_resolve_contradiction`
- `memory_import_transcript`
- `memory_import_project`

See [docs/mcp.md](docs/mcp.md).

## What Is Included

Version `0.1.x` includes:

- standalone domain models
- memory service
- in-memory adapter
- JSON file store
- FastAPI app
- MCP stdio wrapper
- transcript and project-doc importers
- benchmark harness
- Prisma schema draft
- runnable examples

## What Is Not Included

This repository does not try to clone Snipara Cloud.

Not included:

- hosted MCP transport
- SaaS auth and billing
- team dashboard
- review queues
- managed retrieval ranking
- enterprise analytics
- hosted automation policies

Those remain part of Snipara's commercial hosted product.

## Open Core Boundary

Open source:

- memory schemas
- lifecycle primitives
- local storage interfaces
- import formats
- local API and MCP wrappers
- tests and examples

Commercial Snipara:

- hosted orchestration
- managed context ranking
- review and governance workflows
- team and tenant controls
- production analytics
- operational reliability

The language is open. The managed cognition layer is Snipara.

## Relationship To Other Repos

| Repo | Role |
| --- | --- |
| `Snipara/snipara-server` | Hosted and self-hosted server surface |
| `alopez3006/snipara-mcp` | Lightweight stdio MCP connector |
| `Snipara/snipara-memory` | This open memory schema and local engine |

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

Useful docs:

- [Getting started](docs/getting-started.md)
- [MCP wrapper](docs/mcp.md)
- [Importers](docs/importers.md)
- [Publishing](docs/publishing.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
