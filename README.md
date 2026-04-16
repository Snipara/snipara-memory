# snipara-memory

`snipara-memory` is an open source memory engine for coding agents.

Transcript memory remembers what was said. `snipara-memory` remembers what
should keep mattering.

It is the extraction of Snipara's reusable memory domain into a small open
source package that can run without Snipara Cloud, billing, or multi-tenant
SaaS concerns.

The goal is simple:

- store durable memories
- recall them semantically
- load session bundles by tier
- compact or archive low-value memory
- detect and resolve contradictions

## TL;DR

If your agent forgets decisions between sessions, `snipara-memory` gives you a
small local memory layer you can embed into your own tooling.

It is built for cases like:

- coding conventions that should survive restarts
- durable project decisions
- reusable learnings from previous sessions
- session warm-up bundles for agents
- memory stores that need auditability instead of silent overwrites

## The Open Source Wedge

`snipara-memory` is not trying to win by being a generic "AI memory" bucket.

The wedge is narrower and more useful:

- durable memory for coding agents
- explicit memory lifecycle instead of raw transcript dumps
- contradiction handling instead of silent duplication
- auditable graveyard state instead of destructive deletes
- session warm-up bundles instead of ad hoc prompt stuffing

If the open source package is not clearly better for this job, it will not help
Snipara. That is the quality bar for this repository.

## Why This Is Different

Many memory tools stop at "store text, run semantic search".

`snipara-memory` goes further on the memory lifecycle itself:

- tiered retrieval: `CRITICAL`, `DAILY`, `ARCHIVE`
- explicit lifecycle states: `ACTIVE`, `ARCHIVED`, `GRAVEYARD`
- contradiction tracking and resolution
- graveyard restore flow instead of destructive deletes
- session bundle loading for agent warm-up

That makes it closer to a durable memory engine than a simple vector wrapper.

## Transcript Store vs Durable Memory Engine

| Need | Transcript-first memory | `snipara-memory` |
| --- | --- | --- |
| Keep the original conversation | Strong | Not the main goal |
| Preserve durable decisions and conventions | Usually ad hoc | First-class |
| Handle conflicting memories over time | Rare | Built-in |
| Archive without hard deletion | Rare | Built-in graveyard |
| Warm up a new coding session | Manual | Session bundle loading |
| Model project memory as typed objects | Limited | Built-in |

If your main problem is "search my old chats", a transcript store is enough.
If your main problem is "my coding agent should keep stable project memory",
this repository is a better fit.

## Why This Exists

Snipara the SaaS product has grown into a larger surface:

- hosted MCP
- reviewed project memory
- automation policies
- team and workspace controls
- analytics and managed infrastructure

That full product is useful, but too heavy if all you want is a local memory
engine for agents.

`snipara-memory` is the smaller open source layer.

## Who This Is For

Use `snipara-memory` if you are building:

- coding agents
- local-first agent tools
- MCP-compatible developer tooling
- persistent session memory
- project memory layers for your own apps

Do not use it if you are looking for:

- a full hosted MCP platform
- a SaaS dashboard for team memory review
- billing, auth, and tenant management

## What Is Included

Version `0.1.x` includes:

- a standalone domain model for memories, graveyard entries, and contradictions
- a memory service for storage, recall, session loading, compaction, and
  contradiction resolution
- an in-memory adapter for local runs and tests
- a JSON file store for local persistent usage
- a minimal FastAPI app
- a local MCP wrapper for MCP-compatible clients
- transcript and project-doc import commands
- a reproducible benchmark harness
- a Prisma schema draft for a production persistence adapter
- runnable examples for store, recall, and contradiction resolution

## What Is Not Included

This repository does **not** try to clone Snipara Cloud.

Not included:

- billing and plan gating
- hosted MCP transport
- SaaS auth and project/team UI
- approval workflows and review queue
- managed automation policies
- enterprise admin and analytics

Those remain part of Snipara's commercial hosted product.

## Install

Until the first PyPI release is published, install from GitHub:

```bash
pip install git+https://github.com/Snipara/snipara-memory.git
```

Or clone locally:

```bash
pip install -e .
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

Persistent local import:

```bash
snipara-memory import-transcript examples/transcript.txt --namespace demo
```

## Run The Local API

The package ships with a minimal API server backed by the local JSON store by
default.

```bash
snipara-memory serve --host 127.0.0.1 --port 8000
```

Example requests:

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/v1/namespaces/demo/memories \
  -H "content-type: application/json" \
  -d '{
    "title": "Auth convention",
    "content": "JWT auth uses RS256 token pairs and refresh tokens."
  }'
```

```bash
curl -X POST http://127.0.0.1:8000/v1/namespaces/demo/memories/recall \
  -H "content-type: application/json" \
  -d '{
    "query": "How do we handle JWT auth?"
  }'
```

Contradiction example:

```bash
python examples/contradiction_flow.py
```

## Run The Local MCP Server

The package also ships with a local MCP stdio wrapper.

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

## Import Durable Memory

Transcript import:

```bash
snipara-memory import-transcript examples/transcript.txt --namespace demo
```

Project-doc import:

```bash
snipara-memory import-project docs/ --namespace demo
```

These importers are intentionally conservative. They try to extract durable
decisions, preferences, learnings, and todos instead of storing every line.

See [docs/importers.md](docs/importers.md).

## Benchmark Harness

Run the reproducible recall harness:

```bash
snipara-memory benchmark benchmarks/datasets/basic_recall.jsonl
```

This is a regression harness, not a headline benchmark claim.

See [benchmarks/README.md](benchmarks/README.md).

## Why Not Just Keep Raw Transcripts?

Raw conversation storage is useful, but it is not enough on its own for durable
agent memory.

`snipara-memory` adds structure on top of stored content:

- typed memories (`FACT`, `DECISION`, `LEARNING`, `PREFERENCE`, `TODO`, `CONTEXT`)
- explicit retrieval tiers
- lifecycle transitions
- contradiction handling
- graveyard snapshots for auditability

If you only need searchable transcript history, a transcript store may be
enough. If you need reusable project memory with lifecycle rules, this package
is the better fit.

## Core Concepts

- `Namespace`: the container for a memory corpus
- `Memory`: a durable fact, decision, learning, preference, todo, or context item
- `Tier`: `CRITICAL`, `DAILY`, `ARCHIVE`
- `Status`: `ACTIVE`, `ARCHIVED`, `GRAVEYARD`
- `Graveyard`: tombstoned or superseded memory snapshots kept for auditability
- `Contradiction`: tracked overlap/conflict between memories before or after resolution

## Public API

Main exports:

```python
from snipara_memory import (
    InMemoryMemoryStore,
    MemoryService,
    RecallQuery,
    StoreMemoryRequest,
    ResolveContradictionRequest,
)
```

Key operations:

- `store_memory`
- `store_memories_bulk`
- `semantic_recall`
- `list_memories`
- `get_session_memories`
- `compact_memories`
- `detect_contradictions`
- `resolve_contradiction`
- `move_to_graveyard`
- `restore_from_graveyard`

## Project Status

`snipara-memory` is usable today for local experiments and OSS integration work.

Current state:

- domain service: implemented
- in-memory store: implemented
- JSON file store: implemented
- HTTP API: implemented
- MCP wrapper: implemented
- transcript/project import CLI: implemented
- reproducible benchmark harness: implemented
- Prisma/Redis/embeddings production adapters: not published yet

We are intentionally not making benchmark claims yet. The current public value
is clarity, inspectability, and a reusable memory lifecycle for coding agents.
The benchmark harness is now reproducible from this repository, but it is still
positioned as a regression harness rather than a competitive research claim.

## Roadmap

Near-term roadmap:

1. Prisma-backed persistence adapter
2. pluggable embeddings provider packages
3. Redis-backed session bundle cache
4. richer recall filtering and namespace stats
5. larger public benchmark suites
6. richer CLI inspection and export flows

More detail: [ROADMAP.md](ROADMAP.md)

## Contributing

Contributions are welcome.

Start here:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [ROADMAP.md](ROADMAP.md)
- [docs/INDEX.md](docs/INDEX.md)

## Relation To Snipara

Snipara Cloud builds on top of this kind of memory domain and adds:

- hosted MCP surface
- reviewable memory queue
- transcript import
- workspace profiles
- automation policies
- team controls and managed operations

Project site: [snipara.com](https://snipara.com)

## License

Apache 2.0. See [LICENSE](LICENSE).
