# Roadmap

This roadmap is intentionally focused on the standalone memory engine.

## Current Baseline

Already available in `0.1.x`:

- memory domain model
- in-memory store
- semantic recall
- session bundle loading
- contradiction detection and resolution
- graveyard lifecycle
- minimal FastAPI API

## Next

1. Prisma-backed persistence adapter
2. reproducible benchmark harness
3. pluggable embeddings providers
4. richer filtering and namespace stats
5. Redis-backed session cache
6. MCP wrapper package

## After That

- transcript import helpers
- migration tooling between stores
- namespace export and backup flows
- richer policy hooks for compaction and archiving

## Not In Scope

These stay in the hosted Snipara product, not this repository:

- team billing
- SaaS auth and multi-tenant admin
- hosted MCP transport
- managed review queues
- commercial analytics and operations
