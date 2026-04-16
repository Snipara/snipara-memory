# Roadmap

This roadmap is intentionally focused on the standalone memory engine.

## Current Baseline

Already available in `0.1.x`:

- memory domain model
- in-memory store
- JSON file store
- semantic recall
- session bundle loading
- contradiction detection and resolution
- graveyard lifecycle
- minimal FastAPI API
- MCP wrapper
- transcript and project-doc import CLI
- reproducible benchmark harness

## Next

1. Prisma-backed persistence adapter
2. pluggable embeddings providers
3. richer filtering and namespace stats
4. Redis-backed session cache
5. larger benchmark suites
6. export, backup, and migration tooling

## After That

- transcript import helpers
- migration tooling between stores
- namespace export and backup flows
- richer policy hooks for compaction and archiving
- clearer integration contract for external automation runtimes

## Not In Scope

These stay in the hosted Snipara product, not this repository:

- team billing
- SaaS auth and multi-tenant admin
- hosted MCP transport
- managed review queues
- commercial analytics and operations
- client hook installers and session-lifecycle runtimes
