# Contributing

`snipara-memory` is intentionally small. Contributions should make the memory
engine clearer, more reliable, or more useful for coding-agent workflows.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Useful Commands

Run tests:

```bash
python -m pytest
```

Run the local API:

```bash
python -m snipara_memory --host 127.0.0.1 --port 8000
```

Run the examples:

```bash
python examples/quickstart.py
python examples/contradiction_flow.py
```

## Contribution Areas

High-value contributions right now:

- production persistence adapters
- embeddings provider integrations
- benchmark harness and reproducible eval fixtures
- recall filtering and ranking improvements
- MCP wrapper and tooling integrations
- docs that make the lifecycle model easier to understand

## Scope Discipline

This repository is the standalone memory engine, not the whole Snipara product.

Please avoid adding:

- billing or plan logic
- hosted SaaS concerns
- tenant admin UI
- unrelated agent framework abstractions

## Quality Bar

The project should stay:

- local-first
- inspectable
- explicit about lifecycle transitions
- honest about current capabilities
- useful without a hosted dependency
