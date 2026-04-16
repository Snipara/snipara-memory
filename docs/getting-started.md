# Getting Started

## Install

Until the first PyPI release is published:

```bash
pip install git+https://github.com/Snipara/snipara-memory.git
```

For development:

```bash
git clone https://github.com/Snipara/snipara-memory.git
cd snipara-memory
pip install -e ".[dev]"
```

## Local Store

By default, CLI commands persist memory in:

```text
~/.snipara-memory/store.json
```

Use `--in-memory` when you want an ephemeral run for tests or demos.

## Run The API

```bash
snipara-memory serve --host 127.0.0.1 --port 8000
```

Legacy form still works:

```bash
python -m snipara_memory --host 127.0.0.1 --port 8000
```

## Run The MCP Server

```bash
snipara-memory mcp
```

## Import Durable Memory

Transcript import:

```bash
snipara-memory import-transcript examples/transcript.txt --namespace demo
```

Project-doc import:

```bash
snipara-memory import-project docs/ --namespace demo
```

## Run Benchmarks

```bash
snipara-memory benchmark benchmarks/datasets/basic_recall.jsonl
```
