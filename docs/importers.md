# Importers

`snipara-memory` is not a raw transcript database.

The importers try to extract **durable memory candidates** from local inputs:

- transcript files
- project documentation files

## Transcript Import

Supported input shapes:

- plain text lines like `user: ...`
- JSON arrays with `role` / `content`
- JSONL with one message per line

Example:

```bash
snipara-memory import-transcript examples/transcript.txt --namespace demo
```

What gets imported:

- decisions
- preferences
- learnings
- todos

What is skipped by default:

- generic conversational filler
- low-signal context with no durable value

## Project Import

The project importer is intentionally conservative.

By default it scans:

- `.md`
- `.mdx`
- `.txt`
- `.rst`

Example:

```bash
snipara-memory import-project docs/ --namespace demo
```

The importer chunks by paragraph and markdown heading, then keeps the pieces
that look like durable decisions, preferences, learnings, or todos.

## Why The Heuristics Are Narrow

If the importer stores everything, the memory layer turns into noisy archive
storage and the recall quality degrades.

The current bias is:

- import less
- keep higher-signal candidates
- let users inspect or rerun with source files if needed
