# Benchmarks

`snipara-memory` does not make inflated benchmark claims.

This directory exists so the retrieval behavior can be measured
reproducibly from the repository itself.

## Current Goal

The current harness is a sanity benchmark for the standalone memory engine:

- seed a namespace with known memories
- run recall queries
- measure whether the relevant memory is returned in the top-k

Metrics reported:

- `Recall@k`
- `MRR`
- `Top1 accuracy`

## Run

```bash
snipara-memory benchmark benchmarks/datasets/basic_recall.jsonl
```

JSON output:

```bash
snipara-memory benchmark benchmarks/datasets/basic_recall.jsonl --json
```

## Dataset Format

Each case is JSONL with:

```json
{
  "id": "jwt-auth",
  "namespace_id": "demo",
  "query": "How do we handle JWT auth?",
  "setup": [
    {
      "title": "JWT convention",
      "content": "JWT auth uses RS256 token pairs and refresh tokens.",
      "memory_type": "DECISION"
    }
  ],
  "relevant_indices": [0],
  "limit": 5
}
```

`relevant_indices` point to the entries inside `setup` that should be considered
correct answers for the query.

## Important Limitation

This harness is intentionally simple. It is useful for:

- regression testing
- checking retrieval changes
- comparing local ranking behavior across versions

It is **not** yet a competitive long-context benchmark suite.

When a broader benchmark is added, it should remain reproducible from this repo
with committed fixtures or clearly documented download steps.
