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

## LongMemEval ingestion adapter

The first step toward a real external benchmark is now available as an
ingestion adapter. It intentionally indexes extracted facts, not raw chat
turns:

- raw sessions are parsed only in memory and retained in memory metadata as
  session/turn provenance;
- an extractor implements the `FactExtractor` protocol and returns
  `ExtractedFact` objects;
- `fact_key` and `supersedes_fact_key` let the adapter exercise the memory
  graveyard instead of appending every update forever;
- extraction results are cached by question/session, session content hash, and
  extractor version, so changing the extraction prompt is an explicit cache
  invalidation event.

The package includes `HeuristicFactExtractor` only as a dependency-free smoke
test. It is not the production LongMemEval configuration; a real run should
provide an LLM-backed `FactExtractor` and keep the reader/judge stage separate.

### Local LM Studio extractor

The adapter includes an LM Studio implementation using its OpenAI-compatible
local server. Start the server from LM Studio's Developer tab (or with
`lms server start`), load the model you want to evaluate, then run a small
subset first:

```bash
export LM_STUDIO_MODEL="your-loaded-model-id"
snipara-memory longmemeval-ingest \
  /path/to/longmemeval_s_cleaned.json \
  --extractor lm-studio \
  --model "$LM_STUDIO_MODEL" \
  --reasoning-effort low \
  --limit 50 \
  --max-tokens 2048 \
  --retries 4 \
  --cache .cache/longmemeval-lmstudio.json \
  --json
```

The client uses only Python's standard library; no OpenAI SDK dependency is
required. It requests a strict JSON schema containing facts, stable fact keys,
and supersession keys. The dataset's `has_answer` evaluation labels are
deliberately omitted from the model input. `--prompt-version` and the model
identifier are part of the cache key, so changing either starts a fresh
extraction pass while preserving the previous cache for comparison.
Transport failures and malformed structured responses are retried with a
bounded backoff; when the model truncates the JSON tail, complete fact objects
before the truncation are retained and marked in metadata. For
reasoning-capable local models such as gpt-oss, `--reasoning-effort low` keeps
extraction latency bounded while preserving structured output.

The first pass is compute-bound and can take a long time on a local model.
Replay the same command to use the cache, and do not interpret this ingestion
step as the final LongMemEval score: retrieval, reader generation, and the
official LLM judge remain a separate QA layer.

Run a 30–50 question ingestion dry-run after downloading the dataset locally:

```bash
snipara-memory longmemeval-ingest \
  /path/to/longmemeval_s_cleaned.json \
  --limit 50 \
  --cache .cache/longmemeval-extractions.json \
  --json
```

No LongMemEval payload is committed to this repository. The upstream code
repository reports an MIT license, and the `longmemeval-cleaned` dataset card
also declares MIT; re-check both sources before publishing benchmark outputs or
redistributing data:

- [upstream LongMemEval repository](https://github.com/xiaowu0162/LongMemEval)
- [upstream repository license](https://github.com/xiaowu0162/LongMemEval/blob/main/LICENSE)
- [cleaned dataset card](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)

The official QA judge prompts remain upstream-owned plumbing for the next
benchmark phase. This adapter alone measures ingestion mechanics and should not
be presented as a LongMemEval answer-accuracy score.
