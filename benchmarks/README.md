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
  --max-tokens 4096 \
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
The v3 prompt scans explicit user statements before assistant advice, limits
each response to eight high-value facts, and records evidence kind and temporal
anchors. The cache is flushed after every session, so an interrupted run can
resume without losing the completed tail.
Transport failures and malformed structured responses are retried with a
bounded backoff; when the model truncates the JSON tail, complete fact objects
before the truncation are retained and marked in metadata. For
reasoning-capable local models such as gpt-oss, `--reasoning-effort low` keeps
extraction latency bounded while preserving structured output.

An individual session that still returns invalid output after all retries is
recorded in the report as a failed session and skipped so the bounded run can
continue. Its failure is cached for the same extractor version, so a replay
does not repeatedly spend tokens on the same known failure.

The first pass is compute-bound and can take a long time on a local model.
Replay the same command to use the cache, and do not interpret this ingestion
step as the final LongMemEval score. The separate QA layer below performs
retrieval, reader generation, and official LLM judging.

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

### LongMemEval QA: retrieve, reader, judge

The QA command reuses the extraction cache, creates a fresh in-memory namespace
for each question, retrieves active extracted facts, asks a reader to answer
from those facts, and sends the answer to the official upstream yes/no judge
prompt. Retrieval reranks title/content/fact-key signals and preserves
evidence from distinct sessions for multi-session, temporal, and update
questions. Reader and judge outputs are cached independently so an interrupted
run can resume without repeating completed calls.

```bash
snipara-memory longmemeval-qa \
  /path/to/longmemeval_s_cleaned.json \
  --cache .cache/longmemeval-extractions.json \
  --qa-cache .cache/longmemeval-qa.json \
  --model qwen/qwen3-30b-a3b-2507 \
  --limit 50 \
  --retrieval-k 8 \
  --hypotheses .cache/longmemeval-hypotheses.jsonl \
  --json
```

Before paying for a larger run, use a category-balanced sample. This selects
the first N questions from each LongMemEval category and ignores `--limit`:

```bash
snipara-memory longmemeval-qa \
  /path/to/longmemeval_s_cleaned.json \
  --cache .cache/longmemeval-extractions.json \
  --qa-cache .cache/longmemeval-qa-stratified.json \
  --model qwen/qwen3-30b-a3b-2507 \
  --stratified-per-category 10 \
  --hypotheses .cache/longmemeval-stratified-hypotheses.jsonl \
  --json
```

The JSON report contains overall accuracy and coverage, cache hit/miss counts,
failed stages, reader outputs, raw judge outputs, answer-session recall@k, and
accuracy/retrieval recall by question category. `--hypotheses` writes the
upstream evaluator format with one `question_id`/`hypothesis` object per line.
The answer-session recall metric uses the benchmark's `answer_session_ids` only
for diagnosis; those labels are never sent to the reader.

The judge prompt is kept in code as a versioned adapter of
`src/evaluation/evaluate_qa.py` from the upstream LongMemEval repository. A
LongMemEval score still measures conversational memory, not project-memory
continuity for coding agents; publish it beside the proprietary continuity
suite rather than as a replacement for it.
