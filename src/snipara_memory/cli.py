"""Command-line interface for snipara-memory."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import uvicorn

from . import __version__, create_app
from .adapters import InMemoryMemoryStore, JsonFileMemoryStore, get_default_store_path
from .benchmark import (
    benchmark_report_as_json,
    longmemeval_ingestion_report_as_json,
    longmemeval_qa_report_as_json,
    render_benchmark_report,
    render_longmemeval_ingestion_report,
    render_longmemeval_qa_report,
    run_benchmark,
    run_longmemeval_ingestion,
    run_longmemeval_qa,
)
from .domain import MemoryService
from .importers import import_project_documents, import_transcript
from .longmemeval import HeuristicFactExtractor, LmStudioFactExtractor
from .mcp_server import run_stdio_server
from .qa import LmStudioLongMemEvalJudge, LmStudioLongMemEvalReader, write_longmemeval_hypotheses


COMMANDS = {
    "serve",
    "import-transcript",
    "import-project",
    "benchmark",
    "longmemeval-ingest",
    "longmemeval-qa",
    "mcp",
    "version",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="snipara-memory CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the local HTTP API")
    _add_store_options(serve)
    serve.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve.add_argument("--port", default=8000, type=int, help="Bind port")
    serve.add_argument("--reload", action="store_true", help="Enable auto-reload")

    import_transcript_cmd = subparsers.add_parser(
        "import-transcript",
        help="Import durable memories from a transcript file",
    )
    _add_store_options(import_transcript_cmd)
    import_transcript_cmd.add_argument("path", help="Transcript file path")
    import_transcript_cmd.add_argument(
        "--namespace", required=True, help="Namespace ID"
    )
    import_transcript_cmd.add_argument("--source", help="Override source label")
    import_transcript_cmd.add_argument(
        "--max-items", type=int, help="Maximum imported memories"
    )
    import_transcript_cmd.add_argument(
        "--json", action="store_true", help="Render JSON output"
    )

    import_project_cmd = subparsers.add_parser(
        "import-project",
        help="Import durable memory candidates from project docs",
    )
    _add_store_options(import_project_cmd)
    import_project_cmd.add_argument("path", help="Project file or directory path")
    import_project_cmd.add_argument("--namespace", required=True, help="Namespace ID")
    import_project_cmd.add_argument(
        "--max-items", type=int, help="Maximum imported memories"
    )
    import_project_cmd.add_argument(
        "--json", action="store_true", help="Render JSON output"
    )

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run the reproducible benchmark harness",
    )
    benchmark.add_argument("dataset", help="Path to benchmark dataset (json/jsonl)")
    benchmark.add_argument("--json", action="store_true", help="Render JSON output")

    longmemeval = subparsers.add_parser(
        "longmemeval-ingest",
        help="Ingest a LongMemEval subset as extracted memories",
    )
    longmemeval.add_argument("dataset", help="Path to LongMemEval JSON/JSONL dataset")
    longmemeval.add_argument("--cache", help="JSON extraction cache path")
    longmemeval.add_argument(
        "--limit", type=int, default=50, help="Maximum number of questions to ingest"
    )
    longmemeval.add_argument(
        "--extractor",
        choices=("heuristic", "lm-studio"),
        default="heuristic",
        help="Fact extractor implementation",
    )
    longmemeval.add_argument(
        "--model",
        default=os.getenv("LM_STUDIO_MODEL"),
        help="LM Studio model identifier (or LM_STUDIO_MODEL)",
    )
    longmemeval.add_argument(
        "--base-url",
        default=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        help="LM Studio OpenAI-compatible base URL",
    )
    longmemeval.add_argument(
        "--api-key",
        default=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        help="Local API key value, if configured",
    )
    longmemeval.add_argument(
        "--prompt-version",
        default="lmstudio-fact-extractor-v1",
        help="Cache-busting extraction prompt version",
    )
    longmemeval.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=os.getenv("LM_STUDIO_REASONING_EFFORT"),
        help="Reasoning effort for reasoning-capable local models",
    )
    longmemeval.add_argument("--temperature", type=float, default=0.0)
    longmemeval.add_argument("--max-tokens", type=int, default=2048)
    longmemeval.add_argument("--timeout", type=float, default=120.0)
    longmemeval.add_argument("--retries", type=int, default=2)
    longmemeval.add_argument("--json", action="store_true", help="Render JSON output")

    longmemeval_qa = subparsers.add_parser(
        "longmemeval-qa",
        help="Run LongMemEval retrieval, reader generation, and LLM judge",
    )
    longmemeval_qa.add_argument("dataset", help="Path to LongMemEval JSON/JSONL dataset")
    longmemeval_qa.add_argument(
        "--cache",
        help="Extraction cache path (reuse the completed ingestion pass)",
    )
    longmemeval_qa.add_argument(
        "--qa-cache",
        help="Reader/judge output cache path",
    )
    longmemeval_qa.add_argument(
        "--hypotheses",
        help="Optional JSONL path compatible with the upstream evaluator",
    )
    longmemeval_qa.add_argument(
        "--limit", type=int, default=50, help="Maximum number of questions to score"
    )
    longmemeval_qa.add_argument(
        "--retrieval-k", type=int, default=8, help="Number of memories passed to the reader"
    )
    longmemeval_qa.add_argument(
        "--extractor",
        choices=("heuristic", "lm-studio"),
        default="lm-studio",
        help="Extractor used only for sessions missing from the extraction cache",
    )
    longmemeval_qa.add_argument(
        "--model",
        default=os.getenv("LM_STUDIO_MODEL"),
        help="Common LM Studio model fallback",
    )
    longmemeval_qa.add_argument(
        "--extractor-model",
        default=os.getenv("LM_STUDIO_EXTRACTOR_MODEL"),
        help="Extraction model (or LM_STUDIO_EXTRACTOR_MODEL)",
    )
    longmemeval_qa.add_argument(
        "--reader-model",
        default=os.getenv("LM_STUDIO_READER_MODEL"),
        help="Reader model (or LM_STUDIO_READER_MODEL)",
    )
    longmemeval_qa.add_argument(
        "--judge-model",
        default=os.getenv("LM_STUDIO_JUDGE_MODEL"),
        help="Judge model (or LM_STUDIO_JUDGE_MODEL)",
    )
    longmemeval_qa.add_argument(
        "--base-url",
        default=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        help="LM Studio OpenAI-compatible base URL",
    )
    longmemeval_qa.add_argument(
        "--api-key",
        default=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        help="Local API key value, if configured",
    )
    longmemeval_qa.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=os.getenv("LM_STUDIO_REASONING_EFFORT"),
        help="Reasoning effort for reader/judge and cache-miss extraction",
    )
    longmemeval_qa.add_argument("--reader-max-tokens", type=int, default=512)
    longmemeval_qa.add_argument("--judge-max-tokens", type=int, default=10)
    longmemeval_qa.add_argument("--extractor-max-tokens", type=int, default=4096)
    longmemeval_qa.add_argument("--timeout", type=float, default=180.0)
    longmemeval_qa.add_argument("--retries", type=int, default=1)
    longmemeval_qa.add_argument("--json", action="store_true", help="Render JSON output")

    mcp = subparsers.add_parser("mcp", help="Run the MCP stdio server")
    _add_store_options(mcp)
    subparsers.add_parser("version", help="Show package version")

    return parser


def main(argv: list[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args or raw_args[0].startswith("-") or raw_args[0] not in COMMANDS:
        raw_args = ["serve", *raw_args]

    args = build_parser().parse_args(raw_args)

    if args.command == "serve":
        _run_api(args)
        return
    if args.command == "import-transcript":
        asyncio.run(_run_transcript_import(args))
        return
    if args.command == "import-project":
        asyncio.run(_run_project_import(args))
        return
    if args.command == "benchmark":
        asyncio.run(_run_benchmark(args))
        return
    if args.command == "longmemeval-ingest":
        asyncio.run(_run_longmemeval_ingest(args))
        return
    if args.command == "longmemeval-qa":
        asyncio.run(_run_longmemeval_qa(args))
        return
    if args.command == "mcp":
        asyncio.run(
            run_stdio_server(store_path=args.store_path, in_memory=args.in_memory)
        )
        return
    if args.command == "version":
        print(f"snipara-memory {__version__}")
        return

    raise ValueError(f"Unhandled command: {args.command}")


def _add_store_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store-path",
        default=str(get_default_store_path()),
        help="Persistent JSON store path",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use an ephemeral in-memory store",
    )


def _build_service(args: argparse.Namespace) -> MemoryService:
    store = (
        InMemoryMemoryStore()
        if args.in_memory
        else JsonFileMemoryStore(args.store_path)
    )
    return MemoryService(store=store)


def _run_api(args: argparse.Namespace) -> None:
    app = create_app(_build_service(args))
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


async def _run_transcript_import(args: argparse.Namespace) -> None:
    result = await import_transcript(
        _build_service(args),
        args.path,
        args.namespace,
        source=args.source,
        max_items=args.max_items,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "scanned_items": result.scanned_items,
                    "imported_candidates": result.imported_candidates,
                    "skipped_items": result.skipped_items,
                },
                indent=2,
            )
        )
        return
    print(
        f"Imported {result.imported_candidates} durable memories from "
        f"{result.scanned_items} transcript entries."
    )


async def _run_project_import(args: argparse.Namespace) -> None:
    result = await import_project_documents(
        _build_service(args),
        args.path,
        args.namespace,
        max_items=args.max_items,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "scanned_items": result.scanned_items,
                    "imported_candidates": result.imported_candidates,
                    "skipped_items": result.skipped_items,
                },
                indent=2,
            )
        )
        return
    print(
        f"Imported {result.imported_candidates} durable memories from "
        f"{result.scanned_items} project files."
    )


async def _run_benchmark(args: argparse.Namespace) -> None:
    report = await run_benchmark(args.dataset)
    print(
        benchmark_report_as_json(report)
        if args.json
        else render_benchmark_report(report)
    )


async def _run_longmemeval_ingest(args: argparse.Namespace) -> None:
    report = await run_longmemeval_ingestion(
        args.dataset,
        _build_longmemeval_extractor(args),
        cache_path=args.cache,
        limit=args.limit,
    )
    print(
        longmemeval_ingestion_report_as_json(report)
        if args.json
        else render_longmemeval_ingestion_report(report)
    )


async def _run_longmemeval_qa(args: argparse.Namespace) -> None:
    common_model = args.model
    extractor_model = args.extractor_model or common_model
    reader_model = args.reader_model or common_model
    judge_model = args.judge_model or common_model
    if args.extractor == "lm-studio" and not extractor_model:
        raise SystemExit(
            "LongMemEval QA extraction requires --extractor-model, --model, "
            "or LM_STUDIO_MODEL."
        )
    if not reader_model:
        raise SystemExit(
            "LongMemEval QA reader requires --reader-model, --model, "
            "or LM_STUDIO_READER_MODEL."
        )
    if not judge_model:
        raise SystemExit(
            "LongMemEval QA judge requires --judge-model, --model, "
            "or LM_STUDIO_JUDGE_MODEL."
        )

    extractor = (
        HeuristicFactExtractor()
        if args.extractor == "heuristic"
        else LmStudioFactExtractor(
            model=extractor_model,
            base_url=args.base_url,
            api_key=args.api_key,
            prompt_version="lmstudio-fact-extractor-v1",
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.extractor_max_tokens,
            timeout_seconds=args.timeout,
            retries=args.retries,
        )
    )
    reader = LmStudioLongMemEvalReader(
        model=reader_model,
        base_url=args.base_url,
        api_key=args.api_key,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.reader_max_tokens,
        timeout_seconds=args.timeout,
        retries=args.retries,
    )
    judge = LmStudioLongMemEvalJudge(
        model=judge_model,
        base_url=args.base_url,
        api_key=args.api_key,
        reasoning_effort=None,
        max_tokens=args.judge_max_tokens,
        timeout_seconds=args.timeout,
        retries=args.retries,
    )
    report = await run_longmemeval_qa(
        args.dataset,
        extractor,
        reader,
        judge,
        ingestion_cache_path=args.cache,
        qa_cache_path=args.qa_cache,
        limit=args.limit,
        retrieval_k=args.retrieval_k,
    )
    if args.hypotheses:
        write_longmemeval_hypotheses(report, args.hypotheses)
    print(
        longmemeval_qa_report_as_json(report)
        if args.json
        else render_longmemeval_qa_report(report)
    )


def _build_longmemeval_extractor(args: argparse.Namespace):
    if args.extractor == "heuristic":
        return HeuristicFactExtractor()
    if not args.model:
        raise SystemExit("LM Studio extractor requires --model or LM_STUDIO_MODEL.")
    return LmStudioFactExtractor(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        prompt_version=args.prompt_version,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
        retries=args.retries,
    )
