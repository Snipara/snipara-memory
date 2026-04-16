"""Command-line interface for snipara-memory."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import uvicorn

from . import __version__, create_app
from .adapters import InMemoryMemoryStore, JsonFileMemoryStore, get_default_store_path
from .benchmark import benchmark_report_as_json, render_benchmark_report, run_benchmark
from .domain import MemoryService
from .importers import import_project_documents, import_transcript
from .mcp_server import run_stdio_server


COMMANDS = {"serve", "import-transcript", "import-project", "benchmark", "mcp", "version"}


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
    import_transcript_cmd.add_argument("--namespace", required=True, help="Namespace ID")
    import_transcript_cmd.add_argument("--source", help="Override source label")
    import_transcript_cmd.add_argument("--max-items", type=int, help="Maximum imported memories")
    import_transcript_cmd.add_argument("--json", action="store_true", help="Render JSON output")

    import_project_cmd = subparsers.add_parser(
        "import-project",
        help="Import durable memory candidates from project docs",
    )
    _add_store_options(import_project_cmd)
    import_project_cmd.add_argument("path", help="Project file or directory path")
    import_project_cmd.add_argument("--namespace", required=True, help="Namespace ID")
    import_project_cmd.add_argument("--max-items", type=int, help="Maximum imported memories")
    import_project_cmd.add_argument("--json", action="store_true", help="Render JSON output")

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run the reproducible benchmark harness",
    )
    benchmark.add_argument("dataset", help="Path to benchmark dataset (json/jsonl)")
    benchmark.add_argument("--json", action="store_true", help="Render JSON output")

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
    if args.command == "mcp":
        asyncio.run(run_stdio_server(store_path=args.store_path, in_memory=args.in_memory))
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
    store = InMemoryMemoryStore() if args.in_memory else JsonFileMemoryStore(args.store_path)
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
        print(json.dumps({
            "scanned_items": result.scanned_items,
            "imported_candidates": result.imported_candidates,
            "skipped_items": result.skipped_items,
        }, indent=2))
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
        print(json.dumps({
            "scanned_items": result.scanned_items,
            "imported_candidates": result.imported_candidates,
            "skipped_items": result.skipped_items,
        }, indent=2))
        return
    print(
        f"Imported {result.imported_candidates} durable memories from "
        f"{result.scanned_items} project files."
    )


async def _run_benchmark(args: argparse.Namespace) -> None:
    report = await run_benchmark(args.dataset)
    print(benchmark_report_as_json(report) if args.json else render_benchmark_report(report))
