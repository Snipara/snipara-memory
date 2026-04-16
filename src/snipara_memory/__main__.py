"""CLI entrypoint for the standalone API server."""

from __future__ import annotations

import argparse

import uvicorn

from . import InMemoryMemoryStore, MemoryService, create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local snipara-memory API")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", default=8000, type=int, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = MemoryService(store=InMemoryMemoryStore())
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
