from __future__ import annotations

from pathlib import Path

from snipara_memory.importers import (
    extract_project_requests,
    extract_transcript_requests,
    load_transcript_messages,
)


def test_transcript_import_extracts_durable_candidates(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.txt"
    transcript.write_text(
        "user: We decided to use RS256 token pairs for JWT auth.\n"
        "assistant: Prefer explicit graveyard states over hard deletes.\n"
        "user: TODO next step is to add contradiction resolution tests.\n",
        encoding="utf-8",
    )

    messages = load_transcript_messages(transcript)
    requests = extract_transcript_requests(
        messages,
        namespace_id="demo",
        source=str(transcript),
    )

    assert len(requests) == 3
    assert {request.memory_type.value for request in requests} == {
        "DECISION",
        "PREFERENCE",
        "TODO",
    }


def test_project_import_extracts_markdown_decisions(tmp_path: Path) -> None:
    document = tmp_path / "memory.md"
    document.write_text(
        "# Memory Policy\n\n"
        "- Always review automatic memory writes before persistence.\n\n"
        "We prefer explicit graveyard states over destructive deletes.\n",
        encoding="utf-8",
    )

    requests = extract_project_requests([document], namespace_id="demo")

    assert len(requests) == 2
    assert requests[0].title == "Memory Policy"
