from __future__ import annotations

from pathlib import Path

from snipara_memory.importers import (
    TranscriptMessage,
    chunk_transcript_messages,
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


def test_transcript_import_can_retain_bounded_source_context(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.txt"
    transcript.write_text(
        "user: Please describe the blue Plesiosaur illustration in detail.\n"
        "assistant: The Plesiosaur has a blue scaly body and long flippers.\n",
        encoding="utf-8",
    )

    requests = extract_transcript_requests(
        load_transcript_messages(transcript),
        namespace_id="demo",
        source=str(transcript),
        include_source_context=True,
    )

    source_requests = [
        request for request in requests if "source-context" in request.tags
    ]
    assert len(source_requests) == 1
    assert "blue scaly body" in source_requests[0].content
    assert source_requests[0].metadata["source_message_indices"] == [0, 1]


def test_source_context_chunking_never_exceeds_public_bound() -> None:
    chunks = chunk_transcript_messages(
        [
            TranscriptMessage(role="user", content="a" * 300),
            TranscriptMessage(role="assistant", content="b" * 300),
        ],
        max_chars=120,
        overlap_messages=1,
    )

    assert chunks
    assert all(len(chunk.content) <= 120 for chunk in chunks)


def test_source_context_chunking_preserves_facts_across_split_boundaries() -> None:
    phrase = "The Plesiosaur has a blue scaly body and long flippers."
    chunks = chunk_transcript_messages(
        [
            TranscriptMessage(
                role="assistant",
                content=("introductory material " * 15) + phrase + (" ending" * 20),
            )
        ],
        max_chars=180,
        overlap_chars=80,
    )

    assert all(len(chunk.content) <= 180 for chunk in chunks)
    assert any(phrase in chunk.content for chunk in chunks)


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
