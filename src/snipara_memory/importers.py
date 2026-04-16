"""Transcript and project import helpers for snipara-memory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .domain import MemoryScope, MemoryService, MemoryType, StoreMemoryRequest

SUPPORTED_PROJECT_EXTENSIONS = {".md", ".mdx", ".txt", ".rst"}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_ROLE_PREFIX_RE = re.compile(r"^(?P<role>[A-Za-z][\w -]{0,20}):\s*(?P<content>.+)$")

_DECISION_KEYWORDS = ("decide", "decided", "use ", "uses ", "must ", "should ", "policy", "standard", "convention", "require", "requires")
_PREFERENCE_KEYWORDS = ("prefer", "preferred", "avoid", "never ", "always ")
_LEARNING_KEYWORDS = ("learned", "found", "issue", "bug", "fix", "because", "root cause")
_TODO_KEYWORDS = ("todo", "follow up", "next step", "next:", "need to", "needs to")


@dataclass(slots=True)
class TranscriptMessage:
    role: str
    content: str


@dataclass(slots=True)
class ImportPlan:
    requests: list[StoreMemoryRequest]
    scanned_items: int
    imported_candidates: int
    skipped_items: int


async def import_transcript(
    service: MemoryService,
    path: str | Path,
    namespace_id: str,
    *,
    source: str | None = None,
    max_items: int | None = None,
) -> ImportPlan:
    messages = load_transcript_messages(path)
    requests = extract_transcript_requests(
        messages,
        namespace_id=namespace_id,
        source=source or str(path),
    )
    if max_items is not None:
        requests = requests[:max_items]
    if requests:
        await service.store_memories_bulk(requests)
    return ImportPlan(
        requests=requests,
        scanned_items=len(messages),
        imported_candidates=len(requests),
        skipped_items=max(0, len(messages) - len(requests)),
    )


async def import_project_documents(
    service: MemoryService,
    path: str | Path,
    namespace_id: str,
    *,
    extensions: set[str] | None = None,
    max_items: int | None = None,
) -> ImportPlan:
    source_path = Path(path)
    files = _collect_project_files(source_path, extensions=extensions)
    requests = extract_project_requests(files, namespace_id=namespace_id)
    if max_items is not None:
        requests = requests[:max_items]
    if requests:
        await service.store_memories_bulk(requests)
    return ImportPlan(
        requests=requests,
        scanned_items=len(files),
        imported_candidates=len(requests),
        skipped_items=max(0, len(files) - len(requests)),
    )


def load_transcript_messages(path: str | Path) -> list[TranscriptMessage]:
    transcript_path = Path(path)
    suffix = transcript_path.suffix.lower()

    if suffix == ".jsonl":
        messages = []
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            message = _message_from_payload(payload)
            if message is not None:
                messages.append(message)
        return messages

    if suffix == ".json":
        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [
                message
                for item in payload
                if (message := _message_from_payload(item)) is not None
            ]
        if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
            return [
                message
                for item in payload["messages"]
                if (message := _message_from_payload(item)) is not None
            ]

    messages: list[TranscriptMessage] = []
    for raw_line in transcript_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _ROLE_PREFIX_RE.match(line)
        if match:
            messages.append(
                TranscriptMessage(
                    role=match.group("role").lower(),
                    content=match.group("content").strip(),
                )
            )
        else:
            messages.append(TranscriptMessage(role="unknown", content=line))
    return messages


def extract_transcript_requests(
    messages: list[TranscriptMessage],
    *,
    namespace_id: str,
    source: str,
) -> list[StoreMemoryRequest]:
    seen: set[str] = set()
    requests: list[StoreMemoryRequest] = []

    for message in messages:
        for sentence in _split_sentences(message.content):
            content = _clean_chunk(sentence)
            if len(content) < 30:
                continue
            memory_type, confidence = infer_memory_type(content)
            if memory_type is MemoryType.CONTEXT:
                continue
            normalized = content.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            requests.append(
                StoreMemoryRequest(
                    namespace_id=namespace_id,
                    content=content,
                    title=_make_title(content),
                    memory_type=memory_type,
                    scope=MemoryScope.PROJECT,
                    source=source,
                    tags=["imported", "transcript", message.role],
                    metadata={"speaker": message.role, "import_source": source},
                    confidence=confidence,
                )
            )

    return requests


def extract_project_requests(
    files: list[Path],
    *,
    namespace_id: str,
) -> list[StoreMemoryRequest]:
    seen: set[str] = set()
    requests: list[StoreMemoryRequest] = []

    for file_path in files:
        if file_path.suffix.lower() in {".md", ".mdx"}:
            chunks = _extract_markdown_chunks(file_path)
        else:
            chunks = _extract_plaintext_chunks(file_path)

        for title_hint, chunk in chunks:
            content = _clean_chunk(chunk)
            if len(content) < 40:
                continue
            memory_type, confidence = infer_memory_type(content)
            normalized = content.lower()
            if normalized in seen or memory_type is MemoryType.CONTEXT:
                continue
            seen.add(normalized)
            requests.append(
                StoreMemoryRequest(
                    namespace_id=namespace_id,
                    content=content,
                    title=title_hint or _make_title(content),
                    memory_type=memory_type,
                    scope=MemoryScope.PROJECT,
                    source=str(file_path),
                    tags=["imported", "project", file_path.suffix.lower().lstrip(".")],
                    metadata={"path": str(file_path), "import_source": "project"},
                    confidence=confidence,
                )
            )

    return requests


def infer_memory_type(content: str) -> tuple[MemoryType, float]:
    lower = f" {content.lower()} "
    if any(keyword in lower for keyword in _TODO_KEYWORDS):
        return MemoryType.TODO, 0.68
    if any(keyword in lower for keyword in _PREFERENCE_KEYWORDS):
        return MemoryType.PREFERENCE, 0.8
    if any(keyword in lower for keyword in _DECISION_KEYWORDS):
        return MemoryType.DECISION, 0.84
    if any(keyword in lower for keyword in _LEARNING_KEYWORDS):
        return MemoryType.LEARNING, 0.74
    return MemoryType.CONTEXT, 0.55


def _message_from_payload(payload: object) -> TranscriptMessage | None:
    if not isinstance(payload, dict):
        return None
    role = str(payload.get("role") or payload.get("speaker") or "unknown").lower()
    content = payload.get("content") or payload.get("text") or payload.get("message")
    if not isinstance(content, str) or not content.strip():
        return None
    return TranscriptMessage(role=role, content=content.strip())


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def _make_title(content: str) -> str:
    title = content.strip().rstrip(".")
    if len(title) <= 60:
        return title
    return f"{title[:57].rstrip()}..."


def _clean_chunk(chunk: str) -> str:
    cleaned = chunk.strip()
    cleaned = cleaned.lstrip("-*0123456789. ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _collect_project_files(
    path: Path,
    *,
    extensions: set[str] | None = None,
) -> list[Path]:
    normalized_extensions = {ext.lower() for ext in (extensions or SUPPORTED_PROJECT_EXTENSIONS)}
    if path.is_file():
        return [path]
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in normalized_extensions
    )


def _extract_markdown_chunks(file_path: Path) -> list[tuple[str | None, str]]:
    current_heading: str | None = None
    paragraph: list[str] = []
    chunks: list[tuple[str | None, str]] = []

    def flush() -> None:
        if paragraph:
            chunks.append((current_heading, " ".join(paragraph).strip()))
            paragraph.clear()

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            flush()
            current_heading = line.lstrip("#").strip() or current_heading
            continue
        if not line:
            flush()
            continue
        if line.startswith(("-", "*")):
            flush()
            chunks.append((current_heading, line[1:].strip()))
            continue
        paragraph.append(line)

    flush()
    return chunks


def _extract_plaintext_chunks(file_path: Path) -> list[tuple[str | None, str]]:
    chunks = []
    for raw_chunk in file_path.read_text(encoding="utf-8").split("\n\n"):
        chunk = raw_chunk.strip()
        if chunk:
            chunks.append((file_path.name, chunk))
    return chunks
