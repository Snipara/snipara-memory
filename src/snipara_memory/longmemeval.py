"""LongMemEval ingestion primitives.

The benchmark contains timestamped chat sessions, but the memory engine should
index extracted facts rather than raw turns.  This module keeps that product
decision explicit while leaving the extraction model behind a small protocol.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .domain import (
    GraveyardReason,
    MemoryScope,
    MemoryService,
    MemoryStatus,
    MemoryType,
    StoreMemoryRequest,
)
from .importers import TranscriptMessage, extract_transcript_requests

LONGMEMEVAL_CACHE_SCHEMA = "snipara.longmemeval.extraction-cache.v1"
LONGMEMEVAL_SOURCE = "longmemeval-cleaned"
LM_STUDIO_DEFAULT_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_DEFAULT_PROMPT_VERSION = "lmstudio-fact-extractor-v1"
LM_STUDIO_SYSTEM_PROMPT = """You extract durable personal facts from one timestamped chat session.

Return only facts that are useful for answering future questions: stable user
facts, preferences, decisions, plans, or explicit updates. Do not copy the raw
transcript, do not infer facts that are not stated, and do not use any question
or answer outside this session.

Assign a stable semantic fact_key when a fact can be updated later, such as
user.home_city. If this session updates an earlier fact, set
supersedes_fact_key to the earlier key. Use null when no stable key is safe.
source_turn_indices must refer to the zero-based turn indexes in the supplied
session. Use uppercase memory_type values: FACT, DECISION, LEARNING,
PREFERENCE, TODO, or CONTEXT. Return an empty facts array when nothing durable
is present.
"""
LM_STUDIO_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facts"],
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "content",
                    "title",
                    "memory_type",
                    "confidence",
                    "fact_key",
                    "supersedes_fact_key",
                    "source_turn_indices",
                    "tags",
                ],
                "properties": {
                    "content": {"type": "string", "minLength": 1},
                    "title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "memory_type": {
                        "type": "string",
                        "enum": [
                            "FACT",
                            "DECISION",
                            "LEARNING",
                            "PREFERENCE",
                            "TODO",
                            "CONTEXT",
                        ],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "fact_key": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "supersedes_fact_key": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                    "source_turn_indices": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        }
    },
}


@dataclass(frozen=True, slots=True)
class LongMemEvalTurn:
    """One raw turn, retained only during extraction and for provenance."""

    role: str
    content: str
    has_answer: bool = False

    @classmethod
    def from_payload(cls, payload: object) -> "LongMemEvalTurn":
        if not isinstance(payload, Mapping):
            raise ValueError("LongMemEval turns must be JSON objects")
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LongMemEval turns require non-empty string content")
        role = payload.get("role", "unknown")
        return cls(
            role=str(role).lower(),
            content=content.strip(),
            has_answer=bool(payload.get("has_answer", False)),
        )


@dataclass(frozen=True, slots=True)
class LongMemEvalSession:
    """A timestamped session from a LongMemEval question."""

    session_id: str
    date: str
    turns: tuple[LongMemEvalTurn, ...]

    @classmethod
    def from_payload(
        cls,
        session_id: object,
        session_date: object,
        turns: object,
    ) -> "LongMemEvalSession":
        if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
            raise ValueError("LongMemEval sessions must contain a list of turns")
        return cls(
            session_id=str(session_id),
            date=str(session_date),
            turns=tuple(LongMemEvalTurn.from_payload(turn) for turn in turns),
        )

    @property
    def content_hash(self) -> str:
        """Return a stable hash used to invalidate extraction cache entries."""

        payload = [
            {
                "role": turn.role,
                "content": turn.content,
                "has_answer": turn.has_answer,
            }
            for turn in self.turns
        ]
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LongMemEvalQuestion:
    """The portion of a LongMemEval record needed by ingestion and QA."""

    question_id: str
    question_type: str
    question: str
    answer: Any
    question_date: str
    sessions: tuple[LongMemEvalSession, ...]
    answer_session_ids: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: object) -> "LongMemEvalQuestion":
        if not isinstance(payload, Mapping):
            raise ValueError("LongMemEval questions must be JSON objects")

        required = (
            "question_id",
            "question",
            "haystack_session_ids",
            "haystack_dates",
            "haystack_sessions",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(
                f"LongMemEval question is missing fields: {', '.join(missing)}"
            )

        session_ids = payload["haystack_session_ids"]
        dates = payload["haystack_dates"]
        session_payloads = payload["haystack_sessions"]
        if not all(
            isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            for value in (session_ids, dates, session_payloads)
        ):
            raise ValueError("LongMemEval haystack fields must be lists")
        if not len(session_ids) == len(dates) == len(session_payloads):
            raise ValueError(
                "LongMemEval haystack ids, dates, and sessions must have equal lengths"
            )

        return cls(
            question_id=str(payload["question_id"]),
            question_type=str(payload.get("question_type", "unknown")),
            question=str(payload["question"]),
            answer=payload.get("answer"),
            question_date=str(payload.get("question_date", "")),
            sessions=tuple(
                LongMemEvalSession.from_payload(session_id, session_date, session)
                for session_id, session_date, session in zip(
                    session_ids, dates, session_payloads, strict=True
                )
            ),
            answer_session_ids=tuple(
                str(value) for value in payload.get("answer_session_ids", [])
            ),
        )


@dataclass(frozen=True, slots=True)
class ExtractedFact:
    """A structured fact produced by an extraction model.

    ``fact_key`` is the stable identity assigned by the extractor.  When a
    later fact sets ``supersedes_fact_key`` to that identity, ingestion moves
    the earlier memory to the graveyard instead of leaving both facts active.
    """

    content: str
    title: str | None = None
    memory_type: MemoryType = MemoryType.FACT
    confidence: float = 0.8
    fact_key: str | None = None
    supersedes_fact_key: str | None = None
    source_turn_indices: tuple[int, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Extracted facts require non-empty content")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Extracted fact confidence must be between 0 and 1")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExtractedFact":
        raw_type = payload.get(
            "memory_type", payload.get("type", MemoryType.FACT.value)
        )
        raw_indices = payload.get(
            "source_turn_indices", payload.get("turn_indices", [])
        )
        raw_tags = payload.get("tags", [])
        return cls(
            content=str(payload["content"]),
            title=str(payload["title"]) if payload.get("title") is not None else None,
            memory_type=MemoryType(str(raw_type)),
            confidence=float(payload.get("confidence", 0.8)),
            fact_key=str(payload["fact_key"])
            if payload.get("fact_key") is not None
            else None,
            supersedes_fact_key=(
                str(payload["supersedes_fact_key"])
                if payload.get("supersedes_fact_key") is not None
                else None
            ),
            source_turn_indices=tuple(int(value) for value in raw_indices),
            tags=tuple(str(value) for value in raw_tags),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "title": self.title,
            "memory_type": self.memory_type.value,
            "confidence": self.confidence,
            "fact_key": self.fact_key,
            "supersedes_fact_key": self.supersedes_fact_key,
            "source_turn_indices": list(self.source_turn_indices),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


class FactExtractor(Protocol):
    """Extraction boundary for a local heuristic or an LLM-backed adapter."""

    version: str

    async def extract(
        self, session: LongMemEvalSession
    ) -> Sequence[ExtractedFact | Mapping[str, Any]]: ...


class _LmStudioRequestError(OSError):
    """Retryable local transport failure."""


@dataclass(slots=True)
class LmStudioFactExtractor:
    """Extract facts through LM Studio's OpenAI-compatible local API."""

    model: str
    base_url: str = LM_STUDIO_DEFAULT_BASE_URL
    api_key: str = "lm-studio"
    prompt_version: str = LM_STUDIO_DEFAULT_PROMPT_VERSION
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: float = 120.0
    retries: int = 2

    def __post_init__(self) -> None:
        self.model = self.model.strip()
        self.base_url = self.base_url.rstrip("/")
        if not self.model:
            raise ValueError("LM Studio extractor requires a model identifier")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("LM Studio base_url must start with http:// or https://")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.retries < 0:
            raise ValueError("retries cannot be negative")

    @property
    def version(self) -> str:
        """Cache identity for the model and the extraction prompt contract."""

        return f"{self.model}:{self.prompt_version}"

    async def extract(self, session: LongMemEvalSession) -> Sequence[ExtractedFact]:
        payload = self._build_payload(session)
        for attempt in range(self.retries + 1):
            try:
                response = await asyncio.to_thread(self._post_json, payload)
            except _LmStudioRequestError as error:
                if attempt >= self.retries:
                    raise RuntimeError(
                        f"LM Studio request failed after {attempt + 1} attempts: {error}"
                    ) from error
                await asyncio.sleep(min(2**attempt, 8))
                continue
            return _facts_from_lm_studio_response(response)
        raise AssertionError("LM Studio retry loop exited unexpectedly")

    def _build_payload(self, session: LongMemEvalSession) -> dict[str, Any]:
        # Deliberately omit LongMemEval's `has_answer` labels: they are ground
        # truth for evaluation and must never leak into the extraction prompt.
        session_payload = {
            "session_id": session.session_id,
            "session_date": session.date,
            "turns": [
                {"role": turn.role, "content": turn.content} for turn in session.turns
            ],
        }
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": LM_STUDIO_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        session_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "longmemeval_facts",
                    "strict": True,
                    "schema": LM_STUDIO_FACT_SCHEMA,
                },
            },
        }

    def _post_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise _LmStudioRequestError(
                f"HTTP {error.code} from {self.base_url}: {detail}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise _LmStudioRequestError(
                f"Could not reach {self.base_url}: {error}"
            ) from error
        if not isinstance(decoded, Mapping):
            raise _LmStudioRequestError("LM Studio returned a non-object response")
        return decoded


def _facts_from_lm_studio_response(
    response: Mapping[str, Any],
) -> list[ExtractedFact]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LM Studio response has no choices")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise ValueError("LM Studio response choice is not an object")
    message = first_choice.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("LM Studio response choice has no text content")

    content = message["content"].strip()
    if content.startswith("```"):
        lines = content.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        content = "\n".join(lines).strip()
    decoded = json.loads(content)
    if isinstance(decoded, list):
        raw_facts = decoded
    elif isinstance(decoded, Mapping):
        raw_facts = decoded.get("facts")
    else:
        raw_facts = None
    if not isinstance(raw_facts, list):
        raise ValueError("LM Studio response must contain a facts array")
    try:
        return [ExtractedFact.from_mapping(fact) for fact in raw_facts]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"LM Studio returned an invalid fact: {error}") from error


class HeuristicFactExtractor:
    """Dependency-free baseline extractor for adapter smoke tests.

    This is intentionally not presented as the LongMemEval-quality extractor:
    it reuses the package's keyword-based importer so the adapter can be
    validated without making LLM calls. A production run should supply an
    implementation of :class:`FactExtractor` backed by the chosen model.
    """

    version = "heuristic-fact-extractor-v1"

    async def extract(self, session: LongMemEvalSession) -> Sequence[ExtractedFact]:
        facts: list[ExtractedFact] = []
        for turn_index, turn in enumerate(session.turns):
            requests = extract_transcript_requests(
                [TranscriptMessage(role=turn.role, content=turn.content)],
                namespace_id="longmemeval",
                source=LONGMEMEVAL_SOURCE,
            )
            for request in requests:
                facts.append(
                    ExtractedFact(
                        content=request.content,
                        title=request.title,
                        memory_type=request.memory_type,
                        confidence=request.confidence,
                        source_turn_indices=(turn_index,),
                        tags=tuple(
                            tag
                            for tag in request.tags
                            if tag not in {"imported", "transcript"}
                        ),
                        metadata={"speaker": turn.role},
                    )
                )
        return facts


@dataclass(slots=True)
class ExtractionCache:
    """JSON cache keyed by question/session, content hash, and extractor version."""

    path: Path
    _entries: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != LONGMEMEVAL_CACHE_SCHEMA
        ):
            return
        entries = payload.get("entries", {})
        if isinstance(entries, Mapping):
            self._entries = {
                str(key): dict(value)
                for key, value in entries.items()
                if isinstance(value, Mapping)
            }

    def get(
        self,
        key: str,
        *,
        session_hash: str,
        extractor_version: str,
    ) -> list[ExtractedFact] | None:
        entry = self._entries.get(key)
        if (
            not entry
            or entry.get("session_hash") != session_hash
            or entry.get("extractor_version") != extractor_version
        ):
            return None
        raw_facts = entry.get("facts", [])
        if not isinstance(raw_facts, list):
            return None
        try:
            return [ExtractedFact.from_mapping(fact) for fact in raw_facts]
        except (KeyError, TypeError, ValueError):
            return None

    def put(
        self,
        key: str,
        *,
        session_hash: str,
        extractor_version: str,
        facts: Sequence[ExtractedFact],
    ) -> None:
        self._entries[key] = {
            "session_hash": session_hash,
            "extractor_version": extractor_version,
            "facts": [fact.to_dict() for fact in facts],
        }

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": LONGMEMEVAL_CACHE_SCHEMA,
            "entries": self._entries,
        }
        temporary_path = self.path.with_name(f".{self.path.name}.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)


@dataclass(frozen=True, slots=True)
class LongMemEvalIngestionResult:
    question_id: str
    namespace_id: str
    session_count: int
    cache_hits: int
    cache_misses: int
    extracted_fact_count: int
    stored_memory_ids: tuple[str, ...]
    superseded_memory_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LongMemEvalIngestionReport:
    dataset: str
    question_count: int
    session_count: int
    extracted_fact_count: int
    cache_hits: int
    cache_misses: int
    questions: tuple[LongMemEvalIngestionResult, ...]


async def ingest_longmemeval_question(
    service: MemoryService,
    question: LongMemEvalQuestion,
    extractor: FactExtractor,
    *,
    cache: ExtractionCache | None = None,
    namespace_id: str | None = None,
) -> LongMemEvalIngestionResult:
    """Extract and store one question's sessions as structured memories."""

    target_namespace = namespace_id or f"longmemeval:{question.question_id}"
    extractor_version = str(getattr(extractor, "version", extractor.__class__.__name__))
    requests: list[StoreMemoryRequest] = []
    cache_hits = 0
    cache_misses = 0

    for session in question.sessions:
        cache_key = f"{question.question_id}:{session.session_id}"
        facts = (
            cache.get(
                cache_key,
                session_hash=session.content_hash,
                extractor_version=extractor_version,
            )
            if cache
            else None
        )
        if facts is None:
            raw_facts = extractor.extract(session)
            if inspect.isawaitable(raw_facts):
                raw_facts = await raw_facts
            facts = [
                fact
                if isinstance(fact, ExtractedFact)
                else ExtractedFact.from_mapping(fact)
                for fact in raw_facts
            ]
            cache_misses += 1
            if cache:
                cache.put(
                    cache_key,
                    session_hash=session.content_hash,
                    extractor_version=extractor_version,
                    facts=facts,
                )
        else:
            cache_hits += 1
        for fact in facts:
            metadata = dict(fact.metadata)
            metadata.update(
                {
                    "benchmark": "LongMemEval",
                    "question_id": question.question_id,
                    "question_type": question.question_type,
                    "question_date": question.question_date,
                    "source_session_id": session.session_id,
                    "source_session_date": session.date,
                    "source_turn_indices": list(fact.source_turn_indices),
                    "extractor_version": extractor_version,
                    "fact_key": fact.fact_key,
                    "supersedes_fact_key": fact.supersedes_fact_key,
                    "answer_session": session.session_id in question.answer_session_ids,
                }
            )
            source_ref = f"longmemeval://{question.question_id}/{session.session_id}"
            tags = {
                "longmemeval",
                "extracted-fact",
                question.question_type,
                *fact.tags,
            }
            if session.session_id in question.answer_session_ids:
                tags.add("answer-session")
            requests.append(
                StoreMemoryRequest(
                    namespace_id=target_namespace,
                    content=fact.content,
                    title=fact.title,
                    memory_type=fact.memory_type,
                    scope=MemoryScope.USER,
                    category=question.question_type,
                    source=LONGMEMEVAL_SOURCE,
                    tags=sorted(tags),
                    metadata=metadata,
                    confidence=fact.confidence,
                    document_refs=[source_ref],
                )
            )

    existing = await service.list_memories(
        target_namespace, statuses=[MemoryStatus.ACTIVE]
    )
    current_by_fact_key = {
        str(memory.metadata["fact_key"]): memory
        for memory in existing
        if memory.metadata.get("fact_key")
    }
    created = await service.store_memories_bulk(requests)
    superseded_ids: list[str] = []

    for memory in created:
        supersedes_key = memory.metadata.get("supersedes_fact_key")
        if supersedes_key:
            previous = current_by_fact_key.get(str(supersedes_key))
            if previous is not None and previous.id != memory.id:
                await service.move_to_graveyard(
                    previous.id,
                    reason=GraveyardReason.SUPERSEDED,
                    replaced_by_id=memory.id,
                    restore_hint="Re-run the LongMemEval extractor if the newer fact is incorrect.",
                )
                superseded_ids.append(previous.id)
        fact_key = memory.metadata.get("fact_key")
        if fact_key:
            current_by_fact_key[str(fact_key)] = memory

    if cache:
        cache.flush()

    return LongMemEvalIngestionResult(
        question_id=question.question_id,
        namespace_id=target_namespace,
        session_count=len(question.sessions),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        extracted_fact_count=len(requests),
        stored_memory_ids=tuple(memory.id for memory in created),
        superseded_memory_ids=tuple(superseded_ids),
    )


async def ingest_longmemeval_dataset(
    service: MemoryService,
    dataset_path: str | Path,
    extractor: FactExtractor,
    *,
    cache_path: str | Path | None = None,
    limit: int | None = None,
    question_ids: set[str] | None = None,
) -> LongMemEvalIngestionReport:
    """Ingest a bounded LongMemEval subset without bundling the dataset."""

    questions = load_longmemeval_instances(
        dataset_path, limit=limit, question_ids=question_ids
    )
    cache = ExtractionCache(Path(cache_path)) if cache_path is not None else None
    results = tuple(
        [
            await ingest_longmemeval_question(service, question, extractor, cache=cache)
            for question in questions
        ]
    )
    return LongMemEvalIngestionReport(
        dataset=str(dataset_path),
        question_count=len(results),
        session_count=sum(result.session_count for result in results),
        extracted_fact_count=sum(result.extracted_fact_count for result in results),
        cache_hits=sum(result.cache_hits for result in results),
        cache_misses=sum(result.cache_misses for result in results),
        questions=results,
    )


def load_longmemeval_instances(
    dataset_path: str | Path,
    *,
    limit: int | None = None,
    question_ids: set[str] | None = None,
) -> list[LongMemEvalQuestion]:
    """Load LongMemEval JSON/JSONL records and optionally select a subset."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    path = Path(dataset_path)
    if path.suffix.lower() == ".jsonl":
        payloads = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            payloads = payload
        elif isinstance(payload, Mapping) and isinstance(payload.get("data"), list):
            payloads = payload["data"]
        else:
            raise ValueError(f"Unsupported LongMemEval dataset format: {path}")

    questions: list[LongMemEvalQuestion] = []
    for payload in payloads:
        question = LongMemEvalQuestion.from_payload(payload)
        if question_ids is not None and question.question_id not in question_ids:
            continue
        questions.append(question)
        if limit is not None and len(questions) >= limit:
            break
    return questions
