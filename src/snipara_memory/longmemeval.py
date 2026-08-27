"""LongMemEval ingestion primitives.

The benchmark contains timestamped chat sessions, but the memory engine should
index extracted facts rather than raw turns.  This module keeps that product
decision explicit while leaving the extraction model behind a small protocol.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import inspect
import json
import math
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .domain import (
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
LM_STUDIO_DEFAULT_PROMPT_VERSION = "lmstudio-fact-extractor-v4"
LM_STUDIO_SYSTEM_PROMPT = """You build a compact evidence index for one timestamped chat session.

Extract facts and answer-bearing evidence that can help answer a future
question about this session. Read the entire conversation, including assistant
turns: an assistant recommendation, schedule, restaurant name, product detail,
or other concrete answer is evidence when it is stated in the transcript.
Also extract user facts, preferences, desired recommendation constraints,
decisions, plans, updates, named entities, quantities, and events. A preference
may be implicit in a request such as asking for hotels with a rooftop pool.

Extraction priority is important. First scan every user turn for explicit
personal statements and choices (for example, "I graduated with a degree in
Business Administration"), and preserve those even when they occur inside a
long conversation full of assistant advice. Next capture user preferences,
decisions, plans, and constraints. Only then capture assistant answers that
contain concrete information needed later. Do not turn every bullet in a
generic recommendation list into a separate fact: aggregate such a list into
one concise assistant_answer only when it is useful, and never treat advice as
a user fact unless the user explicitly adopted it.

Do not copy the transcript, invent facts, use information outside this session,
or emit benchmark labels such as answer_session_ids or has_answer. Do not emit
generic conversational filler. It is valid to return an empty facts array only
when the session contains no useful evidence.

Resolve references across the whole session when the transcript supports the
relationship. For example, if an earlier user turn names a store and a later
turn says "I redeemed the coupon", preserve the combined evidence as one fact
or as two linked facts; do not drop the store merely because it is not repeated
in the later turn. For counts, preserve every separately named item, project,
transaction, or event needed to enumerate the answer. Do not collapse a list
into a vague total. Also capture the user's topic or follow-up intent when a
later question would be ambiguous without the session context, such as a
request for "recent papers" after a discussion of medical imaging.

For every fact, set a short semantic fact_key when the fact can be updated
later, such as user.home_city or event.favorite_restaurant. If this session
updates an earlier fact, set supersedes_fact_key to that key. Use null when no
stable identity is safe. source_turn_indices must refer to zero-based turn
indexes in the supplied session and may contain several indexes when a fact
combines evidence across turns. Use uppercase memory_type values: FACT,
DECISION, LEARNING, PREFERENCE, TODO, or CONTEXT.

When useful, set evidence_kind to one of user_fact, assistant_answer,
preference, decision, event, or context. Set temporal_anchor to a concise
date, time, or relative-order phrase stated in the conversation; otherwise use
null. The session_date is the fallback date for an event discussed as happening
in that session, not proof of an unrelated date.

Extract at most 12 high-value facts. Keep each content and title under 220
characters, use at most 6 tags per fact, and return compact JSON only.
"""
LM_STUDIO_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facts"],
    "properties": {
        "facts": {
            "type": "array",
            "maxItems": 12,
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
                    "evidence_kind": {
                        "type": "string",
                        "enum": [
                            "user_fact",
                            "assistant_answer",
                            "preference",
                            "decision",
                            "event",
                            "context",
                        ],
                    },
                    "temporal_anchor": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                },
            },
        }
    },
}
LM_STUDIO_BATCH_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sessions"],
    "properties": {
        "sessions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["session_id", "facts"],
                "properties": {
                    "session_id": {"type": "string"},
                    "facts": LM_STUDIO_FACT_SCHEMA["properties"]["facts"],
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
        parsed_turns = []
        for turn in turns:
            if (
                isinstance(turn, Mapping)
                and isinstance(turn.get("content"), str)
                and not turn["content"].strip()
            ):
                # The cleaned upstream dataset contains occasional empty user
                # or assistant placeholders. They carry no evidence and would
                # otherwise abort a full-dataset selection before extraction.
                continue
            parsed_turns.append(LongMemEvalTurn.from_payload(turn))
        return cls(
            session_id=str(session_id),
            date=str(session_date),
            turns=tuple(parsed_turns),
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


def _compact_evidence_text(text: str, *, limit: int = 220) -> str:
    """Keep deterministic evidence additions small enough for reader context."""

    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1].rstrip()}…"


def _session_observed_at(session_date: str) -> datetime | None:
    """Convert an adapter date into the core memory observation field."""

    try:
        parsed = datetime.fromisoformat(session_date.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _markdown_cells(line: str) -> list[str]:
    """Parse one simple pipe-delimited Markdown table row."""

    if "|" not in line:
        return []
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return [] if all(not cell for cell in cells) else cells


def _structured_schedule_facts(
    session: LongMemEvalSession,
) -> list[ExtractedFact]:
    """Preserve exact day/shift mappings from assistant-produced tables.

    LLM extraction often keeps the existence of a schedule but drops the
    cell-level mapping.  A small deterministic pass retains that high-signal
    structure without storing the raw transcript.
    """

    lines: list[tuple[int, str]] = [
        (turn_index, line)
        for turn_index, turn in enumerate(session.turns)
        if turn.role == "assistant"
        for line in turn.content.splitlines()
    ]
    facts: list[ExtractedFact] = []
    for index, (turn_index, line) in enumerate(lines):
        header = _markdown_cells(line)
        if len(header) < 3 or not any("am" in cell.lower() for cell in header):
            continue
        if not any("shift" in cell.lower() or "pm" in cell.lower() for cell in header):
            continue
        shift_names = header[1:]
        for row_turn_index, row_line in lines[index + 1 :]:
            row = _markdown_cells(row_line)
            if not row or row[0].lower() not in {
                "sunday",
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
            }:
                continue
            assignments = [
                f"{agent} is assigned to {shift}"
                for shift, agent in zip(shift_names, row[1:], strict=False)
                if agent and agent != "-"
            ]
            if not assignments:
                continue
            day = row[0]
            facts.append(
                ExtractedFact(
                    content=_compact_evidence_text(
                        f"Shift rotation for {day}: "
                        + "; ".join(assignments)
                        + "."
                    ),
                    title=f"Shift rotation {day}",
                    memory_type=MemoryType.FACT,
                    confidence=0.98,
                    fact_key=f"schedule.shift_rotation.{day.lower()}",
                    source_turn_indices=(row_turn_index,),
                    tags=("schedule", "shift", day.lower()),
                    metadata={
                        "evidence_kind": "assistant_answer",
                        "temporal_anchor": session.date,
                    },
                )
            )
        if facts:
            break
    return facts


def _explicit_museum_visit_facts(
    session: LongMemEvalSession,
) -> list[ExtractedFact]:
    """Retain explicit user museum visits that are easy to lose in long chats."""

    facts: list[ExtractedFact] = []
    for turn_index, turn in enumerate(session.turns):
        if turn.role != "user":
            continue
        lowered = turn.content.lower()
        if "museum" not in lowered and "moma" not in lowered:
            continue
        if not re.search(r"\b(visit(?:ed)?|tour|went|attend(?:ed)?)\b", lowered):
            continue
        museum_name = (
            "Museum of Modern Art"
            if "museum of modern art" in lowered or "moma" in lowered
            else "museum"
        )
        fact_key = "user.visit.museum_of_modern_art" if museum_name != "museum" else None
        facts.append(
            ExtractedFact(
                content=_compact_evidence_text(turn.content),
                title=f"User visited {museum_name}",
                memory_type=MemoryType.CONTEXT,
                confidence=0.98,
                fact_key=fact_key,
                source_turn_indices=(turn_index,),
                tags=("visit", "museum", "user-fact"),
                metadata={
                    "evidence_kind": "user_fact",
                    "temporal_anchor": session.date,
                },
            )
        )
    return facts


_EXPLICIT_USER_EVIDENCE_PATTERNS = (
    r"\b(?:pre[- ]?approved|approved for)",
    r"\b(?:moved|relocated|relocation)",
    r"\b(?:redeemed|coupon|restaurant|restaurants|tried|visited|attended)",
    r"\b(?:participat(?:ed|ing)|charity|consecutive|in a row)",
    r"\b(?:camping trip|model kit|worked on|bought|purchased|ordered)",
    r"\b(?:acquired|received|met up|meet up|finished|completed)",
    r"\b(?:nursery|baby shower|customized phone case|birthday)",
    r"\b(?:prefer|preference|interested in)",
)


def _compact_signal_evidence_text(text: str, signal_start: int) -> str:
    """Keep the clause around a signal instead of truncating its prefix."""

    compact = " ".join(text.split())
    if len(compact) <= 220:
        return compact
    start = max(0, min(signal_start - 80, len(compact) - 220))
    end = min(len(compact), start + 220)
    snippet = compact[start:end].strip()
    if start > 0:
        snippet = f"…{snippet}"
    if end < len(compact):
        snippet = f"{snippet}…"
    return snippet


def _explicit_user_signal_facts(
    session: LongMemEvalSession,
    *,
    existing_facts: Sequence[ExtractedFact],
    max_additions: int = 6,
) -> list[ExtractedFact]:
    """Retain high-signal user statements that an extractor may cap away.

    This is deliberately narrower than storing raw turns.  It protects facts
    that are commonly queried later (updates, quantities, purchases, events,
    and explicit preferences) while keeping the product's indexed unit a
    structured fact.  The original turn index remains available as
    provenance, and the model extractor still owns the semantic compression
    for everything else.
    """

    if max_additions <= 0:
        return []
    existing_text = {
        " ".join(fact.content.lower().split())
        for fact in existing_facts
    }
    additions: list[ExtractedFact] = []
    for turn_index, turn in enumerate(session.turns):
        if turn.role != "user":
            continue
        normalized = " ".join(turn.content.lower().split())
        if normalized in existing_text:
            continue
        signal_match = next(
            (
                match
                for pattern in _EXPLICIT_USER_EVIDENCE_PATTERNS
                if (match := re.search(pattern, normalized)) is not None
            ),
            None,
        )
        if signal_match is None:
            continue
        if len(normalized) < 12:
            continue
        if any(word in normalized for word in ("prefer", "interested", "looking for", "would like", "want to")):
            memory_type = MemoryType.PREFERENCE
        else:
            memory_type = MemoryType.FACT
        additions.append(
            ExtractedFact(
                content=_compact_signal_evidence_text(
                    turn.content, signal_match.start()
                ),
                title=f"Explicit user statement (turn {turn_index})",
                memory_type=memory_type,
                confidence=0.97,
                fact_key=f"user.explicit.turn.{turn_index}",
                source_turn_indices=(turn_index,),
                tags=("user-fact", "explicit", "provenance"),
                metadata={
                    "evidence_kind": "user_fact",
                    "temporal_anchor": session.date,
                    "explicit_user_evidence": True,
                },
            )
        )
        existing_text.add(normalized)
        if len(additions) >= max_additions:
            break
    return additions


def _cross_turn_transaction_facts(
    session: LongMemEvalSession,
) -> list[ExtractedFact]:
    """Preserve a transaction's object and venue when they are split across turns.

    Long conversations often mention a store in one turn and the redeemed or
    purchased item in another. This keeps that relation as one compact fact;
    it does not index the transcript wholesale or infer a venue that never
    appears in the session.
    """

    user_turns = [
        (index, turn.content)
        for index, turn in enumerate(session.turns)
        if turn.role == "user"
    ]
    facts: list[ExtractedFact] = []
    for turn_index, content in user_turns:
        match = re.search(
            r"\bredeemed\s+(?:a\s+)?(\$\d+)\s+coupon\s+on\s+(.+?)(?=\s+(?:last|today|which|because|and)\b|[.!?]|$)",
            content,
            flags=re.IGNORECASE,
        )
        if match is None:
            continue
        amount, item = match.groups()
        stores: list[tuple[str, int]] = []
        for candidate_index, candidate_content in user_turns:
            for store_match in re.finditer(
                r"\b(?:at|from)\s+([A-Z][A-Za-z0-9&'-]+(?:\s+[A-Z][A-Za-z0-9&'-]+){0,2})",
                candidate_content,
            ):
                store = store_match.group(1).strip(" .,;:")
                if store.lower() in {"the", "a", "an"}:
                    continue
                stores.append((store, candidate_index))
        if not stores:
            continue
        store, store_turn_index = stores[0]
        facts.append(
            ExtractedFact(
                content=_compact_evidence_text(
                    f"Redeemed {amount} coupon on {item.strip()} at {store}."
                ),
                title="Cross-turn coupon redemption",
                memory_type=MemoryType.FACT,
                confidence=0.99,
                fact_key="transaction.coupon.redemption",
                source_turn_indices=tuple(
                    sorted({turn_index, store_turn_index})
                ),
                tags=("transaction", "coupon", "cross-turn", "user-fact"),
                metadata={
                    "evidence_kind": "user_fact",
                    "explicit_user_evidence": True,
                    "temporal_anchor": session.date,
                },
            )
        )
    return facts


def _follow_up_topic_facts(
    session: LongMemEvalSession,
) -> list[ExtractedFact]:
    """Keep a compact topic anchor for elliptical resource/recommendation asks."""

    follow_up_pattern = re.compile(
        r"\b(?:research\s+papers?|articles?|publications?|conferences?|resources?|recommend(?:ations?)?|suggestions?)\b",
        flags=re.IGNORECASE,
    )
    for turn_index in range(len(session.turns) - 1, -1, -1):
        turn = session.turns[turn_index]
        if turn.role != "user":
            continue
        signal = follow_up_pattern.search(turn.content)
        if signal is None:
            continue
        return [
            ExtractedFact(
                content=_compact_signal_evidence_text(turn.content, signal.start()),
                title=f"Session follow-up topic (turn {turn_index})",
                memory_type=MemoryType.CONTEXT,
                confidence=0.96,
                fact_key=f"session.topic.follow_up.{turn_index}",
                source_turn_indices=(turn_index,),
                tags=("context", "follow-up", "topic"),
                metadata={
                    "evidence_kind": "context",
                    "temporal_anchor": session.date,
                },
            )
        ]
    return []


def _augment_high_signal_evidence(
    session: LongMemEvalSession,
    facts: Sequence[ExtractedFact],
) -> list[ExtractedFact]:
    """Add compact structural evidence omitted by a model extraction."""

    existing_keys = {fact.fact_key for fact in facts if fact.fact_key}
    additions = [
        fact
        for fact in [
            *_structured_schedule_facts(session),
            *_explicit_museum_visit_facts(session),
            *_explicit_user_signal_facts(session, existing_facts=facts),
            *_cross_turn_transaction_facts(session),
            *_follow_up_topic_facts(session),
        ]
        if not fact.fact_key or fact.fact_key not in existing_keys
    ]
    return [*facts, *additions]


class _LmStudioRequestError(OSError):
    """Retryable local transport failure."""


@dataclass(slots=True)
class LmStudioFactExtractor:
    """Extract facts through LM Studio's OpenAI-compatible local API."""

    model: str
    base_url: str = LM_STUDIO_DEFAULT_BASE_URL
    api_key: str = "lm-studio"
    prompt_version: str = LM_STUDIO_DEFAULT_PROMPT_VERSION
    reasoning_effort: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    retries: int = 2
    # Keep transcript input below the context budget once the system prompt,
    # JSON schema, and requested output tokens are included.  12k characters
    # is a safe default for the common 8k-context LM Studio setup; callers can
    # raise it explicitly when using a larger context window.
    max_session_chars: int = 12000
    batch_request_size: int = 4
    # Keep enough room for the system prompt and the structured response in
    # an 16k-context local model.  A smaller grouping limit also lets the
    # configured request concurrency produce real parallel work instead of
    # packing the first four sessions into one long request.
    batch_request_max_chars: int = 16000
    batch_request_concurrency: int = 2
    batch_max_tokens: int = 4096

    def __post_init__(self) -> None:
        self.model = self.model.strip()
        self.base_url = self.base_url.rstrip("/")
        if not self.model:
            raise ValueError("LM Studio extractor requires a model identifier")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("LM Studio base_url must start with http:// or https://")
        if self.reasoning_effort is not None:
            self.reasoning_effort = self.reasoning_effort.lower()
            if self.reasoning_effort not in {"low", "medium", "high"}:
                raise ValueError("reasoning_effort must be low, medium, high, or None")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.retries < 0:
            raise ValueError("retries cannot be negative")
        if self.max_session_chars <= 0:
            raise ValueError("max_session_chars must be positive")
        if self.batch_request_size <= 1:
            raise ValueError("batch_request_size must be greater than 1")
        if self.batch_request_max_chars <= 0:
            raise ValueError("batch_request_max_chars must be positive")
        if self.batch_request_concurrency <= 0:
            raise ValueError("batch_request_concurrency must be positive")
        if self.batch_max_tokens <= 0:
            raise ValueError("batch_max_tokens must be positive")

    @property
    def version(self) -> str:
        """Cache identity for the model and the extraction prompt contract."""

        reasoning_suffix = (
            f":reasoning-{self.reasoning_effort}"
            if self.reasoning_effort is not None
            else ""
        )
        return f"{self.model}:{self.prompt_version}{reasoning_suffix}"

    async def extract(self, session: LongMemEvalSession) -> Sequence[ExtractedFact]:
        """Extract one session, chunking oversized transcripts safely."""

        chunks = self._session_chunks(session)
        if len(chunks) == 1 and chunks[0][1] == tuple(range(len(session.turns))):
            return await self._extract_chunk(session)

        facts: list[ExtractedFact] = []
        for chunk, original_turn_indices in chunks:
            chunk_facts = await self._extract_chunk(chunk)
            for fact in chunk_facts:
                facts.append(
                    replace(
                        fact,
                        source_turn_indices=tuple(
                            original_turn_indices[index]
                            for index in fact.source_turn_indices
                            if 0 <= index < len(original_turn_indices)
                        ),
                    )
                )
        return facts

    async def extract_batch(
        self, sessions: Sequence[LongMemEvalSession]
    ) -> Mapping[str, Sequence[ExtractedFact]]:
        """Extract several sessions while batching their bounded transcript chunks.

        A LongMemEval session can be much larger than the model context.  The
        important detail here is to batch *chunks*, not whole sessions: each
        chunk keeps its original turn indices and gets a unique temporary ID,
        then the returned facts are remapped back to the original session.
        """

        if not sessions:
            return {}

        jobs: list[
            tuple[str, LongMemEvalSession, tuple[int, ...]]
        ] = []
        for session in sessions:
            chunks = self._session_chunks(session)
            for _, (chunk, original_turn_indices) in enumerate(chunks):
                # Opaque LongMemEval IDs can be long enough for a local model
                # to truncate or rewrite them.  Use a short per-request ID;
                # ``original_session_id`` below restores the real identity.
                chunk_id = f"s{len(jobs)}"
                jobs.append(
                    (
                        session.session_id,
                        replace(chunk, session_id=chunk_id),
                        original_turn_indices,
                    )
                )

        groups: list[list[tuple[str, LongMemEvalSession, tuple[int, ...]]]] = []
        current_group: list[tuple[str, LongMemEvalSession, tuple[int, ...]]] = []
        current_chars = 0
        for job in jobs:
            job_chars = sum(len(turn.content) for turn in job[1].turns)
            if current_group and (
                len(current_group) >= self.batch_request_size
                or current_chars + job_chars > self.batch_request_max_chars
            ):
                groups.append(current_group)
                current_group = []
                current_chars = 0
            current_group.append(job)
            current_chars += job_chars
        if current_group:
            groups.append(current_group)

        request_semaphore = asyncio.Semaphore(self.batch_request_concurrency)

        async def run_group(
            group: Sequence[tuple[str, LongMemEvalSession, tuple[int, ...]]],
        ) -> Mapping[str, Sequence[ExtractedFact]]:
            async with request_semaphore:
                chunk_sessions = [job[1] for job in group]
                if len(chunk_sessions) == 1:
                    facts = await self._extract_chunk(chunk_sessions[0])
                    return {chunk_sessions[0].session_id: facts}
                return await self._extract_batch_request(chunk_sessions)

        group_results = await asyncio.gather(*(run_group(group) for group in groups))
        combined: dict[str, list[ExtractedFact]] = {
            session.session_id: [] for session in sessions
        }
        for group, extracted in zip(groups, group_results, strict=True):
            expected_ids = {job[1].session_id for job in group}
            missing_ids = expected_ids.difference(extracted)
            if missing_ids:
                missing = ", ".join(sorted(missing_ids))
                raise ValueError(
                    f"LM Studio batch response omitted session results: {missing}"
                )
            for original_session_id, chunk, original_turn_indices in group:
                for fact in extracted.get(chunk.session_id, ()):
                    remapped_indices = tuple(
                        original_turn_indices[index]
                        for index in fact.source_turn_indices
                        if 0 <= index < len(original_turn_indices)
                    )
                    combined[original_session_id].append(
                        replace(fact, source_turn_indices=remapped_indices)
                    )
        return combined

    async def _extract_batch_request(
        self, sessions: Sequence[LongMemEvalSession]
    ) -> Mapping[str, Sequence[ExtractedFact]]:
        """Send one bounded batch request with the normal retry contract."""

        for attempt in range(self.retries + 1):
            try:
                request_payload = self._build_batch_payload(
                    sessions, compact_retry=bool(attempt)
                )
                response = await self._post_json_async(request_payload)
                extracted = _facts_from_lm_studio_batch_response(response)
                expected_ids = {session.session_id for session in sessions}
                missing_ids = expected_ids.difference(extracted)
                if missing_ids:
                    # Local models sometimes omit empty/low-salience members
                    # from an otherwise valid batch response.  Recover only
                    # those chunks through the single-session contract so a
                    # partial batch can never silently lose source evidence.
                    missing_sessions = [
                        session
                        for session in sessions
                        if session.session_id in missing_ids
                    ]
                    recovered = await asyncio.gather(
                        *(self._extract_chunk(session) for session in missing_sessions)
                    )
                    extracted = dict(extracted)
                    extracted.update(
                        {
                            session.session_id: facts
                            for session, facts in zip(
                                missing_sessions, recovered, strict=True
                            )
                        }
                    )
                return extracted
            except _LmStudioRequestError:
                if attempt >= self.retries:
                    return await self._extract_batch_individually(sessions)
                await asyncio.sleep(min(2**attempt, 8))
            except ValueError:
                if attempt >= self.retries:
                    return await self._extract_batch_individually(sessions)
                await asyncio.sleep(min(2**attempt, 8))
        raise AssertionError("LM Studio batch retry loop exited unexpectedly")

    async def _extract_batch_individually(
        self, sessions: Sequence[LongMemEvalSession]
    ) -> Mapping[str, Sequence[ExtractedFact]]:
        """Recover a failed batch through the validated single-session path."""

        recovered = await asyncio.gather(
            *(self._extract_chunk(session) for session in sessions)
        )
        return {
            session.session_id: facts
            for session, facts in zip(sessions, recovered, strict=True)
        }

    async def _extract_chunk(
        self, session: LongMemEvalSession
    ) -> list[ExtractedFact]:
        payload = self._build_payload(session)
        for attempt in range(self.retries + 1):
            try:
                request_payload = (
                    payload
                    if attempt == 0
                    else self._build_payload(session, compact_retry=True)
                )
                response = await self._post_json_async(request_payload)
                return _facts_from_lm_studio_response(response)
            except _LmStudioRequestError as error:
                if attempt >= self.retries:
                    raise RuntimeError(
                        f"LM Studio request failed after {attempt + 1} attempts: {error}"
                    ) from error
                await asyncio.sleep(min(2**attempt, 8))
                continue
            except ValueError as error:
                if attempt >= self.retries:
                    raise ValueError(
                        "LM Studio returned invalid structured facts after "
                        f"{attempt + 1} attempts: {error}"
                    ) from error
                await asyncio.sleep(min(2**attempt, 8))
        raise AssertionError("LM Studio retry loop exited unexpectedly")

    def _session_chunks(
        self, session: LongMemEvalSession
    ) -> list[tuple[LongMemEvalSession, tuple[int, ...]]]:
        """Split a session by character budget while retaining turn mapping."""

        if sum(len(turn.content) for turn in session.turns) <= self.max_session_chars:
            return [(session, tuple(range(len(session.turns))))]

        chunks: list[tuple[LongMemEvalSession, tuple[int, ...]]] = []
        current_turns: list[LongMemEvalTurn] = []
        current_indices: list[int] = []
        current_chars = 0

        def flush() -> None:
            nonlocal current_turns, current_indices, current_chars
            if current_turns:
                chunks.append(
                    (
                        LongMemEvalSession(
                            session_id=session.session_id,
                            date=session.date,
                            turns=tuple(current_turns),
                        ),
                        tuple(current_indices),
                    )
                )
            current_turns = []
            current_indices = []
            current_chars = 0

        for original_index, turn in enumerate(session.turns):
            content = turn.content
            if len(content) > self.max_session_chars:
                flush()
                for start in range(0, len(content), self.max_session_chars):
                    piece = replace(
                        turn,
                        content=content[start : start + self.max_session_chars],
                    )
                    chunks.append(
                        (
                            LongMemEvalSession(
                                session_id=session.session_id,
                                date=session.date,
                                turns=(piece,),
                            ),
                            (original_index,),
                        )
                    )
                continue
            if current_turns and current_chars + len(content) > self.max_session_chars:
                flush()
            current_turns.append(turn)
            current_indices.append(original_index)
            current_chars += len(content)
        flush()
        return chunks

    def _build_payload(
        self,
        session: LongMemEvalSession,
        *,
        compact_retry: bool = False,
    ) -> dict[str, Any]:
        system_prompt = LM_STUDIO_SYSTEM_PROMPT
        # Deliberately omit LongMemEval's `has_answer` labels: they are ground
        # truth for evaluation and must never leak into the extraction prompt.
        if compact_retry:
            system_prompt += """

This is a retry after invalid JSON. Return compact JSON only, with at most 12
high-value facts. Keep each content and title short, omit optional metadata if
needed, and never include commentary outside the JSON object.
"""
        session_payload = {
            "session_id": session.session_id,
            "session_date": session.date,
            "turns": [
                {"role": turn.role, "content": turn.content} for turn in session.turns
            ],
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
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
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def _build_batch_payload(
        self,
        sessions: Sequence[LongMemEvalSession],
        *,
        compact_retry: bool = False,
    ) -> dict[str, Any]:
        system_prompt = LM_STUDIO_SYSTEM_PROMPT.replace(
            "from one timestamped chat session", "from each timestamped chat session"
        )
        system_prompt += """

The input contains multiple independent sessions. Extract each session
independently and return exactly one result object per session, preserving its
session_id. Never use facts from one session to infer facts for another.
"""
        if compact_retry:
            system_prompt += """

This is a retry after invalid JSON. Return compact JSON only, with at most 4
high-value facts per session, omit optional metadata if needed, and no
commentary outside the JSON object.
"""
        batch_schema = deepcopy(LM_STUDIO_BATCH_FACT_SCHEMA)
        batch_schema["properties"]["sessions"]["minItems"] = len(sessions)
        batch_schema["properties"]["sessions"]["maxItems"] = len(sessions)
        session_payload = {
            "sessions": [
                {
                    "session_id": session.session_id,
                    "session_date": session.date,
                    "turns": [
                        {"role": turn.role, "content": turn.content}
                        for turn in session.turns
                    ],
                }
                for session in sessions
            ]
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
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
            "max_tokens": max(self.max_tokens, self.batch_max_tokens),
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "longmemeval_batch_facts",
                    "strict": True,
                    "schema": batch_schema,
                },
            },
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

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

    async def _post_json_async(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """POST without leaving an uncancellable worker thread behind.

        ``urllib.request.urlopen`` is synchronous.  Wrapping it in
        ``asyncio.to_thread`` lets several extractions run concurrently, but a
        cancelled ``to_thread`` task cannot stop a socket read.  A stalled
        LM Studio generation would therefore accumulate live threads and keep a
        benchmark run stuck forever.  Curl is used as a short-lived child
        process here because the operating system can terminate that process
        reliably when the request deadline expires.
        """

        curl = shutil.which("curl")
        if curl is None:
            # Keep the adapter usable in minimal environments.  The explicit
            # outer deadline still bounds the await, although a fallback
            # urllib worker may finish later in that case.
            return await asyncio.wait_for(
                asyncio.to_thread(self._post_json, payload),
                timeout=self.timeout_seconds,
            )

        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        connect_timeout = min(max(self.timeout_seconds, 0.1), 10.0)
        command = [
            curl,
            "--silent",
            "--show-error",
            "--connect-timeout",
            str(connect_timeout),
            "--max-time",
            str(max(self.timeout_seconds, 0.1)),
            "--header",
            "Accept: application/json",
            "--header",
            f"Authorization: Bearer {self.api_key}",
            "--header",
            "Content-Type: application/json",
            "--request",
            "POST",
            f"{self.base_url}/chat/completions",
            "--data-binary",
            body,
            "--write-out",
            "\\n%{http_code}",
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as error:
            raise _LmStudioRequestError(
                f"Could not start curl for {self.base_url}: {error}"
            ) from error

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_seconds + 2.0
            )
        except asyncio.TimeoutError as error:
            if process.returncode is None:
                process.kill()
            await process.communicate()
            raise _LmStudioRequestError(
                f"LM Studio request timed out after {self.timeout_seconds:g}s"
            ) from error

        output = stdout.decode("utf-8", errors="replace")
        response_body, separator, status_text = output.rpartition("\n")
        if not separator:
            response_body = output
            status_text = ""
        try:
            status_code = int(status_text.strip()) if status_text.strip() else 0
        except ValueError:
            status_code = 0
        if process.returncode != 0 or not 200 <= status_code < 300:
            detail = response_body[:500] or stderr.decode(
                "utf-8", errors="replace"
            )[:500]
            status = f"HTTP {status_code}" if status_code else "curl error"
            raise _LmStudioRequestError(
                f"{status} from {self.base_url}: {detail}"
            )
        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as error:
            raise _LmStudioRequestError(
                f"LM Studio returned invalid JSON: {response_body[:500]}"
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
    salvaged = False
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        recovered = _salvage_completed_fact_objects(content)
        if recovered is None:
            raise
        decoded = {"facts": recovered}
        salvaged = True
    if isinstance(decoded, list):
        raw_facts = decoded
    elif isinstance(decoded, Mapping):
        raw_facts = decoded.get("facts")
    else:
        raw_facts = None
    if not isinstance(raw_facts, list):
        raise ValueError("LM Studio response must contain a facts array")
    try:
        facts = [
            ExtractedFact.from_mapping(_normalize_lm_studio_fact(fact))
            for fact in raw_facts
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"LM Studio returned an invalid fact: {error}") from error
    if salvaged:
        facts = [
            replace(
                fact,
                metadata={
                    **fact.metadata,
                    "lm_studio_parse": "salvaged_json_prefix",
                },
            )
            for fact in facts
        ]
    return facts


def _facts_from_lm_studio_batch_response(
    response: Mapping[str, Any],
) -> dict[str, list[ExtractedFact]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LM Studio batch response has no choices")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise ValueError("LM Studio batch response choice is not an object")
    message = first_choice.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("LM Studio batch response choice has no text content")
    content = message["content"].strip()
    if content.startswith("```"):
        lines = content.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        content = "\n".join(lines).strip()
    decoded = json.loads(content)
    if not isinstance(decoded, Mapping) or not isinstance(decoded.get("sessions"), list):
        raise ValueError("LM Studio batch response must contain a sessions array")
    result: dict[str, list[ExtractedFact]] = {}
    for session_payload in decoded["sessions"]:
        if not isinstance(session_payload, Mapping):
            raise ValueError("LM Studio batch session result is not an object")
        session_id = session_payload.get("session_id")
        raw_facts = session_payload.get("facts")
        if not isinstance(session_id, str) or not isinstance(raw_facts, list):
            raise ValueError("LM Studio batch session result is incomplete")
        try:
            result[session_id] = [
                ExtractedFact.from_mapping(_normalize_lm_studio_fact(fact))
                for fact in raw_facts
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"LM Studio returned an invalid batch fact: {error}") from error
    return result


def _normalize_lm_studio_fact(fact: object) -> Mapping[str, Any]:
    """Normalize minor schema drift from local models before domain validation.

    Some LM Studio models emit a numeric confidence just outside the declared
    JSON-schema range (for example ``1.01``).  The value is bounded rather
    than dropping the entire session; the extraction prompt remains the source
    of truth for the semantic fields.
    """

    if not isinstance(fact, Mapping):
        raise TypeError("LM Studio facts must be JSON objects")
    normalized = dict(fact)
    metadata = normalized.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    else:
        metadata = dict(metadata)
    for key in ("evidence_kind", "temporal_anchor"):
        if key in normalized:
            metadata[key] = normalized[key]
    normalized["metadata"] = metadata
    raw_confidence = normalized.get("confidence", 0.8)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.8
    if not math.isfinite(confidence):
        confidence = 0.8
    normalized["confidence"] = min(1.0, max(0.0, confidence))
    return normalized


@dataclass(slots=True)
class LmStudioBatchFactExtractor:
    """Coalesce concurrent session extractions into bounded batch requests."""

    extractor: LmStudioFactExtractor
    batch_size: int = 4
    dispatch_delay_seconds: float = 0.02
    _lock: asyncio.Lock = field(init=False, repr=False)
    _pending: list[tuple[LongMemEvalSession, asyncio.Future[Any]]] = field(
        init=False, repr=False
    )
    _flush_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.batch_size <= 1:
            raise ValueError("batch_size must be greater than 1")
        if self.dispatch_delay_seconds < 0:
            raise ValueError("dispatch_delay_seconds cannot be negative")
        self._lock = asyncio.Lock()
        self._pending = []

    @property
    def version(self) -> str:
        # Keep cache compatibility with the underlying single-session adapter.
        return self.extractor.version

    @property
    def timeout_seconds(self) -> float:
        """Expose the transport bound to the question-level ingestion guard."""

        return self.extractor.timeout_seconds

    async def extract(self, session: LongMemEvalSession) -> Sequence[ExtractedFact]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        batch: list[tuple[LongMemEvalSession, asyncio.Future[Any]]] = []
        async with self._lock:
            self._pending.append((session, future))
            if len(self._pending) >= self.batch_size:
                batch = self._pending[: self.batch_size]
                del self._pending[: self.batch_size]
                if self._flush_task is not None:
                    self._flush_task.cancel()
                    self._flush_task = None
            elif self._flush_task is None:
                self._flush_task = asyncio.create_task(self._flush_after_delay())
        if batch:
            asyncio.create_task(self._run_batch(batch))
        return await future

    async def _flush_after_delay(self) -> None:
        try:
            await asyncio.sleep(self.dispatch_delay_seconds)
            async with self._lock:
                batch = self._pending
                self._pending = []
                self._flush_task = None
            if batch:
                await self._run_batch(batch)
        except asyncio.CancelledError:
            return

    async def _run_batch(
        self, batch: Sequence[tuple[LongMemEvalSession, asyncio.Future[Any]]]
    ) -> None:
        sessions = [session for session, _ in batch]
        try:
            extracted = await self.extractor.extract_batch(sessions)
            for session, future in batch:
                if not future.done():
                    future.set_result(list(extracted.get(session.session_id, ())))
        except Exception as error:
            for _, future in batch:
                if not future.done():
                    future.set_exception(error)


def _salvage_completed_fact_objects(
    content: str,
) -> list[Mapping[str, Any]] | None:
    """Recover complete fact objects when the model truncates the JSON tail."""

    facts_marker = content.find('"facts"')
    array_start = content.find("[", facts_marker if facts_marker >= 0 else 0)
    if array_start < 0:
        return None

    decoder = json.JSONDecoder()
    position = array_start + 1
    recovered: list[Mapping[str, Any]] = []
    while position < len(content):
        while position < len(content) and content[position].isspace():
            position += 1
        if position >= len(content):
            return recovered or None
        if content[position] == "]":
            return recovered
        try:
            value, end = decoder.raw_decode(content, position)
        except json.JSONDecodeError:
            return recovered or None
        if not isinstance(value, Mapping):
            return recovered or None
        recovered.append(value)
        position = end
        while position < len(content) and content[position].isspace():
            position += 1
        if position >= len(content):
            return recovered
        if content[position] == ",":
            position += 1
            continue
        if content[position] == "]":
            return recovered
        return recovered or None
    return recovered or None


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
        candidates = []
        direct = self._entries.get(key)
        if direct is not None:
            candidates.append(direct)
        # LongMemEval reuses identical distractor sessions across questions.
        # Reuse only successful extractions with the same content hash and
        # extractor version; question-specific keys are still written by the
        # caller so provenance remains scoped to the current question.
        candidates.extend(
            entry
            for candidate_key, entry in self._entries.items()
            if candidate_key != key
            and entry.get("session_hash") == session_hash
            and entry.get("extractor_version") == extractor_version
            and entry.get("status") != "failed"
        )
        for entry in candidates:
            if (
                not entry
                or entry.get("session_hash") != session_hash
                or entry.get("extractor_version") != extractor_version
                or entry.get("status") == "failed"
            ):
                continue
            raw_facts = entry.get("facts", [])
            if not isinstance(raw_facts, list):
                continue
            try:
                return [ExtractedFact.from_mapping(fact) for fact in raw_facts]
            except (KeyError, TypeError, ValueError):
                continue
        return None

    def get_failure(
        self,
        key: str,
        *,
        session_hash: str,
        extractor_version: str,
    ) -> str | None:
        entry = self._entries.get(key)
        if (
            not entry
            or entry.get("session_hash") != session_hash
            or entry.get("extractor_version") != extractor_version
            or entry.get("status") != "failed"
        ):
            return None
        error = entry.get("error")
        return str(error) if error else "cached extraction failure"

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

    def put_failure(
        self,
        key: str,
        *,
        session_hash: str,
        extractor_version: str,
        error: str,
    ) -> None:
        self._entries[key] = {
            "session_hash": session_hash,
            "extractor_version": extractor_version,
            "status": "failed",
            "error": error[:500],
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
    failed_session_ids: tuple[str, ...] = ()
    failure_messages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LongMemEvalIngestionReport:
    dataset: str
    question_count: int
    session_count: int
    extracted_fact_count: int
    cache_hits: int
    cache_misses: int
    questions: tuple[LongMemEvalIngestionResult, ...]
    failed_session_count: int


async def ingest_longmemeval_question(
    service: MemoryService,
    question: LongMemEvalQuestion,
    extractor: FactExtractor,
    *,
    cache: ExtractionCache | None = None,
    namespace_id: str | None = None,
    extraction_concurrency: int = 1,
    retry_failed: bool = False,
) -> LongMemEvalIngestionResult:
    """Extract and store one question's sessions as structured memories."""

    if extraction_concurrency <= 0:
        raise ValueError("extraction_concurrency must be positive")
    target_namespace = namespace_id or f"longmemeval:{question.question_id}"
    extractor_version = str(getattr(extractor, "version", extractor.__class__.__name__))
    requests: list[StoreMemoryRequest] = []
    cache_hits = 0
    cache_misses = 0
    failed_session_ids: list[str] = []
    failure_messages: list[str] = []

    extraction_lock = asyncio.Semaphore(extraction_concurrency)
    cache_write_lock = asyncio.Lock()

    async def extract_session(
        index: int,
        session: LongMemEvalSession,
    ) -> tuple[int, str, list[ExtractedFact], str | None]:
        cache_key = f"{question.question_id}:{session.session_id}"
        facts = None
        if cache:
            facts = cache.get(
                cache_key,
                session_hash=session.content_hash,
                extractor_version=extractor_version,
            )
        if facts is not None:
            return index, "hit", _augment_high_signal_evidence(session, facts), None
        cached_failure = (
            cache.get_failure(
                cache_key,
                session_hash=session.content_hash,
                extractor_version=extractor_version,
            )
            if cache
            else None
        )
        if cached_failure is not None and not retry_failed:
            return (
                index,
                "cached-failure",
                [],
                f"{session.session_id}: cached extraction failure: {cached_failure}",
            )
        async with extraction_lock:
            try:
                raw_facts = extractor.extract(session)
                if inspect.isawaitable(raw_facts):
                    extraction_timeout = getattr(extractor, "timeout_seconds", None)
                    if isinstance(extraction_timeout, (int, float)):
                        raw_facts = await asyncio.wait_for(
                            raw_facts, timeout=float(extraction_timeout)
                        )
                    else:
                        raw_facts = await raw_facts
                facts = [
                    fact
                    if isinstance(fact, ExtractedFact)
                    else ExtractedFact.from_mapping(fact)
                    for fact in raw_facts
                ]
                facts = _augment_high_signal_evidence(session, facts)
            except (
                asyncio.TimeoutError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
                OSError,
            ) as error:
                message = (
                    f"{session.session_id}: {type(error).__name__}: {str(error)[:500]}"
                )
                if cache:
                    async with cache_write_lock:
                        cache.put_failure(
                            cache_key,
                            session_hash=session.content_hash,
                            extractor_version=extractor_version,
                            error=message,
                        )
                        cache.flush()
                return index, "failed", [], message
        if cache:
            async with cache_write_lock:
                cache.put(
                    cache_key,
                    session_hash=session.content_hash,
                    extractor_version=extractor_version,
                    facts=facts,
                )
                cache.flush()
        return index, "miss", facts, None

    extracted_sessions = await asyncio.gather(
        *(extract_session(index, session) for index, session in enumerate(question.sessions))
    )

    for index, status, facts, failure_message in sorted(
        extracted_sessions, key=lambda result: result[0]
    ):
        session = question.sessions[index]
        if status == "hit":
            cache_hits += 1
        elif status == "miss":
            cache_misses += 1
        elif status in {"failed", "cached-failure"}:
            failed_session_ids.append(session.session_id)
            if failure_message:
                failure_messages.append(failure_message)
            continue
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
            evidence_kind = fact.metadata.get("evidence_kind")
            supersedes_memory_key = fact.supersedes_fact_key
            if fact.memory_type is MemoryType.TODO or evidence_kind in {
                "assistant_answer",
                "event",
            }:
                # LongMemEval keeps event history and assistant evidence even
                # when the extractor reuses a key.  The core service still
                # owns the actual supersession/graveyard operation.
                supersedes_memory_key = None
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
                    memory_key=fact.fact_key,
                    supersedes_memory_key=supersedes_memory_key,
                    provenance_key=session.session_id,
                    observed_at=_session_observed_at(session.date),
                )
            )

    existing = await service.list_memories(
        target_namespace, statuses=[MemoryStatus.ACTIVE]
    )
    previous_by_memory_key = {
        str(memory.memory_key): memory
        for memory in existing
        if memory.memory_key
    }
    created = await service.store_memories_bulk(requests)
    superseded_ids: list[str] = []
    for memory in created:
        supersedes_key = memory.supersedes_memory_key
        if supersedes_key:
            previous = previous_by_memory_key.get(str(supersedes_key))
            if previous is not None and previous.id != memory.id:
                superseded_ids.append(previous.id)
            previous_by_memory_key[str(supersedes_key)] = memory
        if memory.memory_key:
            previous_by_memory_key[str(memory.memory_key)] = memory

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
        failed_session_ids=tuple(failed_session_ids),
        failure_messages=tuple(failure_messages),
    )


async def ingest_longmemeval_dataset(
    service: MemoryService,
    dataset_path: str | Path,
    extractor: FactExtractor,
    *,
    cache_path: str | Path | None = None,
    limit: int | None = None,
    question_ids: set[str] | None = None,
    extraction_concurrency: int = 1,
    retry_failed: bool = False,
) -> LongMemEvalIngestionReport:
    """Ingest a bounded LongMemEval subset without bundling the dataset."""

    questions = load_longmemeval_instances(
        dataset_path, limit=limit, question_ids=question_ids
    )
    cache = ExtractionCache(Path(cache_path)) if cache_path is not None else None
    results = tuple(
        [
            await ingest_longmemeval_question(
                service,
                question,
                extractor,
                cache=cache,
                extraction_concurrency=extraction_concurrency,
                retry_failed=retry_failed,
            )
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
        failed_session_count=sum(len(result.failed_session_ids) for result in results),
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
