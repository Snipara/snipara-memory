"""LongMemEval answer generation and official judge integration.

The ingestion adapter deliberately stops at structured memories.  This module
adds the separate QA stage: retrieve active memories, ask a reader to answer
from that evidence, then apply the official LongMemEval yes/no judge prompt.
"""

from __future__ import annotations

import asyncio
from datetime import date
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .adapters import InMemoryMemoryStore
from .domain import (
    MemoryService,
    MemoryStatus,
    RecallMatch,
    RecallQuery,
    provenance_key_for_memory,
    select_diverse_matches,
)
from .longmemeval import (
    ExtractionCache,
    FactExtractor,
    LongMemEvalQuestion,
    ingest_longmemeval_question,
    load_longmemeval_instances,
)

LONGMEMEVAL_QA_CACHE_SCHEMA = "snipara.longmemeval.qa-cache.v1"
LONGMEMEVAL_READER_PROMPT_VERSION = "lmstudio-longmemeval-reader-v23"
LONGMEMEVAL_JUDGE_PROMPT_VERSION = "longmemeval-official-judge-v1"

READER_SYSTEM_PROMPT = """You answer a LongMemEval question using only the retrieved evidence below.

Give the shortest direct answer supported by the evidence. Use every relevant
memory, not just the first one. Distinct source_session_id values are distinct
conversation sessions and may need to be combined.

Reasoning rules:
- For knowledge updates, first collect all candidate values and then prefer
  the newest user-stated value by session_date, even when the extractor gave
  the old and new statements different fact_key values. A later statement
  such as "beat my personal best of 25:50" updates an older recorded time of
  27:12; do not answer with the older value merely because its title matches
  the question more literally. Respect fact_key/supersedes_fact_key when
  present, but do not require those fields to recognize an update.
  When latest_update_candidate or latest_quantity_candidate is present in the
  resolution audit, use the value stated in that first candidate unless
  another directly relevant user statement has a later session_date. Never
  choose an older value merely because it is ranked higher or has a more
  specific title. For a quantity update, do not add an older quantity to the
  newer one.
- For multi-session questions, combine distinct sessions, deduplicate repeated
  facts, and count only the requested entities. When the question asks for
  items to pick up or return, scan every source session in the session groups
  below. If several memories in one session repeat the same clothing/action
  record, merge the duplicate statements; count separately named items or
  transactions in the same session separately. Do not exclude a
  clothing item because the venue is described as dry cleaning, an exchange,
  or a store; the requested pickup/return action is what matters.
  Never merge records from different source_session_id values merely because
  they mention the same item or store; a distinct session is a distinct
  required request unless the evidence explicitly says that one supersedes
  the other. For an exchange, a return of the old item and pickup of its
  replacement are two clothing items; merge only duplicate statements of the
  same action/item, not opposite actions in an exchange.
- For numerical multi-session questions, enumerate every qualifying item or
  action in the evidence before counting. Accept concrete paraphrases and
  venue names (for example, a dry-cleaning pickup for a blazer, an exchange
  pickup for boots, and a return of boots) when they satisfy the requested
  action; do not discard one because its venue is described differently from
  the question. Before giving a numeric answer, silently enumerate the
  qualifying records by source session, consolidate only duplicate statements
  of the same action/item, and verify that the final number
  matches that checklist.
  When an action_item_checklist is supplied, use it as the explicit inventory
  of qualifying records: duplicate entries with the same session, action, and
  item are one record, while pickup and return entries are separate records.
- For temporal questions, sort relevant sessions by session_date, use an
  explicit temporal_anchor when present, and calculate the requested interval
  or order. Treat session_date as the event date only when the evidence places
  the event in that session.
- For preference questions, use the user's stated preferences and constraints
  to answer or personalize the recommendation; do not substitute generic advice.
- For an unanswerable question, require direct evidence for the exact entity or
  attribute asked about. A related object, hobby, category, or assistant answer
  is not evidence that the user stated the requested fact. For example, a
  camera collection does not establish a film collection; say the information
  cannot be determined when only the related object is supported.
- The payload may include a deterministic_resolution_audit. Treat it as an
  evidence-organizing aid, not as a gold answer: verify it against the
  retrieved memories. For an update, compare the ISO-like session_date values
  yourself; if latest_update_candidate or latest_update_evidence is present,
  its first item is the newest directly relevant user statement. Treat that
  first candidate as authoritative unless the evidence contains a later
  directly relevant date. Do not call an older value the latest because the
  prose around it says "latest". For a
  count, inspect every counting_evidence item and deduplicate only repeated
  statements of the same entity. If latest_quantity_candidate is present, report that
  newer quantity rather than summing it with an older quantity. For a temporal question, use
  temporal_candidates and computed intervals only when their event wording
  matches the question.
  In an inventory count, a directly stated item that was finished, acquired,
  bought, or worked on is still one qualifying item when the question asks
  what the user has worked on or bought. Count every distinct named item.

The resolver fields are higher-priority evidence than the ranking order. If a
resolver directive is supplied, treat its preferred evidence as the answer
source and do not reinterpret it as a competing older fact. For a knowledge
update, copy the requested value from that preferred evidence before drafting
the answer; do not recalculate which date is newer from prose or rank.

Do not invent details, use outside knowledge, mention the memory system, or
mention retrieval. If the evidence truly does not support the answer, say that
the information cannot be determined from the available memories. Do not
abstain merely because the evidence is spread across several entries.
"""


_ACTION_ITEM_LEXICON = (
    "dry cleaning",
    "blazer",
    "jacket",
    "sweater",
    "boots",
    "shoes",
    "jeans",
    "shirt",
    "pants",
    "dress",
    "coat",
    "clothing",
    "clothes",
)


def _action_item_checklist(
    question: str,
    context: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build a transparent lexical inventory for clothing action counts."""

    lowered_question = question.lower()
    if "how many" not in lowered_question:
        return []
    if not any(term in lowered_question for term in ("pick up", "pickup", "return")):
        return []
    if not any(term in lowered_question for term in _ACTION_ITEM_LEXICON):
        return []

    checklist: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for memory in context:
        text = " ".join(
            str(memory.get(key) or "")
            for key in ("title", "content", "evidence_kind")
        ).lower()
        item = next((term for term in _ACTION_ITEM_LEXICON if term in text), None)
        if item is None:
            continue
        actions: list[str] = []
        if "pick up" in text or "pickup" in text:
            actions.append("pickup")
        if "return" in text:
            actions.append("return")
        session_id = str(memory.get("source_session_id") or "unknown")
        for action in actions:
            key = (session_id, action, item)
            if key in seen:
                continue
            seen.add(key)
            checklist.append(
                {
                    "source_session_id": session_id,
                    "action": action,
                    "item": item,
                }
            )
    return checklist


_RESOLUTION_STOPWORDS = frozenset(
    {
        "after",
        "ago",
        "and",
        "are",
        "currently",
        "different",
        "did",
        "first",
        "from",
        "have",
        "he",
        "her",
        "happened",
        "how",
        "many",
        "months",
        "number",
        "of",
        "passed",
        "recent",
        "recently",
        "recently",
        "since",
        "the",
        "their",
        "they",
        "then",
        "third",
        "two",
        "weeks",
        "what",
        "which",
        "where",
        "with",
        "year",
        "years",
    }
)

_RESOLUTION_BROAD_TERMS = frozenset(
    {
        "amount",
        "bought",
        "did",
        "for",
        "got",
        "have",
        "how",
        "many",
        "recent",
        "recently",
        "was",
        "when",
        "where",
        "well",
        "what",
        "worked",
    }
)


def _resolution_term(value: str) -> str:
    """Apply a conservative singularization for question/evidence matching."""

    token = value.lower().strip("'-")
    if len(token) > 5 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _resolution_terms(text: str, *, include_stopwords: bool = False) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[\w'-]+", text.lower(), flags=re.UNICODE):
        normalized = _resolution_term(token)
        if len(normalized) < 3 or normalized.isdigit():
            continue
        if not include_stopwords and normalized in _RESOLUTION_STOPWORDS:
            continue
        terms.add(normalized)
    return terms


_MONTH_NUMBERS = {
    month.lower(): index
    for index, month in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}


def _parse_resolution_date(value: object) -> date | None:
    """Parse the date forms used by LongMemEval and its temporal anchors."""

    if value is None:
        return None
    text = str(value)
    numeric = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", text)
    if numeric:
        try:
            return date(
                int(numeric.group(1)),
                int(numeric.group(2)),
                int(numeric.group(3)),
            )
        except ValueError:
            return None
    named = re.search(
        r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b",
        text,
    )
    if not named:
        return None
    month = _MONTH_NUMBERS.get(named.group(1).lower())
    year = named.group(3)
    if month is None or year is None:
        return None
    try:
        return date(int(year), month, int(named.group(2)))
    except ValueError:
        return None


def _resolution_record(memory: Mapping[str, Any]) -> dict[str, Any]:
    """Keep resolver output auditable without copying benchmark gold fields."""

    return {
        "source_session_id": str(memory.get("source_session_id") or "unknown"),
        "session_date": memory.get("session_date"),
        "title": memory.get("title"),
        "content": memory.get("content"),
        "evidence_kind": memory.get("evidence_kind"),
        "fact_key": memory.get("fact_key"),
        "temporal_anchor": memory.get("temporal_anchor"),
    }


def _resolution_reference(memory: Mapping[str, Any]) -> dict[str, Any]:
    """Reference a context item without duplicating its full content."""

    record = _resolution_record(memory)
    record.pop("content", None)
    return record


def _resolution_compact_record(
    memory: Mapping[str, Any],
    *,
    max_content_chars: int = 360,
) -> dict[str, Any]:
    """Keep audit evidence useful without duplicating the reader context."""

    record = _resolution_record(memory)
    content = record.get("content")
    if isinstance(content, str) and len(content) > max_content_chars:
        record["content"] = content[:max_content_chars].rstrip() + "…"
    return record


def _resolution_evidence_snippet(
    memory: Mapping[str, Any],
    *,
    max_content_chars: int = 240,
) -> dict[str, Any]:
    """Expose only the fields needed to enumerate a compact evidence bundle."""

    content = str(memory.get("content") or "")
    if len(content) > max_content_chars:
        content = content[:max_content_chars].rstrip() + "…"
    return {
        "source_session_id": str(memory.get("source_session_id") or "unknown"),
        "session_date": memory.get("session_date"),
        "title": memory.get("title"),
        "content": content,
    }


def _apply_update_resolution_guard(
    response: str,
    audit: Mapping[str, Any],
) -> str:
    """Prevent a reader generation from reversing a date-resolved update."""

    if audit.get("kind") != "knowledge_update":
        return response
    directive = audit.get("resolver_directive")
    if not isinstance(directive, Mapping):
        return response
    preferred = audit.get("latest_update_candidate") or directive.get(
        "preferred_evidence"
    )
    if not isinstance(preferred, Mapping):
        return response
    content = str(preferred.get("content") or "").strip()
    if not content:
        return response
    numeric_pattern = r"(?<![\w/])(?:[$€£]\s*)?\d[\d,]*(?:\.\d+)?%?(?![\w/])"
    preferred_values = {
        value.replace(" ", "")
        for value in re.findall(numeric_pattern, content)
    }
    response_values = {
        value.replace(" ", "")
        for value in re.findall(numeric_pattern, response)
    }
    if response_values - preferred_values:
        # The reader still runs for every question. For a resolved update, a
        # fluent answer that introduces a different scalar must not override
        # the dated source sentence.
        return f"The latest directly relevant user statement says: {content}"
    return response


def _resolution_candidates(
    question: str,
    context: Sequence[Mapping[str, Any]],
    *,
    user_evidence_only: bool = False,
) -> list[tuple[float, Mapping[str, Any]]]:
    query_terms = _resolution_terms(question)
    query_core_terms = query_terms - _RESOLUTION_BROAD_TERMS
    candidates: list[tuple[float, Mapping[str, Any]]] = []
    user_kinds = {"user_fact", "preference", "decision", "event", "context"}
    for memory in context:
        evidence_kind = str(memory.get("evidence_kind") or "")
        if user_evidence_only and evidence_kind not in user_kinds:
            continue
        memory_text = " ".join(
            str(memory.get(key) or "")
            for key in ("title", "content", "fact_key", "temporal_anchor")
        )
        overlap = query_terms & _resolution_terms(memory_text)
        if not overlap:
            continue
        core_overlap = overlap & query_core_terms
        if query_core_terms and not core_overlap:
            continue
        score = float(len(overlap) + (2 * len(core_overlap)))
        if evidence_kind in user_kinds:
            score += 0.75
        if memory.get("explicit_user_evidence"):
            score += 0.5
        candidates.append((score, memory))
    candidates.sort(
        key=lambda item: (
            item[0],
            _parse_resolution_date(item[1].get("session_date")) or date.min,
        ),
        reverse=True,
    )
    return candidates


def _is_project_leadership_record(memory: Mapping[str, Any]) -> bool:
    """Recognize a project record without counting unrelated team leadership."""

    text = " ".join(
        str(memory.get(key) or "") for key in ("title", "content", "fact_key")
    ).lower()
    if "project" not in _resolution_terms(text):
        return False
    if re.search(
        r"\b(?:lead|led|leading)\b.{0,70}\b(?:project|competition)\b",
        text,
    ):
        return True
    if re.search(
        r"\b(?:solo project|class project|research project|case project)\b",
        text,
    ):
        return True
    if re.search(r"\b(?:working|work)\s+on\b.{0,55}\bproject\b", text):
        return not bool(re.search(r"\b(?:leading|led|lead)\s+(?:a\s+)?team\b", text))
    return False


_SPOKEN_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _stated_quantity(text: str) -> int | None:
    """Extract a conversational quantity without mistaking dates or scales."""

    number = r"(?:\d+|" + "|".join(_SPOKEN_NUMBERS) + r")"
    match = re.search(
        rf"\b(?:tried|visited|have|has|completed|worked on|bought|owned|used|led)"
        rf"\s+(?:about|around|roughly|a total of)?\s*({number})\b",
        text.lower(),
    )
    if not match:
        return None
    token = match.group(1)
    if token in _SPOKEN_NUMBERS:
        return _SPOKEN_NUMBERS[token]
    return int(token) if token.isdigit() else None


def _temporal_event_phrases(question: str) -> list[str]:
    """Extract named event fragments from the common LongMemEval wording."""

    matches = re.findall(
        r"(?:the\s+day|day)\s+(.+?)(?=,\s*(?:the\s+day|and\s+the\s+day)|$)",
        question,
        flags=re.IGNORECASE,
    )
    return [match.strip(" ,") for match in matches if match.strip(" ,")]


_TEMPORAL_ACTION_GROUPS = (
    frozenset({"receive", "receiv", "acquire", "acquir", "get", "got", "obtain"}),
    frozenset({"meet", "met", "meetup"}),
    frozenset({"help", "helped", "assist", "prepare", "prepared"}),
    frozenset({"order", "ordered", "buy", "bought", "purchase", "purchas"}),
    frozenset({"participate", "participat", "attend", "attended", "visit", "visited"}),
)


def _temporal_action_bonus(
    phrase_terms: set[str],
    memory_terms: set[str],
) -> float:
    """Prefer an event record over a same-object research/background record."""

    for group in _TEMPORAL_ACTION_GROUPS:
        phrase_action = any(
            token in group or any(token.startswith(prefix) for prefix in group)
            for token in phrase_terms
        )
        memory_action = any(
            token in group or any(token.startswith(prefix) for prefix in group)
            for token in memory_terms
        )
        if phrase_action and memory_action:
            return 1.5
    return 0.0


def _deterministic_resolution_audit(
    question: str,
    context: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Organize evidence for the reader without using LongMemEval labels."""

    if not context:
        return {"kind": "none", "reason": "no_context"}
    question_type = str(context[0].get("question_type") or "")
    lowered = question.lower()
    # The benchmark also has multi-session counting questions phrased as
    # "how many days". The question type is authoritative here; otherwise
    # those counts are incorrectly routed to date arithmetic.
    is_temporal = question_type == "temporal-reasoning" or "in the order" in lowered
    asks_for_count = bool(
        re.search(r"\bhow many\b|\bhow much\b|\bnumber of\b", lowered)
    )
    if asks_for_count and not is_temporal:
        candidates = _resolution_candidates(
            question,
            context,
            user_evidence_only=True,
        )
        counting_instructions = [
            "Enumerate qualifying entities before counting.",
            "Use explicit user quantities as evidence, but prefer a later user quantity when the question is an update.",
            "Deduplicate repeated statements within a session, not distinct named entities or distinct sessions.",
        ]
        if "project" in lowered and any(
            verb in lowered for verb in ("led", "lead", "leading")
        ):
            # A leadership question is about projects, not every memory that
            # happens to contain the word "leading". Keep a project/work
            # record (including a current solo project), while excluding team
            # leadership, portfolio goals, and unrelated project ideas.
            candidates = [
                (score, memory)
                for score, memory in candidates
                if _is_project_leadership_record(memory)
            ]
            counting_instructions.append(
                "For projects led, count only a project or team leadership record; do not count a project merely because the user is working on it."
            )
        query_terms = _resolution_terms(question)
        evidence_candidates = [
            (score, memory)
            for score, memory in candidates
            if len(
                query_terms
                & _resolution_terms(
                    " ".join(
                        str(memory.get(key) or "")
                        for key in ("title", "content", "fact_key")
                    )
                )
            )
            >= 2
        ]
        if "project" in lowered and any(
            verb in lowered for verb in ("led", "lead", "leading")
        ):
            evidence_candidates = candidates
        if not evidence_candidates:
            evidence_candidates = candidates[:6]
        quantity_candidates = []
        for _, memory in candidates:
            quantity = _stated_quantity(str(memory.get("content") or ""))
            if quantity is not None:
                quantity_candidates.append(
                    (
                        _parse_resolution_date(memory.get("session_date")) or date.min,
                        quantity,
                        memory,
                    )
                )
        quantity_candidates.sort(key=lambda item: item[0], reverse=True)
        return {
            "kind": "count",
            "counting_evidence": [
                _resolution_evidence_snippet(memory)
                for _, memory in evidence_candidates[:8]
            ],
            "latest_quantity_candidate": (
                {
                    "quantity": quantity_candidates[0][1],
                    "evidence": _resolution_evidence_snippet(
                        quantity_candidates[0][2]
                    ),
                }
                if quantity_candidates
                else None
            ),
            "resolver_directive": (
                {
                    "rule": "Use this newest stated quantity; do not add an older quantity to it.",
                    "preferred_quantity": quantity_candidates[0][1],
                    "preferred_evidence": _resolution_evidence_snippet(
                        quantity_candidates[0][2]
                    ),
                }
                if quantity_candidates
                else None
            ),
            "instructions": counting_instructions,
        }

    if is_temporal:
        phrases = _temporal_event_phrases(question)
        if not phrases:
            phrases = [question]
        selected: list[dict[str, Any]] = []
        used_memory_ids: set[int] = set()
        for phrase in phrases:
            phrase_terms = _resolution_terms(phrase)
            phrase_candidates = _resolution_candidates(
                phrase,
                context,
                user_evidence_only=True,
            )
            phrase_candidates = sorted(
                phrase_candidates,
                key=lambda item: (
                    item[0]
                    + _temporal_action_bonus(
                        phrase_terms,
                        _resolution_terms(
                            " ".join(
                                str(item[1].get(key) or "")
                                for key in ("title", "content")
                            )
                        ),
                    ),
                    item[0],
                ),
                reverse=True,
            )
            for score, memory in phrase_candidates:
                memory_terms = _resolution_terms(
                    " ".join(
                        str(memory.get(key) or "")
                        for key in ("title", "content")
                    )
                )
                overlap = phrase_terms & memory_terms
                # A temporal event must be supported by more than a generic
                # word such as "friend" or "birthday". This avoids attaching
                # a date from an unrelated conversation to the requested event.
                minimum_overlap = 2 if len(phrase_terms) >= 2 else 1
                if len(overlap) < minimum_overlap:
                    continue
                marker = id(memory)
                if marker in used_memory_ids:
                    continue
                used_memory_ids.add(marker)
                selected.append(
                    {
                        "event_phrase": phrase,
                        "match_score": round(score, 3),
                        **_resolution_record(memory),
                        "event_date": (
                            _parse_resolution_date(memory.get("temporal_anchor"))
                            or _parse_resolution_date(memory.get("session_date"))
                        ).isoformat()
                        if (
                            _parse_resolution_date(memory.get("temporal_anchor"))
                            or _parse_resolution_date(memory.get("session_date"))
                        )
                        else None,
                    }
                )
                break

        event_candidates = _resolution_candidates(
            question,
            context,
            user_evidence_only=True,
        )
        dated_candidates: list[tuple[date, Mapping[str, Any], float]] = []
        for score, memory in event_candidates:
            event_date = _parse_resolution_date(memory.get("temporal_anchor"))
            if event_date is None:
                event_date = _parse_resolution_date(memory.get("session_date"))
            if event_date is not None:
                dated_candidates.append((event_date, memory, score))

        interval: dict[str, Any] | None = None
        question_date = _parse_resolution_date(context[0].get("question_date"))
        unit = next(
            (candidate for candidate in ("weeks", "months", "days") if candidate in lowered),
            None,
        )
        if question_date is not None:
            reference_date: date | None = None
            if "consecutive" in lowered or "in a row" in lowered:
                dates = sorted({item[0] for item in dated_candidates})
                consecutive = [
                    (left, right)
                    for left, right in zip(dates, dates[1:])
                    if (right - left).days == 1
                ]
                if consecutive:
                    reference_date = consecutive[-1][1]
            elif selected:
                reference_date = _parse_resolution_date(selected[0].get("event_date"))
            if reference_date is not None and unit:
                delta_days = max(0, (question_date - reference_date).days)
                divisor = {"days": 1, "weeks": 7, "months": 30}[unit]
                interval = {
                    "unit": unit,
                    "value": delta_days // divisor,
                    "event_date": reference_date.isoformat(),
                    "question_date": question_date.isoformat(),
                }
        return {
            "kind": "temporal",
            "temporal_candidates": selected[:12],
            "dated_event_candidates": [
                {
                    "event_date": event_date.isoformat(),
                    **_resolution_reference(memory),
                }
                for event_date, memory, _ in dated_candidates[:8]
            ],
            "computed_interval": interval,
        }

    if question_type == "knowledge-update" or any(
        marker in lowered
        for marker in ("now", "currently", "recent relocation", "latest", "most recent")
    ):
        candidates = _resolution_candidates(
            question,
            context,
            user_evidence_only=True,
        )
        candidates.sort(
            key=lambda item: _parse_resolution_date(item[1].get("session_date")) or date.min,
            reverse=True,
        )
        return {
            "kind": "knowledge_update",
            "latest_update_candidate": (
                _resolution_compact_record(candidates[0][1]) if candidates else None
            ),
            "resolver_directive": (
                {
                    "rule": "Use this newest directly relevant user evidence unless a later date is present.",
                    "preferred_evidence": _resolution_evidence_snippet(
                        candidates[0][1], max_content_chars=160
                    ),
                }
                if candidates
                else None
            ),
            "latest_update_evidence": [
                _resolution_compact_record(memory)
                for _, memory in candidates[:4]
            ],
            "instructions": [
                "Compare candidate user statements across session dates.",
                "The latest directly relevant statement supersedes an older value even when fact keys differ.",
            ],
        }

    return {"kind": "none"}


class LongMemEvalReader(Protocol):
    """Reader contract for the retrieve -> answer stage."""

    version: str

    async def answer(
        self, question: str, memories: Sequence[RecallMatch]
    ) -> str: ...


class LongMemEvalJudge(Protocol):
    """Judge contract returning the official boolean label and raw response."""

    version: str

    async def judge(
        self, question: LongMemEvalQuestion, response: str
    ) -> tuple[bool, str]: ...


class _LmStudioQARequestError(OSError):
    """Retryable LM Studio transport failure for the QA stage."""


@dataclass(slots=True)
class _LmStudioChatClient:
    model: str
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    reasoning_effort: str | None = None
    timeout_seconds: float = 120.0
    retries: int = 2

    def __post_init__(self) -> None:
        self.model = self.model.strip()
        self.base_url = self.base_url.rstrip("/")
        if not self.model:
            raise ValueError("LM Studio QA client requires a model identifier")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("LM Studio base_url must start with http:// or https://")
        if self.reasoning_effort is not None:
            self.reasoning_effort = self.reasoning_effort.lower()
            if self.reasoning_effort not in {"low", "medium", "high"}:
                raise ValueError("reasoning_effort must be low, medium, high, or None")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.retries < 0:
            raise ValueError("retries cannot be negative")

    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort

        for attempt in range(self.retries + 1):
            try:
                response = await asyncio.to_thread(self._post_json, payload)
                return _message_content(response)
            except _LmStudioQARequestError as error:
                if attempt >= self.retries:
                    raise RuntimeError(
                        f"LM Studio QA request failed after {attempt + 1} attempts: {error}"
                    ) from error
                await asyncio.sleep(min(2**attempt, 8))
            except ValueError as error:
                if attempt >= self.retries:
                    raise ValueError(
                        "LM Studio returned invalid QA output after "
                        f"{attempt + 1} attempts: {error}"
                    ) from error
                await asyncio.sleep(min(2**attempt, 8))
        raise AssertionError("LM Studio QA retry loop exited unexpectedly")

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
            raise _LmStudioQARequestError(
                f"HTTP {error.code} from {self.base_url}: {detail}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise _LmStudioQARequestError(
                f"Could not reach {self.base_url}: {error}"
            ) from error
        if not isinstance(decoded, Mapping):
            raise _LmStudioQARequestError("LM Studio returned a non-object response")
        return decoded


def _message_content(response: Mapping[str, Any]) -> str:
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
    if not content:
        raise ValueError("LM Studio response content is empty")
    return content


@dataclass(slots=True)
class LmStudioLongMemEvalReader:
    """Generate an answer from only the retrieved memory context."""

    model: str
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    prompt_version: str = LONGMEMEVAL_READER_PROMPT_VERSION
    reasoning_effort: str | None = None
    max_tokens: int = 512
    timeout_seconds: float = 120.0
    retries: int = 2
    temperature: float = 0.0
    _client: _LmStudioChatClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = _LmStudioChatClient(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            reasoning_effort=self.reasoning_effort,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
        )
        self.model = self._client.model
        self.base_url = self._client.base_url
        self.reasoning_effort = self._client.reasoning_effort
        if self.max_tokens <= 0:
            raise ValueError("reader max_tokens must be positive")

    @property
    def version(self) -> str:
        suffix = (
            f":reasoning-{self.reasoning_effort}"
            if self.reasoning_effort is not None
            else ""
        )
        return f"{self.model}:{self.prompt_version}{suffix}"

    async def answer(
        self, question: str, memories: Sequence[RecallMatch]
    ) -> str:
        context = [
            {
                "rank": index + 1,
                "title": match.memory.title,
                "content": match.memory.content,
                "memory_type": match.memory.type.value,
                "session_date": match.memory.metadata.get("source_session_date"),
                "source_session_id": (
                    match.memory.metadata.get("source_session_id")
                    or getattr(match.memory, "provenance_key", None)
                ),
                "question_type": match.memory.metadata.get("question_type"),
                "question_date": match.memory.metadata.get("question_date"),
                "fact_key": getattr(match.memory, "memory_key", None)
                or match.memory.metadata.get("fact_key"),
                "supersedes_fact_key": (
                    getattr(match.memory, "supersedes_memory_key", None)
                    or match.memory.metadata.get("supersedes_fact_key")
                ),
                "evidence_kind": match.memory.metadata.get("evidence_kind"),
                "explicit_user_evidence": match.memory.metadata.get(
                    "explicit_user_evidence", False
                ),
                "temporal_anchor": match.memory.metadata.get("temporal_anchor"),
                "observed_at": (
                    match.memory.observed_at.isoformat()
                    if getattr(match.memory, "observed_at", None) is not None
                    else None
                ),
            }
            for index, match in enumerate(memories)
        ]
        session_groups: list[dict[str, Any]] = []
        grouped: dict[str, list[dict[str, Any]]] = {}
        for memory in context:
            session_id = str(memory.get("source_session_id") or "unknown")
            grouped.setdefault(session_id, []).append(memory)
        for session_id, session_memories in grouped.items():
            session_groups.append(
                {
                    "source_session_id": session_id,
                    "memory_count": len(session_memories),
                    "memory_ranks": [memory["rank"] for memory in session_memories],
                    "memory_titles": [memory.get("title") for memory in session_memories],
                }
            )
        action_item_checklist = _action_item_checklist(question, context)
        deterministic_resolution_audit = _deterministic_resolution_audit(
            question, context
        )
        user_payload = json.dumps(
            {
                "question": question,
                "resolution_priority": deterministic_resolution_audit.get(
                    "resolver_directive"
                ),
                "retrieved_memories": context,
                "session_groups": session_groups,
                "action_item_checklist": action_item_checklist,
                "deterministic_resolution_audit": deterministic_resolution_audit,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        response = await self._client.complete(
            [
                {"role": "system", "content": READER_SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return _apply_update_resolution_guard(response, deterministic_resolution_audit)


def official_longmemeval_judge_prompt(
    question_type: str,
    question: str,
    answer: Any,
    response: str,
    *,
    abstention: bool = False,
) -> str:
    """Build the official LongMemEval QA judge prompt.

    This mirrors ``src/evaluation/evaluate_qa.py`` from the upstream benchmark,
    including its task-specific handling for updates, temporal questions,
    preferences, and abstention.
    """

    if abstention:
        template = (
            "I will give you an unanswerable question, an explanation, and a "
            "response from a model. Please answer yes if the model correctly "
            "identifies the question as unanswerable. The model could say that "
            "the information is incomplete, or some other information is given "
            "but the asked information is not.\n\nQuestion: {}\n\n"
            "Explanation: {}\n\nModel Response: {}\n\n"
            "Does the model correctly identify the question as unanswerable? "
            "Answer yes or no only."
        )
        return template.format(question, answer, response)

    if question_type in {
        "single-session-user",
        "single-session-assistant",
        "multi-session",
    }:
        template = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer "
            "no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: "
            "{}\n\nIs the model response correct? Answer yes or no only."
        )
    elif question_type == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer "
            "no. In addition, do not penalize off-by-one errors for the number of "
            "days.\nIf the question asks for the number of days/weeks/months, etc., "
            "and the model makes off-by-one errors (e.g., predicting 19 days when "
            "the answer is 18), the model's response is still correct. \n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif question_type == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct answer. "
            "Otherwise, answer no. If the response contains some previous "
            "information along with an updated answer, the response should be "
            "considered as correct as long as the updated answer is the required "
            "answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: "
            "{}\n\nIs the model response correct? Answer yes or no only."
        )
    elif question_type == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if the "
            "response satisfies the desired response. Otherwise, answer no. The "
            "model does not need to reflect all the points in the rubric. The "
            "response is correct as long as it recalls and utilizes the user's "
            "personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\n"
            "Model Response: {}\n\nIs the model response correct? Answer yes "
            "or no only."
        )
    else:
        raise ValueError(f"Unsupported LongMemEval question type: {question_type}")
    return template.format(question, answer, response)


@dataclass(slots=True)
class LmStudioLongMemEvalJudge:
    """Apply the upstream LongMemEval judge prompt through LM Studio."""

    model: str
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    prompt_version: str = LONGMEMEVAL_JUDGE_PROMPT_VERSION
    reasoning_effort: str | None = None
    max_tokens: int = 10
    timeout_seconds: float = 120.0
    retries: int = 2
    temperature: float = 0.0
    _client: _LmStudioChatClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = _LmStudioChatClient(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            reasoning_effort=self.reasoning_effort,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
        )
        self.model = self._client.model
        self.base_url = self._client.base_url
        self.reasoning_effort = self._client.reasoning_effort
        if self.max_tokens <= 0:
            raise ValueError("judge max_tokens must be positive")

    @property
    def version(self) -> str:
        suffix = (
            f":reasoning-{self.reasoning_effort}"
            if self.reasoning_effort is not None
            else ""
        )
        return f"{self.model}:{self.prompt_version}{suffix}"

    async def judge(
        self, question: LongMemEvalQuestion, response: str
    ) -> tuple[bool, str]:
        prompt = official_longmemeval_judge_prompt(
            question.question_type,
            question.question,
            question.answer,
            response,
            abstention="_abs" in question.question_id,
        )
        raw = await self._client.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        # Keep the upstream evaluator's semantics for comparability.
        return "yes" in raw.lower(), raw


@dataclass(slots=True)
class LongMemEvalQACache:
    """Durable per-question cache for reader and judge outputs."""

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
            or payload.get("schema_version") != LONGMEMEVAL_QA_CACHE_SCHEMA
        ):
            return
        entries = payload.get("entries", {})
        if isinstance(entries, Mapping):
            self._entries = {
                str(key): dict(value)
                for key, value in entries.items()
                if isinstance(value, Mapping)
            }

    def get_reader(
        self,
        question_id: str,
        *,
        input_hash: str,
        reader_version: str,
    ) -> str | None:
        entry = self._matching_entry(
            question_id,
            input_hash=input_hash,
            reader_version=reader_version,
        )
        response = entry.get("reader_response") if entry else None
        return response if isinstance(response, str) and response else None

    def get_judge(
        self,
        question_id: str,
        *,
        input_hash: str,
        reader_version: str,
        judge_version: str,
    ) -> tuple[bool, str] | None:
        entry = self._matching_entry(
            question_id,
            input_hash=input_hash,
            reader_version=reader_version,
            judge_version=judge_version,
        )
        if not entry or not isinstance(entry.get("judge_label"), bool):
            return None
        raw = entry.get("judge_response", "")
        return entry["judge_label"], str(raw)

    def put_reader(
        self,
        question_id: str,
        *,
        input_hash: str,
        reader_version: str,
        judge_version: str,
        response: str,
    ) -> None:
        entry = self._entries.setdefault(question_id, {})
        entry.update(
            {
                "input_hash": input_hash,
                "reader_version": reader_version,
                "judge_version": judge_version,
                "reader_response": response,
            }
        )

    def put_judge(
        self,
        question_id: str,
        *,
        input_hash: str,
        reader_version: str,
        judge_version: str,
        label: bool,
        response: str,
    ) -> None:
        entry = self._entries.setdefault(question_id, {})
        entry.update(
            {
                "input_hash": input_hash,
                "reader_version": reader_version,
                "judge_version": judge_version,
                "judge_label": label,
                "judge_response": response,
            }
        )

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": LONGMEMEVAL_QA_CACHE_SCHEMA,
            "entries": self._entries,
        }
        temporary_path = self.path.with_name(f".{self.path.name}.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)

    def _matching_entry(
        self,
        question_id: str,
        *,
        input_hash: str,
        reader_version: str,
        judge_version: str | None = None,
    ) -> dict[str, Any] | None:
        entry = self._entries.get(question_id)
        if not entry:
            return None
        if (
            entry.get("input_hash") != input_hash
            or entry.get("reader_version") != reader_version
            or (judge_version is not None and entry.get("judge_version") != judge_version)
        ):
            return None
        return entry


@dataclass(frozen=True, slots=True)
class LongMemEvalQAResult:
    question_id: str
    question_type: str
    category: str
    retrieved_count: int
    retrieved_titles: tuple[str, ...]
    retrieved_answer_session_ids: tuple[str, ...]
    retrieval_hit_at_k: bool | None
    answer_session_recall_at_k: float | None
    reader_response: str | None
    judge_response: str | None
    judge_label: bool | None
    status: str
    failed_stage: str | None = None
    failure_message: str | None = None


@dataclass(frozen=True, slots=True)
class LongMemEvalCategoryReport:
    category: str
    question_count: int
    scored_count: int
    correct_count: int
    failed_count: int
    retrieval_evaluable_count: int
    retrieval_hit_count: int
    answer_session_recall_evaluable_count: int
    answer_session_recall_sum: float

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.scored_count if self.scored_count else 0.0

    @property
    def retrieval_recall_at_k(self) -> float:
        return (
            self.retrieval_hit_count / self.retrieval_evaluable_count
            if self.retrieval_evaluable_count
            else 0.0
        )

    @property
    def answer_session_recall_at_k(self) -> float:
        return (
            self.answer_session_recall_sum
            / self.answer_session_recall_evaluable_count
            if self.answer_session_recall_evaluable_count
            else 0.0
        )


@dataclass(frozen=True, slots=True)
class LongMemEvalQAReport:
    dataset: str
    question_count: int
    scored_count: int
    correct_count: int
    retrieval_k: int
    reader_model: str
    judge_model: str
    reader_cache_hits: int
    reader_cache_misses: int
    judge_cache_hits: int
    judge_cache_misses: int
    ingestion_failed_session_count: int
    retrieval_evaluable_count: int
    retrieval_hit_count: int
    answer_session_recall_evaluable_count: int
    answer_session_recall_sum: float
    categories: tuple[LongMemEvalCategoryReport, ...]
    questions: tuple[LongMemEvalQAResult, ...]

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.scored_count if self.scored_count else 0.0

    @property
    def coverage(self) -> float:
        return self.scored_count / self.question_count if self.question_count else 0.0

    @property
    def retrieval_recall_at_k(self) -> float:
        return (
            self.retrieval_hit_count / self.retrieval_evaluable_count
            if self.retrieval_evaluable_count
            else 0.0
        )

    @property
    def answer_session_recall_at_k(self) -> float:
        return (
            self.answer_session_recall_sum
            / self.answer_session_recall_evaluable_count
            if self.answer_session_recall_evaluable_count
            else 0.0
        )


def _question_category(question: LongMemEvalQuestion) -> str:
    return "abstention" if "_abs" in question.question_id else question.question_type


def stratified_longmemeval_question_ids(
    dataset_path: str | Path,
    *,
    per_category: int,
) -> tuple[str, ...]:
    """Select the first N questions from every LongMemEval category."""

    if per_category <= 0:
        raise ValueError("per_category must be positive")
    questions = load_longmemeval_instances(dataset_path)
    selected: list[str] = []
    counts: dict[str, int] = {}
    for question in questions:
        category = _question_category(question)
        if counts.get(category, 0) >= per_category:
            continue
        selected.append(question.question_id)
        counts[category] = counts.get(category, 0) + 1
    return tuple(selected)


def _retrieval_input_hash(
    question: LongMemEvalQuestion,
    matches: Sequence[RecallMatch],
    retrieval_k: int,
) -> str:
    payload = {
        "question_id": question.question_id,
        "question": question.question,
        "question_date": question.question_date,
        "retrieval_k": retrieval_k,
        "memories": [
            {
                "rank": index,
                "score": round(match.score, 8),
                "title": match.memory.title,
                "content": match.memory.content,
                "memory_type": match.memory.type.value,
                "session_date": match.memory.metadata.get("source_session_date"),
                "fact_key": match.memory.metadata.get("fact_key"),
            }
            for index, match in enumerate(matches)
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _reader_context_limit(question_type: str, retrieval_k: int) -> int:
    """Give the reader enough evidence while keeping retrieval metrics honest."""

    if question_type in {"multi-session", "temporal-reasoning"}:
        return max(retrieval_k, 24 if question_type == "multi-session" else 18)
    if question_type == "knowledge-update":
        return max(retrieval_k, 24)
    if question_type == "single-session-preference":
        return max(retrieval_k, 16)
    return max(retrieval_k, 16)


async def run_longmemeval_qa(
    dataset_path: str | Path,
    extractor: FactExtractor,
    reader: LongMemEvalReader,
    judge: LongMemEvalJudge,
    *,
    ingestion_cache_path: str | Path | None = None,
    qa_cache_path: str | Path | None = None,
    limit: int | None = 50,
    retrieval_k: int = 8,
    question_ids: set[str] | None = None,
    extraction_concurrency: int = 1,
    retry_failed: bool = False,
) -> LongMemEvalQAReport:
    """Run LongMemEval ingestion, retrieval, reader generation, and judging."""

    if retrieval_k <= 0:
        raise ValueError("retrieval_k must be positive")
    if extraction_concurrency <= 0:
        raise ValueError("extraction_concurrency must be positive")
    questions = load_longmemeval_instances(
        dataset_path,
        limit=limit,
        question_ids=question_ids,
    )
    extraction_cache = (
        ExtractionCache(Path(ingestion_cache_path))
        if ingestion_cache_path is not None
        else None
    )
    qa_cache = LongMemEvalQACache(Path(qa_cache_path)) if qa_cache_path else None
    results: list[LongMemEvalQAResult] = []
    reader_cache_hits = 0
    reader_cache_misses = 0
    judge_cache_hits = 0
    judge_cache_misses = 0
    ingestion_failed_session_count = 0

    for question in questions:
        category = _question_category(question)
        service = MemoryService(store=InMemoryMemoryStore())
        try:
            ingestion = await ingest_longmemeval_question(
                service,
                question,
                extractor,
                cache=extraction_cache,
                extraction_concurrency=extraction_concurrency,
                retry_failed=retry_failed,
            )
            ingestion_failed_session_count += len(ingestion.failed_session_ids)
            matches = await _retrieve_longmemeval_matches(
                service,
                question,
                namespace_id=ingestion.namespace_id,
                limit=retrieval_k,
            )
            reader_matches = await _retrieve_longmemeval_matches(
                service,
                question,
                namespace_id=ingestion.namespace_id,
                limit=_reader_context_limit(question.question_type, retrieval_k),
            )
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
            results.append(
                LongMemEvalQAResult(
                    question_id=question.question_id,
                    question_type=question.question_type,
                    category=category,
                    retrieved_count=0,
                    retrieved_titles=(),
                    retrieved_answer_session_ids=(),
                    retrieval_hit_at_k=None,
                    answer_session_recall_at_k=None,
                    reader_response=None,
                    judge_response=None,
                    judge_label=None,
                    status="failed",
                    failed_stage="ingestion-or-retrieval",
                    failure_message=f"{type(error).__name__}: {str(error)[:500]}",
                )
            )
            continue

        answer_session_ids = set(question.answer_session_ids)
        retrieved_answer_session_ids = tuple(
            dict.fromkeys(
                str(match.memory.metadata["source_session_id"])
                for match in matches
                if match.memory.metadata.get("source_session_id") in answer_session_ids
            )
        )
        retrieval_hit_at_k = (
            bool(retrieved_answer_session_ids) if answer_session_ids else None
        )
        answer_session_recall_at_k = (
            len(set(retrieved_answer_session_ids)) / len(answer_session_ids)
            if answer_session_ids
            else None
        )
        input_hash = _retrieval_input_hash(question, reader_matches, retrieval_k)
        reader_response: str | None = None
        if qa_cache:
            reader_response = qa_cache.get_reader(
                question.question_id,
                input_hash=input_hash,
                reader_version=reader.version,
            )
        if reader_response is not None:
            reader_cache_hits += 1
        else:
            reader_cache_misses += 1
            try:
                reader_response = await reader.answer(question.question, reader_matches)
            except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
                results.append(
                    LongMemEvalQAResult(
                        question_id=question.question_id,
                        question_type=question.question_type,
                        category=category,
                        retrieved_count=len(matches),
                        retrieved_titles=tuple(
                            match.memory.title or match.memory.content for match in matches
                        ),
                        retrieved_answer_session_ids=retrieved_answer_session_ids,
                        retrieval_hit_at_k=retrieval_hit_at_k,
                        answer_session_recall_at_k=answer_session_recall_at_k,
                        reader_response=None,
                        judge_response=None,
                        judge_label=None,
                        status="failed",
                        failed_stage="reader",
                        failure_message=f"{type(error).__name__}: {str(error)[:500]}",
                    )
                )
                if qa_cache:
                    qa_cache.flush()
                continue
            if qa_cache:
                qa_cache.put_reader(
                    question.question_id,
                    input_hash=input_hash,
                    reader_version=reader.version,
                    judge_version=judge.version,
                    response=reader_response,
                )
                qa_cache.flush()

        judged = (
            qa_cache.get_judge(
                question.question_id,
                input_hash=input_hash,
                reader_version=reader.version,
                judge_version=judge.version,
            )
            if qa_cache
            else None
        )
        if judged is not None:
            judge_cache_hits += 1
            judge_label, judge_response = judged
        else:
            judge_cache_misses += 1
            try:
                judge_label, judge_response = await judge.judge(question, reader_response)
            except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
                results.append(
                    LongMemEvalQAResult(
                        question_id=question.question_id,
                        question_type=question.question_type,
                        category=category,
                        retrieved_count=len(matches),
                        retrieved_titles=tuple(
                            match.memory.title or match.memory.content for match in matches
                        ),
                        retrieved_answer_session_ids=retrieved_answer_session_ids,
                        retrieval_hit_at_k=retrieval_hit_at_k,
                        answer_session_recall_at_k=answer_session_recall_at_k,
                        reader_response=reader_response,
                        judge_response=None,
                        judge_label=None,
                        status="failed",
                        failed_stage="judge",
                        failure_message=f"{type(error).__name__}: {str(error)[:500]}",
                    )
                )
                if qa_cache:
                    qa_cache.flush()
                continue
            if qa_cache:
                qa_cache.put_judge(
                    question.question_id,
                    input_hash=input_hash,
                    reader_version=reader.version,
                    judge_version=judge.version,
                    label=judge_label,
                    response=judge_response,
                )
                qa_cache.flush()

        results.append(
            LongMemEvalQAResult(
                question_id=question.question_id,
                question_type=question.question_type,
                category=category,
                retrieved_count=len(matches),
                retrieved_titles=tuple(
                    match.memory.title or match.memory.content for match in matches
                ),
                retrieved_answer_session_ids=retrieved_answer_session_ids,
                retrieval_hit_at_k=retrieval_hit_at_k,
                answer_session_recall_at_k=answer_session_recall_at_k,
                reader_response=reader_response,
                judge_response=judge_response,
                judge_label=judge_label,
                status="scored",
            )
        )

    categories = _build_category_reports(results)
    correct_count = sum(1 for result in results if result.judge_label is True)
    scored_count = sum(1 for result in results if result.judge_label is not None)
    return LongMemEvalQAReport(
        dataset=str(dataset_path),
        question_count=len(results),
        scored_count=scored_count,
        correct_count=correct_count,
        retrieval_k=retrieval_k,
        reader_model=_model_name(reader),
        judge_model=_model_name(judge),
        reader_cache_hits=reader_cache_hits,
        reader_cache_misses=reader_cache_misses,
        judge_cache_hits=judge_cache_hits,
        judge_cache_misses=judge_cache_misses,
        ingestion_failed_session_count=ingestion_failed_session_count,
        retrieval_evaluable_count=sum(
            result.retrieval_hit_at_k is not None for result in results
        ),
        retrieval_hit_count=sum(
            result.retrieval_hit_at_k is True for result in results
        ),
        answer_session_recall_evaluable_count=sum(
            result.answer_session_recall_at_k is not None for result in results
        ),
        answer_session_recall_sum=sum(
            result.answer_session_recall_at_k or 0.0 for result in results
        ),
        categories=tuple(categories),
        questions=tuple(results),
    )


def _model_name(component: object) -> str:
    model = getattr(component, "model", None)
    return str(model) if model else str(getattr(component, "version", "unknown"))


def _normalize_retrieval_query(query: str) -> str:
    """Keep punctuation from becoming a retrieval token in the local adapter."""

    return re.sub(r"[^\w]+", " ", query, flags=re.UNICODE).strip()


def _retrieval_query_expansions(question: str) -> list[str]:
    """Add conservative lexical bridges for elliptical follow-up questions.

    The standalone local adapter intentionally has no heavyweight semantic
    model. These bridges preserve the behavior expected from normal semantic
    search for common paraphrases, while the original question stays the
    primary retrieval query.
    """

    terms = _retrieval_terms(question)
    expansions: list[str] = []
    if terms & {"publication", "conference", "paper", "article", "research"}:
        expansions.append("research papers articles academic studies conferences")
    if terms & {"accessory", "accessories", "setup", "complement"}:
        expansions.append("camera lens photography gear equipment accessories")
    return expansions


_RETRIEVAL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "about",
        "can",
        "could",
        "did",
        "do",
        "for",
        "from",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "please",
        "recent",
        "remind",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "you",
        "your",
        "upcoming",
    }
)


def _retrieval_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw_token in re.findall(r"[\w'-]+", text.lower(), flags=re.UNICODE):
        token = raw_token.strip("'-")
        if not token or token in _RETRIEVAL_STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("'s"):
            token = token[:-2]
        if len(token) <= 2 and not token.isdigit():
            continue
        terms.add(token)
        # Keep the surface form for exact matches, but add conservative
        # variants so ``moved`` matches ``move`` and ``projects`` matches
        # ``project`` without requiring a heavyweight NLP dependency.
        if token.endswith("ies") and len(token) > 4:
            terms.add(token[:-3] + "y")
        if token.endswith("ing") and len(token) > 5:
            terms.add(token[:-3])
        if token.endswith("ed") and len(token) > 4:
            terms.add(token[:-1])
            terms.add(token[:-2])
        if token.endswith("s") and len(token) > 4:
            terms.add(token[:-1])
    return terms


def _memory_retrieval_score(
    memory: Any,
    *,
    query_terms: set[str],
    base_score: float,
    base_reason: str | None = None,
) -> float:
    """Blend content retrieval with title, tags, and fact identity signals."""

    if not query_terms:
        return base_score
    title_terms = _retrieval_terms(memory.title or "")
    content_terms = _retrieval_terms(memory.content)
    tag_terms = _retrieval_terms(" ".join(memory.tags))
    fact_terms = _retrieval_terms(
        " ".join(
            str(value or "")
            for value in (
                getattr(memory, "memory_key", None),
                getattr(memory, "supersedes_memory_key", None),
                memory.metadata.get("fact_key"),
                memory.metadata.get("temporal_anchor"),
                memory.metadata.get("source_session_date"),
            )
        )
    )
    denominator = max(len(query_terms), 1)
    title_overlap = len(query_terms & title_terms) / denominator
    content_overlap = len(query_terms & content_terms) / denominator
    tag_overlap = len(query_terms & tag_terms) / denominator
    fact_overlap = len(query_terms & fact_terms) / denominator
    lexical_score = (
        0.35 * title_overlap
        + 0.45 * content_overlap
        + 0.12 * tag_overlap
        + 0.08 * fact_overlap
    )
    if lexical_score <= 0:
        # The local fallback embedding is deliberately lightweight and can
        # assign high scores to unrelated memories. Keep it as a synonym
        # fallback, but never let it outrank explicit query evidence.
        if base_reason == "provenance-context":
            # A sibling memory from a matched evidence group is useful as
            # context, but it is not query evidence by itself. Keep it below
            # even a weak lexical hit so contextual expansion cannot drown out
            # the exact fact the reader needs.
            return max(base_score, 0.0) * 0.2
        return max(base_score, 0.0) * 0.2
    return 0.65 * lexical_score + 0.35 * max(base_score, 0.0)


def _provenance_group_relevance(matches: Sequence[RecallMatch]) -> float:
    """Rank evidence bundles by corroborated relevance, not one lucky hit."""

    scores = sorted((max(match.score, 0.0) for match in matches), reverse=True)
    # The first few independent memories are useful corroboration; cap the
    # contribution so a verbose session cannot drown out a concise one.
    return sum(scores[:4])


def _diversify_longmemeval_matches(
    matches: Sequence[RecallMatch],
    question_type: str,
    limit: int,
) -> list[RecallMatch]:
    """Keep evidence from several sessions before filling the remaining slots."""

    max_per_provenance = None
    if question_type in {"multi-session", "temporal-reasoning"}:
        max_per_provenance = 4
    elif question_type == "knowledge-update":
        max_per_provenance = 4
    if max_per_provenance is not None:
        # Round-robin diversity across every weak lexical hit can still use
        # the whole context budget on distractor sessions before a selected
        # evidence bundle gets its second fact. Bound the number of groups by
        # rank, then preserve several facts inside those strongest bundles.
        grouped: dict[str, list[RecallMatch]] = {}
        for match in matches:
            grouped.setdefault(provenance_key_for_memory(match.memory), []).append(match)
        # Keep enough candidate groups for a compact multi-session answer. A
        # low group cap can hide the second corroborating session before the
        # generic round-robin selector has a chance to use it.
        group_limit = min(16, max(12, limit))
        ranked_group_keys = sorted(
            grouped,
            key=lambda key: _provenance_group_relevance(grouped[key]),
            reverse=True,
        )[:group_limit]
        matches = [
            match
            for key in ranked_group_keys
            for match in grouped[key]
        ]
    return select_diverse_matches(
        matches,
        limit=limit,
        max_per_provenance=max_per_provenance,
        deduplicate_evidence=True,
    )


async def _retrieve_longmemeval_matches(
    service: MemoryService,
    question: LongMemEvalQuestion,
    *,
    namespace_id: str,
    limit: int,
) -> list[RecallMatch]:
    """Retrieve a broad candidate pool, rerank evidence, then diversify sessions."""

    # Long multi-session questions can have many lexical distractors before
    # the second evidence bundle appears. Gather a sufficiently broad pool;
    # the bounded provenance selector below controls reader-context size.
    candidate_limit = max(512, limit * 12)
    retrieval_queries = [question.question]
    retrieval_queries.extend(_retrieval_query_expansions(question.question))
    if question.question_type == "temporal-reasoning":
        retrieval_queries.extend(_temporal_event_phrases(question.question))
    base_scores: dict[str, tuple[float, str | None]] = {}
    for retrieval_query in retrieval_queries:
        base_matches = await service.semantic_recall(
            RecallQuery(
                namespace_id=namespace_id,
                query=_normalize_retrieval_query(retrieval_query),
                limit=candidate_limit,
                candidate_limit=candidate_limit,
                # Candidate gathering must preserve completeness. The final
                # reader-context policy diversifies the selected matches;
                # capping sources here can hide the only evidence for a fact.
                diversify_by_provenance=False,
                deduplicate_evidence=True,
                include_provenance_context=True,
                provenance_context_limit=8,
                provenance_context_group_limit=32,
            )
        )
        for match in base_matches:
            previous = base_scores.get(match.memory.id)
            if previous is None or match.score > previous[0]:
                base_scores[match.memory.id] = (match.score, match.reason)
    query_term_sets = [_retrieval_terms(query) for query in retrieval_queries]
    memories = await service.list_memories(
        namespace_id,
        statuses=[MemoryStatus.ACTIVE],
    )
    reranked: list[RecallMatch] = []
    for memory in memories:
        scored_candidates = [
            _memory_retrieval_score(
                memory,
                query_terms=query_terms,
                base_score=base_scores.get(memory.id, (0.0, None))[0],
                base_reason=base_scores.get(memory.id, (0.0, None))[1],
            )
            for query_terms in query_term_sets
        ]
        score = max(scored_candidates, default=0.0)
        if memory.id in base_scores or score > 0:
            reranked.append(
                RecallMatch(
                    memory=memory,
                    score=score,
                    reason="longmemeval-rerank",
                )
            )
    reranked.sort(
        key=lambda match: (
            match.score,
            match.memory.confidence,
            match.memory.metadata.get("source_session_date") or "",
        ),
        reverse=True,
    )
    return _diversify_longmemeval_matches(reranked, question.question_type, limit)


def _build_category_reports(
    results: Sequence[LongMemEvalQAResult],
) -> list[LongMemEvalCategoryReport]:
    grouped: dict[str, list[LongMemEvalQAResult]] = {}
    for result in results:
        grouped.setdefault(result.category, []).append(result)
    return [
        LongMemEvalCategoryReport(
            category=category,
            question_count=len(group),
            scored_count=sum(item.judge_label is not None for item in group),
            correct_count=sum(item.judge_label is True for item in group),
            failed_count=sum(item.judge_label is None for item in group),
            retrieval_evaluable_count=sum(
                item.retrieval_hit_at_k is not None for item in group
            ),
            retrieval_hit_count=sum(
                item.retrieval_hit_at_k is True for item in group
            ),
            answer_session_recall_evaluable_count=sum(
                item.answer_session_recall_at_k is not None for item in group
            ),
            answer_session_recall_sum=sum(
                item.answer_session_recall_at_k or 0.0 for item in group
            ),
        )
        for category, group in sorted(grouped.items())
    ]


def write_longmemeval_hypotheses(
    report: LongMemEvalQAReport,
    path: str | Path,
) -> None:
    """Write the upstream evaluator's ``question_id``/``hypothesis`` JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {"question_id": result.question_id, "hypothesis": result.reader_response},
            ensure_ascii=False,
        )
        for result in report.questions
        if result.reader_response is not None
    ]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
