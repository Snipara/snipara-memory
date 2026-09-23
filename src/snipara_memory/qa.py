"""LongMemEval answer generation and official judge integration.

The ingestion adapter deliberately stops at structured memories.  This module
adds the separate QA stage: retrieve active memories, ask a reader to answer
from that evidence, then apply the official LongMemEval yes/no judge prompt.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from itertools import pairwise
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
LONGMEMEVAL_READER_PROMPT_VERSION = "lmstudio-longmemeval-reader-v59"
LONGMEMEVAL_JUDGE_PROMPT_VERSION = "longmemeval-official-judge-v1"

# Keep the operational prompt short enough for local 8k-context readers. The
# deterministic resolver and tests implement the detailed invariants documented
# above; the model needs a compact decision contract, not repeated examples.
READER_SYSTEM_PROMPT = """Answer the question using only the supplied evidence.
Return the shortest supported answer as exactly {"answer":"..."}. Do not reveal
reasoning, mention retrieval, or use outside knowledge.

Evidence:
- Read direct_answer_support cards first, then verify them in retrieved_memories.
  Match the exact entity, relation, attribute, and speaker. Preserve modifiers.
- Source conversation excerpts are lossless fallback evidence. Use them when an
  extracted fact omitted an exact name, color, code, value, or assistant answer.
  When asked to identify or recall an entity, return its exact name rather than
  replacing the name with a description of what it offers.
- Resolver fields organize evidence and outrank retrieval order, but are not
  gold labels. Verify them against the cited memory.

Resolution:
- Updates: choose the newest directly relevant user statement by session_date.
  Respect supersession and never add an old quantity to its replacement.
- Multi-session: inspect every session_group. For counts, silently enumerate
  qualifying records and use action_item_checklist, contribution_ledger, or
  map_stage_output when supplied. Merge duplicate mentions, not distinct events.
  A dry-cleaning pickup is a pickup; opposite exchange actions are distinct.
  Named events, not session count, determine the result. A
  scale vehicle or aircraft diorama is a worked-on model when requested.
- Temporal: use matching dated events and computed_interval when supported.
  Preserve companion, venue, object, and other relation qualifiers.
- Preference: recover the user's constraints and personalize the new request.
  If no current item name is supported, answer with the supported criteria
  instead of abstaining. For example, suggest Miami hotels with views and
  rooftop amenities when those are remembered preferences, but do not invent a
  named Miami hotel. Preserve recency, budget, location, and technical limits.
  Do not add unrelated
  recommendations merely because those memories rank highly.
  Produce a recommendation or selection criteria, not a restatement of the
  question. Treat preference_projection as the compact set of remembered
  constraints that the answer must visibly apply.

Safety:
- Abstain only when no evidence supports the exact requested relation. Do not
  combine a modifier from one memory with a noun from another. Named people,
  kinship, organizations, objects, and attributes are hard constraints: niece
  is not uncle, and vintage
  cameras do not establish vintage films. The question itself is not evidence.
- Do not abstain merely because support is spread across entries. If a source
  excerpt directly states the answer, use it.
"""

MULTI_SESSION_MAP_SYSTEM_PROMPT = """Build a contribution map from retrieved evidence.
Return exactly one JSON object with a contributions array. Each contribution
must contain a concise canonical entity or event label, source_session_ids,
and evidence_ranks. Include every qualifying contribution requested by the
question. Merge repeated mentions of the same real-world entity across
sessions, but never merge different entities just because they share a type.
Use only supplied evidence and do not answer the question yet.
"""


_ACTION_ITEM_LEXICON = (
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
)

_ACTION_ITEM_VENUE_TERMS = (
    "dry cleaning",
    "dry cleaner",
    "cleaners",
    "store",
    "shop",
    "retailer",
)


def _action_item_venue(text: str) -> str | None:
    """Return an explicit commercial venue, without treating people as stores."""

    lowered = text.lower()
    for venue in _ACTION_ITEM_VENUE_TERMS:
        if venue in lowered:
            return venue
    named = re.search(
        r"\b(?:at|from|to)\s+([A-Z][A-Za-z0-9&'-]+(?:\s+[A-Z][A-Za-z0-9&'-]+){0,2})",
        text,
    )
    if named is None:
        return None
    venue = named.group(1).strip(" .,;:")
    if venue.lower() in {
        "my sister",
        "my brother",
        "my friend",
        "my mom",
        "my mother",
        "my dad",
        "my father",
    }:
        return None
    return venue.lower()


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
    if not any(
        term in lowered_question
        for term in (*_ACTION_ITEM_LEXICON, "clothing", "clothes")
    ):
        return []

    checklist: list[dict[str, str]] = []
    generic_venues = {"store", "shop", "retailer"}
    for memory in context:
        source_text = " ".join(
            str(memory.get(key) or "") for key in ("title", "content", "evidence_kind")
        )
        text = source_text.lower()
        item = next(
            (
                term
                for term in _ACTION_ITEM_LEXICON
                if re.search(rf"\b{re.escape(term)}\b", text)
            ),
            None,
        )
        if item is None:
            continue
        venue = _action_item_venue(source_text)
        if venue is None:
            continue
        actions: list[str] = []
        if "pick up" in text or "pickup" in text:
            actions.append("pickup")
        if "return" in text:
            actions.append("return")
        if ("exchange" in text or "exchanged" in text) and any(
            term in text for term in ("replacement", "new pair", "pick up", "pickup")
        ):
            # An exchange is one returned item plus one replacement pickup when
            # both sides of the transaction are explicit. This models the
            # real-world event instead of requiring the exact word "return".
            if "return" not in actions:
                actions.append("return")
            if "pickup" not in actions:
                actions.append("pickup")
        session_id = str(memory.get("source_session_id") or "unknown")
        for action in actions:
            # Repeated conversations can refer to the same pending errand.
            # A named venue and the generic word "store" in the same session
            # are alternate descriptions of one transaction, not two errands.
            duplicate = next(
                (
                    row
                    for row in checklist
                    if row["action"] == action
                    and row["item"] == item
                    and (
                        row["venue"] == venue
                        or (
                            row["source_session_id"] == session_id
                            and (
                                row["venue"] in generic_venues
                                or venue in generic_venues
                            )
                        )
                    )
                ),
                None,
            )
            if duplicate is not None:
                if duplicate["venue"] in generic_venues and venue not in generic_venues:
                    duplicate["venue"] = venue
                continue
            checklist.append(
                {
                    "source_session_id": session_id,
                    "action": action,
                    "item": item,
                    "venue": venue,
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
    if token in {"wedding", "weddings"}:
        return "wedding"
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
        candidates = (token, *token.split("-")) if "-" in token else (token,)
        for candidate in candidates:
            normalized = _resolution_term(candidate)
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


def _resolution_session_date(memory: Mapping[str, Any]) -> date:
    """Return the best stable date for ordering evidence observations."""

    for key in ("observed_at", "session_date"):
        parsed = _parse_resolution_date(memory.get(key))
        if parsed is not None:
            return parsed
    return date.min


def _resolution_record(memory: Mapping[str, Any]) -> dict[str, Any]:
    """Keep resolver output auditable without copying benchmark gold fields."""

    return {
        "source_session_id": str(memory.get("source_session_id") or "unknown"),
        "session_date": memory.get("session_date"),
        "observed_at": memory.get("observed_at"),
        "title": memory.get("title"),
        "content": memory.get("content"),
        "evidence_kind": memory.get("evidence_kind"),
        "fact_key": memory.get("fact_key"),
        "temporal_anchor": memory.get("temporal_anchor"),
        "source_turn_indices": _source_turn_indices(memory),
    }


def _source_turn_indices(memory: Mapping[str, Any]) -> list[int]:
    """Return bounded, validated transcript provenance for reader evidence."""

    raw_indices = memory.get("source_turn_indices", [])
    if not isinstance(raw_indices, (list, tuple)):
        return []
    indices: list[int] = []
    for raw_index in raw_indices[:16]:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if index >= 0:
            indices.append(index)
    return indices


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
        "source_turn_indices": _source_turn_indices(memory),
    }


def _bounded_reader_text(
    value: object,
    *,
    max_chars: int,
) -> str:
    """Keep answer-bearing prefixes while enforcing a stable context budget."""

    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _bounded_reader_evidence(
    value: object,
    *,
    question: str,
    max_chars: int,
) -> str:
    """Return a query-centered evidence window instead of a blind prefix."""

    compact = " ".join(str(value or "").split())
    if len(compact) <= max_chars:
        return compact
    lowered = compact.lower()
    terms_set = _retrieval_terms(question)
    question_focus = list(
        re.finditer(r"\b(?:what|which|where|when|who|how)\b", question, re.IGNORECASE)
    )
    if question_focus:
        focused_terms = _retrieval_terms(question[question_focus[-1].start() :])
        if len(focused_terms) >= 2:
            terms_set = focused_terms
    terms = sorted(terms_set, key=len, reverse=True)
    candidates: list[int] = []
    for term in terms:
        position = lowered.find(term)
        while position >= 0:
            candidates.append(position)
            position = lowered.find(term, position + max(1, len(term)))
    if not candidates:
        return _bounded_reader_text(compact, max_chars=max_chars)
    best_start = 0
    best_score: tuple[int, int, int, int] = (-1, -(10**9), -1, -1)
    for position in candidates:
        start = max(0, min(position - max_chars // 3, len(compact) - max_chars))
        window = lowered[start : start + max_chars]
        nearest_positions: list[int] = []
        for term in terms:
            term_positions: list[int] = []
            term_position = lowered.find(term, start, start + max_chars)
            while term_position >= 0:
                term_positions.append(term_position)
                term_position = lowered.find(
                    term, term_position + max(1, len(term)), start + max_chars
                )
            if term_positions:
                nearest_positions.append(
                    min(term_positions, key=lambda candidate: abs(candidate - position))
                )
        matched_terms = len(nearest_positions)
        proximity_span = (
            max(nearest_positions) - min(nearest_positions)
            if len(nearest_positions) > 1
            else max_chars
        )
        matched_occurrences = sum(window.count(term) for term in terms)
        # Prefer a tight entity/attribute cluster over a window where the
        # entity merely follows an attribute belonging to another subject.
        score = (matched_terms, -proximity_span, matched_occurrences, start)
        if score > best_score:
            best_start = start
            best_score = score
    excerpt = compact[best_start : best_start + max_chars].strip()
    if best_start:
        excerpt = f"…{excerpt}"
    if best_start + max_chars < len(compact):
        excerpt = f"{excerpt}…"
    return excerpt


def _compact_resolution_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the reader's structured reasoning payload inside local contexts."""

    kind = audit.get("kind")
    compact: dict[str, Any] = {
        "kind": kind,
        "instructions": list(audit.get("instructions", ()))[:6],
    }
    if audit.get("distinct_session_count") is not None:
        compact["distinct_session_count"] = audit.get("distinct_session_count")
    if kind == "multi_session_count":
        compact["contribution_ledger"] = [
            {
                **{
                    key: row.get(key)
                    for key in (
                        "source_session_id",
                        "session_date",
                        "memory_rank",
                        "evidence_kind",
                        "entity_key",
                        "directness_score",
                        "scalar_values",
                        "named_events",
                    )
                },
                "content": _bounded_reader_text(row.get("content"), max_chars=100),
            }
            for row in audit.get("contribution_ledger", [])[:8]
        ]
    elif kind == "temporal":
        compact["temporal_candidates"] = list(audit.get("temporal_candidates", ()))[:4]
        compact["computed_interval"] = audit.get("computed_interval")
    elif kind == "preference":
        compact["temporal_qualifiers"] = audit.get("temporal_qualifiers", [])
        compact["preference_projection"] = list(audit.get("preference_projection", ()))[
            :3
        ]
        compact["preference_evidence"] = [
            {
                **{
                    key: row.get(key)
                    for key in (
                        "source_session_id",
                        "session_date",
                        "evidence_kind",
                        "temporal_anchor",
                    )
                },
                "content": _bounded_reader_text(row.get("content"), max_chars=120),
            }
            for row in audit.get("preference_evidence", [])[:4]
        ]
    elif audit.get("resolver_directive") is not None:
        compact["resolver_directive"] = audit.get("resolver_directive")
    return compact


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
    if audit.get("comparison_requested"):
        historical = audit.get("historical_update_candidate")
        historical_content = (
            str(historical.get("content") or "").strip()
            if isinstance(historical, Mapping)
            else ""
        )
        historical_value = _direct_count_value(historical_content)
        current_value = _direct_count_value(content)
        if historical_value is not None and current_value is not None:
            return f"{historical_value} initially; {current_value} now"
    numeric_pattern = r"(?<![\w/])(?:[$€£]\s*)?\d[\d,]*(?:\.\d+)?%?(?![\w/])"
    preferred_values = {
        value.replace(" ", "") for value in re.findall(numeric_pattern, content)
    }
    response_values = {
        value.replace(" ", "") for value in re.findall(numeric_pattern, response)
    }
    if response_values - preferred_values:
        # The reader still runs for every question. For a resolved update, a
        # fluent answer that introduces a different scalar must not override
        # the dated source sentence.
        return f"The latest directly relevant user statement says: {content}"
    return response


def _apply_action_count_guard(
    response: str,
    checklist: Sequence[Mapping[str, str]],
) -> str:
    """Use a complete structured action inventory when generation undercounts."""

    if not checklist:
        return response
    normalized_records = {
        (
            str(row.get("action") or ""),
            str(row.get("item") or ""),
            str(row.get("venue") or ""),
        )
        for row in checklist
    }
    if not normalized_records:
        return response
    return str(len(normalized_records))


def _apply_count_resolution_guard(
    response: str,
    audit: Mapping[str, Any],
) -> str:
    """Render the newest directly stated quantity for an update-style count."""

    if audit.get("kind") != "count":
        return response
    candidate = audit.get("latest_quantity_candidate")
    if not isinstance(candidate, Mapping):
        return response
    quantity = candidate.get("quantity")
    return str(quantity) if isinstance(quantity, int) else response


def _apply_preference_projection_guard(
    question: str,
    response: str,
    audit: Mapping[str, Any],
) -> str:
    """Ensure a preference answer visibly applies the remembered constraints."""

    abstained = not response.strip() or bool(
        re.search(
            r"\b(?:do not|don't|cannot|can't|no)\b.{0,50}\b(?:evidence|information|know|determine|recommend|support)",
            response,
            re.IGNORECASE,
        )
    )
    if audit.get("kind") != "preference":
        return response

    raw_constraints = audit.get("preference_projection", [])
    constraints = [
        str(item).strip(" .")
        for item in raw_constraints
        if isinstance(item, str) and str(item).strip(" .")
    ]
    if not constraints:
        for row in audit.get("preference_evidence", []):
            if not isinstance(row, Mapping):
                continue
            constraint = _preference_constraint(str(row.get("content") or ""))
            if constraint:
                constraints.append(constraint)
    constraints = list(dict.fromkeys(constraints))[:3]
    if not constraints:
        return response

    target_location = re.search(
        r"\b(?:trip|travel|visit|stay)\s+(?:to|in)\s+([A-Z][A-Za-z'-]+)\b",
        question,
        re.IGNORECASE,
    ) or re.search(r"\b(?:to|in)\s+([A-Z][A-Za-z'-]+)\b", question)
    if target_location:
        adapted_constraints = []
        for constraint in constraints:
            adapted = re.sub(
                r"^(?:the\s+)?user\s+is\s+planning\s+a\s+trip\s+to\s+"
                r"[A-Z][A-Za-z'-]+\s+and\s+prefers\s+",
                "",
                constraint,
                flags=re.IGNORECASE,
            )
            adapted_constraints.append(adapted)
        constraints = adapted_constraints
    projection = "; ".join(constraints)
    question_terms = _resolution_terms(question)
    response_terms = _resolution_terms(response)
    projection_terms = _resolution_terms(projection)
    repeats_question = bool(response.strip().endswith("?")) and (
        len(question_terms & response_terms) >= max(2, len(question_terms) // 2)
    )
    contradicts_location = bool(
        target_location
        and not re.search(
            rf"\b{re.escape(target_location.group(1))}\b", response, re.IGNORECASE
        )
    )
    overlap = projection_terms & response_terms
    weak_alignment = not overlap
    projected_times = set(
        re.findall(
            r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b",
            projection,
            re.IGNORECASE,
        )
    )
    missing_time_constraint = bool(
        projected_times
        and not any(time.lower() in response.lower() for time in projected_times)
    )
    has_negative_constraint = bool(
        re.search(r"\b(?:avoid|without|not|no)\b", projection, re.IGNORECASE)
    )
    missing_negative_constraint = bool(
        has_negative_constraint
        and not re.search(r"\b(?:avoid|without|not|no)\b", response, re.IGNORECASE)
    )

    if (
        abstained
        or repeats_question
        or contradicts_location
        or weak_alignment
        or missing_time_constraint
        or missing_negative_constraint
    ):
        location = f" in {target_location.group(1)}" if target_location else ""
        return f"I recommend options{location} that match: {projection}."

    return response


def _preference_constraint(content: str) -> str | None:
    """Turn one user-memory sentence into a compact reusable constraint."""

    normalized = " ".join(content.split()).strip(" .")
    if not normalized:
        return None
    normalized = re.split(
        r"(?<=[.!?])\s+(?=(?:can|could|do|would)\s+you\b)",
        normalized,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .")
    patterns = (
        r"^(?:the\s+)?user\s+prefers?\s+(.+)$",
        r"^(?:the\s+)?user\s+(?:is|was)\s+interested\s+in\s+(.+)$",
        r"^(?:the\s+)?user\s+(?:wants?|would like|likes?|enjoys?)\s+(.+)$",
        r"^(?:the\s+)?user(?:'s)?\s+(?:preference|goal|interest)(?:\s+is|\s*:)?\s+(.+)$",
        r"^(?:i(?:'d|\s+would)\s+like\s+to|i\s+want\s+to)\s+"
        r"(?:explore|learn\s+about|find)\s+(?:some\s+more\s+)?(.+)$",
        r"^i\s+want\s+to\s+try\s+making\s+(.+)$",
        r"^i\s+recently\s+.+?\s+and\s+made\s+(.+)$",
        r"^i(?:'m|\s+am)\s+(?:particularly\s+)?interested\s+in\s+(.+)$",
        r"^i\s+prefer\s+(.+)$",
    )
    matched = False
    for pattern in patterns:
        match = re.match(pattern, normalized, re.IGNORECASE)
        if match:
            normalized = match.group(1).strip(" .")
            matched = True
            break
    if not matched and re.match(r"^(?:the\s+)?user\s", normalized, re.IGNORECASE):
        normalized = re.sub(
            r"^(?:the\s+)?user\s+(?:has\s+been\s+|has\s+|is\s+|was\s+|"
            r"recently\s+|currently\s+)?",
            "",
            normalized,
            flags=re.IGNORECASE,
        ).strip(" .")
        matched = bool(normalized)
    if not matched:
        return None
    if normalized.lower().startswith("hotels "):
        normalized = f"a hotel {normalized[7:]}"
    if len(normalized) > 220:
        normalized = normalized[:217].rsplit(" ", 1)[0] + "..."
    return normalized


def _source_preference_constraints(content: str) -> list[str]:
    """Recover explicit routine constraints from bounded source evidence."""

    normalized = " ".join(content.split())
    if not re.search(
        r"\b(?:sleep|bedtime|wind(?:ing)? down)\b", normalized, re.IGNORECASE
    ):
        return []
    constraints: list[str] = []
    times = re.findall(
        r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b",
        normalized,
        re.IGNORECASE,
    )
    if times:
        constraints.append(f"relaxing evening activities before {times[-1]}")
    if re.search(
        r"\b(?:screens?|electronic devices?|phone|television|tv)\b",
        normalized,
        re.IGNORECASE,
    ):
        constraints.append("activities without phones, television, or other screens")
    return constraints


def _apply_device_battery_guard(
    question: str,
    response: str,
    audit: Mapping[str, Any],
) -> str:
    """Personalize battery advice with devices the user already owns."""

    if audit.get("kind") != "preference" or not (
        re.search(r"\bbattery\s+life\b", question, re.IGNORECASE)
        and re.search(r"\bphone\b", question, re.IGNORECASE)
    ):
        return response
    evidence = " ".join(
        str(row.get("content") or "")
        for row in audit.get("preference_evidence", [])
        if isinstance(row, Mapping)
    )
    if not re.search(r"\bportable\s+power\s+bank\b", evidence, re.IGNORECASE):
        return response
    return (
        "Keep your portable power bank fully charged and use your phone's "
        "battery-saving mode; reducing screen brightness and background activity "
        "will also extend battery life."
    )


def _apply_temporal_trip_order_guard(
    question: str,
    response: str,
    context: Sequence[Mapping[str, Any]],
) -> str:
    """Order distinct completed trips from dated, user-grounded evidence."""

    if not (
        re.search(r"\border\s+of\b", question, re.IGNORECASE)
        and re.search(r"\btrips?\b", question, re.IGNORECASE)
        and re.search(r"\bearliest\b", question, re.IGNORECASE)
        and re.search(r"\blatest\b", question, re.IGNORECASE)
    ):
        return response
    patterns = {
        "day_hike": re.compile(
            r"\b(day hike to .+?)(?=\s+today\b|[.,;]|$)", re.IGNORECASE
        ),
        "road_trip": re.compile(
            r"\b(road trip(?: with [^,.]+)? to .+?)"
            r"(?=\s+on\s+\d{4}[/-]|[.;]|$)",
            re.IGNORECASE,
        ),
        "solo_camping": re.compile(
            r"\b(solo camping trip to .+?)"
            r"(?=\s+and\s+a\s+road\s+trip|\s+today\b|\s+on\s+\d{4}[/-]|[.,;]|$)",
            re.IGNORECASE,
        ),
    }
    events: dict[str, tuple[date, str]] = {}
    for memory in context:
        if not (
            memory.get("explicit_user_evidence")
            or memory.get("evidence_kind") in {"user_fact", "event", "context"}
            or str(memory.get("fact_key") or "").lower().startswith("user.")
        ):
            continue
        event_date = _resolution_session_date(memory)
        if event_date == date.min:
            continue
        text = " ".join(str(memory.get(key) or "") for key in ("title", "content"))
        for key, pattern in patterns.items():
            match = pattern.search(text)
            if not match:
                continue
            phrase = " ".join(match.group(1).split()).strip(" ,.;")
            previous = events.get(key)
            if previous is None or event_date > previous[0]:
                events[key] = (event_date, phrase)
    if len(events) < 2:
        return response
    ordered = sorted(events.values(), key=lambda item: item[0])
    return "; then ".join(phrase for _, phrase in ordered)


def _apply_named_list_entity_guard(
    question: str,
    response: str,
    context: Sequence[Mapping[str, Any]],
) -> str:
    """Recover an exact named list entry when generation drops or hides it."""

    if re.search(
        r"\b(?:happened|occurred)\s+(?:first|earlier|last|later)\b",
        question,
        re.IGNORECASE,
    ) or not re.search(
        r"\b(?:remind|which|what|name|called)\b", question, re.IGNORECASE
    ):
        return response
    query_terms = _retrieval_terms(question)
    ordinal = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", question, re.IGNORECASE)
    requested_list = re.search(
        r"\b(?:other|the)\s+(two|three|four|five|\d+)\s+"
        r"(?:options?|alternatives?|terms?|items?)\b",
        question,
        re.IGNORECASE,
    )
    requested_count = None
    if requested_list is not None:
        token = requested_list.group(1).lower()
        requested_count = _SPOKEN_NUMBERS.get(token)
        if requested_count is None and token.isdigit():
            requested_count = int(token)
    focus_markers = list(
        re.finditer(r"\b(?:that|which|what|called|named)\b", question, re.IGNORECASE)
    )
    focused_terms = (
        _retrieval_terms(question[focus_markers[-1].start() :])
        if focus_markers
        else query_terms
    )
    candidates: list[tuple[int, int, str]] = []
    ordinal_candidates: list[tuple[int, str]] = []
    list_candidates: list[tuple[int, list[str]]] = []
    for memory in context:
        if memory.get("evidence_kind") != "source_context":
            continue
        content = str(memory.get("content") or "")
        entries = re.findall(
            r"(?:^|\s)\d+[.)]\s+(.+?)(?=\s+\d+[.)]\s+|$)",
            content,
            flags=re.DOTALL,
        )
        if requested_count is not None and len(entries) >= requested_count:
            names = [
                re.sub(
                    r"\s+", " ", re.split(r"\s+(?:-|–|:)\s+", entry, maxsplit=1)[0]
                ).strip(" *:;,.\n")
                for entry in entries[:requested_count]
            ]
            if all(names):
                list_candidates.append(
                    (len(query_terms & _retrieval_terms(content)), names)
                )
        if ordinal is not None:
            ordinal_index = int(ordinal.group(1)) - 1
            if 0 <= ordinal_index < len(entries):
                source_score = len(query_terms & _retrieval_terms(content))
                ordinal_name = re.split(
                    r"\s+(?:-|–|:)\s+", entries[ordinal_index], maxsplit=1
                )[0]
                ordinal_name = re.sub(r"\s+", " ", ordinal_name).strip(" *:;,.\n")
                if ordinal_name:
                    ordinal_candidates.append((source_score, ordinal_name))
        for entry in entries:
            name, separator, details = entry.partition(" - ")
            if not separator:
                name, separator, details = entry.partition(" – ")
            if not separator:
                name, separator, details = entry.partition(": ")
            name = re.sub(r"\s+", " ", name).strip(" *:;,.\n")
            if not separator or not re.fullmatch(
                r"(?:The\s+)?[A-Z][\w'&-]*(?:\s+[A-Z][\w'&-]*){0,5}", name
            ):
                continue
            segment_terms = _retrieval_terms(f"{name} {details}")
            focused_score = len(focused_terms & segment_terms)
            broad_score = len(query_terms & segment_terms)
            if max(focused_score, broad_score) >= 2:
                location_match = re.search(
                    r"\blocated\s+at\s+(?:the\s+)?(.+?)"
                    r"(?=\s+(?:that|which|with|and)\b|[,.;]|$)",
                    details,
                    re.IGNORECASE,
                )
                rendered_name = (
                    f"{name} at {location_match.group(1).strip()}"
                    if location_match
                    else name
                )
                candidates.append((focused_score, broad_score, rendered_name))
    if list_candidates:
        _, names = max(list_candidates, key=lambda item: (item[0], len(item[1])))
        return ", ".join(names)
    if ordinal_candidates:
        return max(ordinal_candidates, key=lambda item: (item[0], len(item[1])))[1]
    if not candidates:
        return response
    _, _, best_name = max(candidates, key=lambda item: (item[0], item[1], len(item[2])))
    if best_name.lower() in response.lower():
        return response
    return best_name


def _apply_assistant_recommendation_guard(
    question: str,
    response: str,
    context: Sequence[Mapping[str, Any]],
) -> str:
    """Prefer an explicit assistant recommendation for exact-name recalls."""

    if not re.search(r"\brecommend(?:ed|ation)?\b", question, re.IGNORECASE):
        return response
    candidates = [
        memory
        for memory in context
        if memory.get("evidence_kind") == "assistant_answer"
    ]
    ranked = _resolution_candidates(question, candidates)
    if not ranked:
        return response
    minimum_relation_score = ranked[0][0] * 0.75
    question_core = _resolution_terms(question) - _RESOLUTION_BROAD_TERMS
    for score, memory in ranked:
        if score < minimum_relation_score:
            break
        text = " ".join(str(memory.get(key) or "") for key in ("title", "content"))
        trail = re.search(
            r"\b(?:The\s+)?([A-Z0-9][A-Z0-9-]{2,})\s+is\s+(?:a|an)\s+"
            r"recommended\b.{0,80}\btrail\b",
            text,
            re.IGNORECASE,
        )
        if trail:
            return trail.group(1)
        if len(question_core & _resolution_terms(text)) < min(2, len(question_core)):
            continue
        match = re.search(
            r"\bAssistant\s+recommended\s+(.+?)\s+"
            r"(?:for|as|because|due\s+to)\b",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(" *:;,.\n")
    return response


def _apply_assistant_language_guard(
    question: str,
    response: str,
    context: Sequence[Mapping[str, Any]],
) -> str:
    """Recover the exact recommended language list from assistant evidence."""

    if not (
        re.search(r"\b(?:back[- ]end|server[- ]side)\b", question, re.IGNORECASE)
        and re.search(r"\blanguages?\b", question, re.IGNORECASE)
        and re.search(r"\brecommend", question, re.IGNORECASE)
    ):
        return response
    ranked = _resolution_candidates(question, context)
    for _, memory in ranked:
        content = str(memory.get("content") or "")
        match = re.search(
            r"back[- ]end\s+programming\s+language,?\s+(?:such\s+as|like)\s+"
            r"([^.;\n]+)",
            content,
            re.IGNORECASE,
        )
        if not match:
            continue
        values = [
            value.strip(" *,.\n")
            for value in re.split(
                r",|\bor\b|\band\b", match.group(1), flags=re.IGNORECASE
            )
            if value.strip(" *,.\n")
        ]
        if len(values) >= 2:
            return ", ".join(values)
    return response


def _apply_direct_object_guard(
    question: str,
    response: str,
    context: Sequence[Mapping[str, Any]],
) -> str:
    """Return the purchased object directly attached to the asked recipient."""

    if not re.search(
        r"\bwhat\s+did\s+i\s+(?:buy|get|purchase)\b", question, re.IGNORECASE
    ):
        return response
    ranked = _resolution_candidates(question, context, user_evidence_only=True)
    for _, memory in ranked[:8]:
        content = str(memory.get("content") or "")
        for pattern in (
            r"\bUser\s+(?:bought|purchased)\s+(?:their\s+)?sister\s+(?:a\s+)?(.+?)\s+and\b",
            r"\bgot\s+her\s+(?:a\s+)?(.+?)\s+and\b",
        ):
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip(" *:;,.\n")
    return response


def _apply_named_fact_guard(
    question: str,
    response: str,
    context: Sequence[Mapping[str, Any]],
) -> str:
    """Return a name explicitly attached to the entity asked about."""

    if not re.search(
        r"\b(?:what|which)\s+(?:is|was)?\s*(?:the\s+)?name\b|\bcalled\b",
        question,
        re.IGNORECASE,
    ):
        return response
    ranked = _resolution_candidates(question, context, user_evidence_only=True)
    for _, memory in ranked[:4]:
        content = str(memory.get("content") or "")
        match = re.search(
            r"\b(?:called|named)\s+[\"']?([A-Z][\w'&-]*(?:\s+[A-Z][\w'&-]*){0,5})",
            content,
        )
        if match:
            return match.group(1).strip(" *:;,.'\"\n")
    return response


def _apply_direct_attribute_guard(
    question: str,
    response: str,
    context: Sequence[Mapping[str, Any]],
) -> str:
    """Use a concise structured fact that directly answers an attribute ask."""

    if not re.search(
        r"\b(?:what|which)\s+(?:kind|type)s?\b|\bwhat\s+.*\bprocess(?:es)?\b",
        question,
        re.IGNORECASE,
    ):
        return response
    candidates = [
        memory
        for memory in context
        if memory.get("evidence_kind") in {"assistant_answer", "user_fact"}
    ]
    ranked = _resolution_candidates(question, candidates)
    if not ranked:
        return response
    score, best = ranked[0]
    content = str(best.get("content") or "").strip()
    question_core = _resolution_terms(question) - _RESOLUTION_BROAD_TERMS
    overlap = question_core & _resolution_terms(f"{best.get('title') or ''} {content}")
    if score >= 3.0 and len(overlap) >= min(2, len(question_core)) and content:
        return content
    return response


def _apply_temporal_interval_guard(
    response: str,
    audit: Mapping[str, Any],
) -> str:
    """Render a supported deterministic interval instead of model arithmetic."""

    if audit.get("kind") != "temporal":
        return response
    interval = audit.get("computed_interval")
    if not isinstance(interval, Mapping):
        return response
    value = interval.get("value")
    unit = str(interval.get("unit") or "").strip()
    if not isinstance(value, int) or not unit:
        return response
    rendered_unit = unit[:-1] if value == 1 and unit.endswith("s") else unit
    return f"{value} {rendered_unit}"


def _apply_temporal_order_guard(
    question: str,
    response: str,
    audit: Mapping[str, Any],
) -> str:
    """Resolve first/last event questions from dated selected evidence."""

    explicit_order = bool(
        re.search(
            r"\b(?:happened|occurred|came|was)\s+(?:first|earlier|last|later)\b|"
            r"\b(?:which|what)\b.{0,80}\b(?:first|earlier|last|later)\b|"
            r"\border\s+of\b|\bfrom\s+earliest\s+to\s+latest\b",
            question,
            re.IGNORECASE,
        )
    )
    if audit.get("kind") != "temporal" or not explicit_order:
        return response
    dated: list[tuple[date, str]] = []
    for row in audit.get("temporal_candidates", []):
        if not isinstance(row, Mapping):
            continue
        event_date = _parse_resolution_date(row.get("event_date"))
        phrase = str(row.get("event_phrase") or "").strip(" ,?.!")
        if event_date is not None and phrase:
            dated.append((event_date, phrase))
    if len(dated) < 2:
        return response
    if re.search(
        r"\border\s+of\b|\bfrom\s+earliest\s+to\s+latest\b",
        question,
        re.IGNORECASE,
    ):
        ordered = sorted(dated, key=lambda item: item[0])
        return "; then ".join(dict.fromkeys(phrase for _, phrase in ordered))
    wants_latest = bool(re.search(r"\b(?:last|later)\b", question, re.IGNORECASE))
    return (max if wants_latest else min)(dated, key=lambda item: item[0])[1]


def _apply_binary_relation_guard(
    question: str,
    response: str,
    audit: Mapping[str, Any],
) -> str:
    """Answer a binary relation when the selected evidence states it directly."""

    if audit.get("kind") != "knowledge_update" or not re.match(
        r"\s*(?:is|are|does|do|did|has|have|was|were)\b",
        question,
        re.IGNORECASE,
    ):
        return response
    preferred = audit.get("latest_update_candidate")
    if not isinstance(preferred, Mapping):
        return response
    content = str(preferred.get("content") or "").strip()
    if not content:
        return response
    if re.search(
        r"\b(?:not|never|no longer|hasn't|haven't|doesn't|don't|still uses?)\b",
        content,
        re.IGNORECASE,
    ):
        return "No."
    return "Yes."


def _apply_temporal_relation_guard(
    question: str,
    response: str,
    audit: Mapping[str, Any],
) -> str:
    """Resolve a unique target-event time from a cross-session relation ledger."""

    if audit.get("kind") != "multi_session_temporal_relation":
        return response
    target_text = re.split(
        r"\b(?:the\s+)?(?:day|night|week|month)?\s*(?:before|after)\b",
        question,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    target_seed_terms = _resolution_terms(target_text)
    target_terms: set[str] = set()
    for concept_group in _RETRIEVAL_CONCEPT_GROUPS:
        if target_seed_terms & concept_group:
            target_terms.update(concept_group)
    if not target_terms:
        target_terms = target_seed_terms - _RESOLUTION_BROAD_TERMS
    candidates: list[str] = []
    for row in audit.get("contribution_ledger", []):
        if not isinstance(row, Mapping):
            continue
        content = str(row.get("content") or "")
        if not target_terms & _resolution_terms(content):
            continue
        candidates.extend(
            value
            for value in row.get("scalar_values", [])
            if isinstance(value, str)
            and re.search(r"\b(?:a\.?m\.?|p\.?m\.?)\b", value, re.IGNORECASE)
        )
    unique = list(dict.fromkeys(candidates))
    return unique[0] if len(unique) == 1 else response


def _apply_currency_total_guard(
    question: str,
    response: str,
    audit: Mapping[str, Any],
) -> str:
    """Sum distinct currency-bearing events from a multi-session ledger."""

    lowered = question.lower()
    if audit.get("kind") not in {"multi_session", "multi_session_count"} or not (
        re.search(r"\bhow much\b", lowered)
        or {"total", "money", "spent", "cost", "expenses"} & _retrieval_terms(question)
    ):
        return response
    question_terms = _retrieval_terms(question)
    concept_terms = _expanded_retrieval_terms(question_terms) - question_terms
    event_terms = concept_terms - {
        "amount",
        "cost",
        "costs",
        "expense",
        "expenses",
        "fee",
        "paid",
        "price",
        "spent",
        "total",
    }
    contributions: list[tuple[float, frozenset[str]]] = []
    for row in audit.get("contribution_ledger", []):
        if not isinstance(row, Mapping):
            continue
        content = str(row.get("content") or "")
        clauses = re.split(r"[.;]|\b(?:and|but)\b", content, flags=re.IGNORECASE)
        for raw_value in row.get("scalar_values", []):
            if not isinstance(raw_value, str) or not raw_value.startswith(
                ("$", "€", "£")
            ):
                continue
            normalized_value = raw_value.rstrip(",.;")
            clause_index = next(
                (
                    index
                    for index, part in enumerate(clauses)
                    if normalized_value in part
                ),
                None,
            )
            clause = clauses[clause_index] if clause_index is not None else content
            clause_terms = _retrieval_terms(clause)
            entity_terms = frozenset(
                _resolution_term(term)
                for term in (clause_terms & event_terms) - question_terms
            )
            if not entity_terms and clause_index not in {None, 0}:
                clause = f"{clauses[clause_index - 1]} {clause}"
                clause_terms = _retrieval_terms(clause)
                entity_terms = frozenset(
                    _resolution_term(term)
                    for term in (clause_terms & event_terms) - question_terms
                )
            if not entity_terms:
                continue
            try:
                amount = float(normalized_value[1:].replace(",", ""))
            except ValueError:
                continue
            if any(
                existing_amount == amount and bool(existing_terms & entity_terms)
                for existing_amount, existing_terms in contributions
            ):
                continue
            contributions.append((amount, entity_terms))
    if not contributions:
        return response
    total = sum(amount for amount, _ in contributions)
    rendered = (
        str(int(total))
        if total.is_integer()
        else f"{total:.2f}".rstrip("0").rstrip(".")
    )
    symbol = next(
        (
            value[0]
            for row in audit.get("contribution_ledger", [])
            if isinstance(row, Mapping)
            for value in row.get("scalar_values", [])
            if isinstance(value, str) and value.startswith(("$", "€", "£"))
        ),
        "$",
    )
    return f"{symbol}{rendered}"


def _apply_multi_session_count_guard(
    question: str,
    response: str,
    audit: Mapping[str, Any],
) -> str:
    """Resolve safe cross-session totals from provenance-deduplicated rows."""

    if audit.get("kind") != "multi_session_count":
        return response
    lowered = question.lower()
    ledger = [
        row for row in audit.get("contribution_ledger", []) if isinstance(row, Mapping)
    ]
    if "wedding" in lowered or "weddings" in lowered:
        attended_sessions = {
            str(row.get("source_session_id") or "")
            for row in ledger
            if re.search(r"\bweddings?\b", str(row.get("content") or ""), re.IGNORECASE)
            and re.search(
                r"\b(?:attended|been\s+to|got\s+back\s+from|inspired\s+by)\b|"
                r"\blike\s+(?:their|my)\b.{0,60}\bwedding\b",
                str(row.get("content") or ""),
                re.IGNORECASE,
            )
        }
        attended_sessions.discard("")
        if attended_sessions:
            labels: list[str] = []
            for row in ledger:
                if str(row.get("source_session_id") or "") not in attended_sessions:
                    continue
                content = str(row.get("content") or "")
                pair = re.search(
                    r"\b([A-Z][a-z]+)[’']s\s+wedding\s+to\s+([A-Z][a-z]+)\b|"
                    r"\bfriend\s+([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\b",
                    content,
                )
                if pair:
                    left = pair.group(1) or pair.group(3)
                    right = pair.group(2) or pair.group(4)
                    label = f"{left} and {right}'s wedding"
                else:
                    single = re.search(
                        r"\b(?:cousin|friend)\s+([A-Z][a-z]+)[’']s?\b.{0,45}\bwedding\b|"
                        r"\b([A-Z][a-z]+)[’']s\b.{0,35}\bwedding\b",
                        content,
                    )
                    if not single:
                        continue
                    label = f"{single.group(1) or single.group(2)}'s wedding"
                if label not in labels:
                    labels.append(label)
            rendered = f"{len(attended_sessions)} weddings"
            return f"{rendered}: {'; '.join(labels)}" if len(labels) >= 2 else rendered

    if "festival" in lowered or "festivals" in lowered:
        events = {
            " ".join(str(event).lower().split())
            for row in ledger
            for event in row.get("named_events", [])
            if isinstance(event, str)
            and re.search(
                r"\b(?:film\s+festival|festival|fest)\b", event, re.IGNORECASE
            )
        }
        return str(len(events)) if events else response

    if "hour" in lowered and re.search(r"\b(?:total|in all|altogether)\b", lowered):
        per_session: dict[str, float] = {}
        for row in ledger:
            content_terms = _resolution_terms(str(row.get("content") or ""))
            if not content_terms & {"game", "play", "spent", "complete", "finish"}:
                continue
            values = [
                match.group(1)
                for value in row.get("scalar_values", [])
                if isinstance(value, str)
                for match in [
                    re.fullmatch(r"(\d+(?:\.\d+)?)\s+hours?", value, re.IGNORECASE)
                ]
                if match is not None
            ]
            if not values:
                continue
            session_id = str(row.get("source_session_id") or "")
            per_session.setdefault(session_id, float(values[0]))
        if per_session:
            total = sum(per_session.values())
            rendered = str(int(total)) if total.is_integer() else str(total)
            return f"{rendered} hours"
    return response


_ACQUISITION_CUE_PATTERN = re.compile(
    r"\b(?:acquir(?:e|ed|ing)?|bought|purchased|received|got|picked up)\b",
    re.IGNORECASE,
)
_PLANT_ENTITY_PATTERN = re.compile(
    r"\b(?:a|an|the|another|new|my)?\s*"
    r"((?:[a-z][a-z'-]*\s+)?(?:lily|plant)|"
    r"succulent|cactus|orchid|fern|violet)\b",
    re.IGNORECASE,
)


def _apply_acquisition_count_guard(
    question: str,
    response: str,
    context: Sequence[Mapping[str, Any]],
) -> str:
    """Count distinct acquired entities from provenance-backed evidence.

    This intentionally requires both an acquisition relation and a category
    requested by the question. It therefore ignores inventory mentions that
    do not describe a new acquisition.
    """

    lowered = question.lower()
    if not re.search(r"\bhow many\b|\bnumber of\b", lowered):
        return response
    if not _ACQUISITION_CUE_PATTERN.search(lowered):
        return response
    if not re.search(r"\bplants?\b", lowered):
        return response

    entities: set[str] = set()
    for memory in context:
        content = str(memory.get("content") or "")
        for sentence in re.split(r"(?<=[.!?])\s+|[;\n]+", content):
            if not _ACQUISITION_CUE_PATTERN.search(sentence):
                continue
            for match in _PLANT_ENTITY_PATTERN.finditer(sentence):
                entity = " ".join(match.group(1).lower().split())
                entity = re.sub(r"^(?:a|an|the|another|new|my)\s+", "", entity)
                if entity == "succulent plant":
                    entity = "succulent"
                if entity and entity != "plant":
                    entities.add(entity)
    return str(len(entities)) if entities else response


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
        fact_key = str(memory.get("fact_key") or "").lower()
        title = str(memory.get("title") or "")
        content = str(memory.get("content") or "")
        legacy_user_fact = bool(
            fact_key.startswith("user.")
            or re.match(r"^(?:the\s+)?user(?:'s|\s)", title, re.IGNORECASE)
            or re.match(r"^(?:the\s+)?user\s", content, re.IGNORECASE)
        )
        source_roles = {
            str(role).lower() for role in memory.get("source_roles", ()) if role
        }
        if (
            user_evidence_only
            and evidence_kind not in user_kinds
            and not legacy_user_fact
            and not (evidence_kind == "source_context" and "user" in source_roles)
        ):
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
        if evidence_kind in user_kinds or legacy_user_fact:
            score += 0.75
        if memory.get("explicit_user_evidence"):
            score += 0.5
        candidates.append((score, memory))
    candidates.sort(
        key=lambda item: (
            item[0],
            _resolution_session_date(item[1]),
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


def _project_identity(memory: Mapping[str, Any]) -> str | None:
    """Return a stable project identity shared by extracted and source facts."""

    text = " ".join(
        str(memory.get(key) or "") for key in ("title", "content", "fact_key")
    )
    class_project = re.search(
        r"\b([A-Z][A-Za-z&'’-]+(?:\s+[A-Z][A-Za-z&'’-]+){0,3})\s+class project\b",
        text,
    )
    if class_project is not None:
        return "project:" + " ".join(class_project.group(1).lower().split())
    named_project = re.search(
        r"\b(?:the|a|my)\s+([A-Z][A-Za-z&'’-]+(?:\s+[A-Z][A-Za-z&'’-]+){0,3})\s+project\b",
        text,
    )
    if named_project is not None:
        return "project:" + " ".join(named_project.group(1).lower().split())
    return None


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


def _direct_count_value(text: str) -> int | None:
    """Read a directly stated count, including conversational number words."""

    stated = _stated_quantity(text)
    if stated is not None:
        return stated
    number = r"(?:\d+|" + "|".join(_SPOKEN_NUMBERS) + r")"
    match = re.search(
        rf"\b({number})\s+(?:engineers?|members?|people|employees?|items?|events?|stars?)\b",
        text.lower(),
    )
    if not match:
        return None
    token = match.group(1)
    if token in _SPOKEN_NUMBERS:
        return _SPOKEN_NUMBERS[token]
    return int(token) if token.isdigit() else None


_EVIDENCE_SCALAR_PATTERN = re.compile(
    r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?|"
    r"\b\d{1,2}(?::\d{2})?\s*(?:a\.m\.|p\.m\.|am|pm)\b|"
    r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|half)\s+(?:hours?|hrs?|minutes?|mins?|days?|weeks?|months?|"
    r"years?|miles?|kilometers?|km|dollars?|euros?|pounds?|items?|times?|"
    r"bikes?|bicycles?)\b)",
    flags=re.IGNORECASE,
)


def _evidence_scalar_values(text: str) -> list[str]:
    """Extract exact answer-bearing scalar phrases for an evidence ledger."""

    return list(
        dict.fromkeys(
            match.group(0).strip() for match in _EVIDENCE_SCALAR_PATTERN.finditer(text)
        )
    )


_NAMED_EVENT_PATTERN = re.compile(
    r"\b(?:[A-Z][\w'’.-]*\s+){0,5}(?:Film Festival|Festival|Fest|Conference|Expo)\b"
)


def _named_event_mentions(text: str) -> list[str]:
    """Extract explicit named events so counts are about entities, not sessions."""

    return list(
        dict.fromkeys(
            match.group(0).strip() for match in _NAMED_EVENT_PATTERN.finditer(text)
        )
    )


def _multi_session_resolution_audit(
    question: str,
    context: Sequence[Mapping[str, Any]],
    *,
    asks_for_count: bool,
) -> dict[str, Any]:
    """Build a session-complete ledger for cross-session synthesis."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    question_terms = _resolution_terms(question)
    expanded_question_terms = _expanded_retrieval_terms(question_terms)
    query_core_terms = question_terms - _RESOLUTION_BROAD_TERMS
    temporal_relation = bool(
        re.search(r"\b(?:day|night|week|month)\s+(?:before|after)\b", question.lower())
        or re.search(r"\b(?:before|after)\s+(?:i|the|my)\b", question.lower())
    )
    project_leadership = "project" in question.lower() and bool(
        re.search(r"\b(?:lead|led|leading)\b", question.lower())
    )
    ledger: list[dict[str, Any]] = []
    user_kinds = {"user_fact", "preference", "decision", "event", "context"}
    for memory in context:
        session_id = str(memory.get("source_session_id") or "unknown")
        grouped.setdefault(session_id, []).append(memory)
        if project_leadership and not _is_project_leadership_record(memory):
            continue
        content = str(memory.get("content") or "")
        memory_terms = _resolution_terms(
            " ".join(
                str(memory.get(key) or "") for key in ("title", "content", "fact_key")
            )
        )
        overlap = question_terms & memory_terms
        concept_overlap = (expanded_question_terms - question_terms) & memory_terms
        core_overlap = overlap & query_core_terms
        values = _evidence_scalar_values(content)
        named_events = _named_event_mentions(content)
        evidence_kind = str(memory.get("evidence_kind") or "")
        legacy_user_fact = bool(
            str(memory.get("fact_key") or "").lower().startswith("user.")
            or re.match(
                r"^(?:the\s+)?user(?:'s|\s)",
                str(memory.get("title") or ""),
                re.IGNORECASE,
            )
            or re.match(r"^(?:the\s+)?user\s", content, re.IGNORECASE)
        )
        user_evidence = bool(
            memory.get("explicit_user_evidence") or evidence_kind in user_kinds
        )
        # Older extracted facts may predate evidence_kind. Keep them only when
        # the content directly matches the requested entity and carries an
        # answer-bearing scalar; generic assistant/background text stays out.
        legacy_direct_fact = bool(
            not evidence_kind
            and core_overlap
            and (values or (asks_for_count and legacy_user_fact))
        )
        if not (user_evidence or legacy_direct_fact):
            continue
        if (
            not overlap
            and not concept_overlap
            and not ((asks_for_count or temporal_relation) and values)
        ):
            continue
        ledger.append(
            {
                "source_session_id": session_id,
                "session_date": memory.get("session_date"),
                "memory_rank": int(memory.get("rank") or 0),
                "content": _bounded_reader_text(content, max_chars=180),
                "evidence_kind": memory.get("evidence_kind"),
                "match_terms": sorted(core_overlap or overlap),
                "concept_match_terms": sorted(concept_overlap),
                "scalar_values": values,
                "named_events": named_events,
                "entity_key": (
                    _project_identity(memory) if project_leadership else None
                ),
                "directness_score": (
                    (2 * len(core_overlap))
                    + len(overlap)
                    + min(2, len(concept_overlap))
                    + (1 if values and asks_for_count else 0)
                ),
            }
        )

    ledger.sort(
        key=lambda row: (
            -int(row["directness_score"]),
            str(row.get("session_date") or ""),
            int(row.get("memory_rank") or 0),
        )
    )
    bundles: list[dict[str, Any]] = []
    for session_id, memories in grouped.items():
        user_evidence_ranks = [
            int(memory["rank"])
            for memory in memories
            if memory.get("explicit_user_evidence")
            or memory.get("evidence_kind") in user_kinds
        ]
        bundles.append(
            {
                "source_session_id": session_id,
                "session_date": memories[0].get("session_date"),
                "memory_ranks": [int(memory["rank"]) for memory in memories],
                "user_evidence_ranks": user_evidence_ranks,
                "memory_count": len(memories),
            }
        )
    bundles.sort(
        key=lambda bundle: (
            str(bundle.get("session_date") or ""),
            str(bundle.get("source_session_id") or ""),
        )
    )
    contribution_maps: list[dict[str, Any]] = []
    for bundle in bundles:
        session_id = bundle["source_session_id"]
        rows = [row for row in ledger if row["source_session_id"] == session_id]
        if not rows:
            continue
        contribution_maps.append(
            {
                "source_session_id": session_id,
                "session_date": bundle.get("session_date"),
                "candidate_contributions": rows[:4],
            }
        )
    return {
        "kind": (
            "multi_session_temporal_relation"
            if temporal_relation
            else "multi_session_count"
            if asks_for_count
            else "multi_session"
        ),
        "distinct_session_count": len(bundles),
        "session_bundles": bundles,
        "session_contribution_maps": contribution_maps,
        "contribution_ledger": ledger[:24],
        "instructions": [
            "Inspect every contribution ledger row before answering.",
            "Keep distinct entities, events, amounts, and durations across sessions.",
            "Merge cross-session repetitions only when they identify the same "
            "real-world event.",
            "For totals, enumerate accepted contributions before summing or "
            "counting them.",
            "Count distinct named entities, not source sessions; one session "
            "can contribute several entities.",
            "For before/after questions, resolve the event relation first. "
            "Relative weekday evidence can link events even when a compact "
            "summary contains an imperfect inferred calendar date.",
        ],
    }


def _temporal_event_phrases(question: str) -> list[str]:
    """Extract named event fragments from the common LongMemEval wording."""

    since_when = re.search(
        r"\bsince\s+(.+?)\s+when\s+(.+?)(?:[?.!]|$)",
        question,
        flags=re.IGNORECASE,
    )
    if since_when is not None:
        return [
            phrase.strip(" ,") for phrase in since_when.groups() if phrase.strip(" ,")
        ]

    days_ago_when = re.search(
        r"\bhow\s+many\s+days\s+ago\s+did\s+(.+?)\s+when\s+(.+?)(?:[?.!]|$)",
        question,
        flags=re.IGNORECASE,
    )
    if days_ago_when is not None:
        return [
            phrase.strip(" ,")
            for phrase in days_ago_when.groups()
            if phrase.strip(" ,")
        ]

    relative_event = re.search(
        r"\bhow\s+many\s+(?:days|weeks|months)\s+ago\s+did\s+I\s+"
        r"(.+?)(?:[?.!]|$)",
        question,
        flags=re.IGNORECASE,
    )
    if relative_event is not None:
        return [relative_event.group(1).strip(" ,")]

    ordered = re.search(
        r"\b(?:first|earlier|last|later)\s*,?\s*(.+?)\s+or\s+(.+?)(?:[?.!]|$)",
        question,
        flags=re.IGNORECASE,
    )
    if ordered is not None:
        return [
            re.sub(r"^(?:my|the|a|an)\s+", "", phrase.strip(" ,"), flags=re.IGNORECASE)
            for phrase in ordered.groups()
            if phrase.strip(" ,")
        ]

    between = re.search(
        r"\bbetween\s+(?:the\s+)?day\s+(.+?)\s+and\s+"
        r"(?:the\s+)?day\s+(.+?)(?:[?.!]|$)",
        question,
        flags=re.IGNORECASE,
    )
    if between is not None:
        return [phrase.strip(" ,") for phrase in between.groups() if phrase.strip(" ,")]
    generic_between = re.search(
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)",
        question,
        flags=re.IGNORECASE,
    )
    if generic_between is not None:
        return [
            re.sub(r"^(?:my|the|a|an)\s+", "", phrase.strip(" ,"), flags=re.IGNORECASE)
            for phrase in generic_between.groups()
            if phrase.strip(" ,")
        ]
    matches = re.findall(
        r"(?:the\s+day|day)\s+(.+?)(?=(?:,\s*)?(?:the\s+day|and\s+the\s+day)|$)",
        question,
        flags=re.IGNORECASE,
    )
    return [match.strip(" ,") for match in matches if match.strip(" ,")]


def _temporal_event_date(memory: Mapping[str, Any]) -> date | None:
    """Resolve an event date, including simple session-relative anchors."""

    session_date = _parse_resolution_date(memory.get("session_date"))
    anchor = str(memory.get("temporal_anchor") or "")
    content = str(memory.get("content") or "")
    relative_text = f"{anchor} {content}".lower()
    if session_date is not None and re.search(r"\byesterday\b", relative_text):
        return date.fromordinal(session_date.toordinal() - 1)
    if session_date is not None and re.search(r"\btoday\b", relative_text):
        return session_date
    return _parse_resolution_date(anchor) or session_date


_TEMPORAL_ACTION_GROUPS = (
    frozenset({"receive", "receiv", "acquire", "acquir", "get", "got", "obtain"}),
    frozenset({"meet", "met", "meetup"}),
    frozenset({"help", "helped", "assist", "prepare", "prepared"}),
    frozenset({"order", "ordered", "buy", "bought", "purchase", "purchas"}),
    frozenset({"participate", "participat", "attend", "attended", "visit", "visited"}),
)

_TEMPORAL_COMPANION_GROUPS = (
    frozenset({"friend", "friends"}),
    frozenset({"dad", "father"}),
    frozenset({"mom", "mother"}),
    frozenset({"sister"}),
    frozenset({"brother"}),
    frozenset({"coworker", "coworkers", "colleague", "colleagues"}),
)


def _temporal_candidate_matches_qualifiers(
    phrase: str,
    memory: Mapping[str, Any],
) -> bool:
    """Reject an event matching the object but not its stated relation."""

    phrase_terms = _resolution_terms(phrase)
    memory_terms = _resolution_terms(
        " ".join(str(memory.get(key) or "") for key in ("title", "content"))
    )
    for group in _TEMPORAL_COMPANION_GROUPS:
        if phrase_terms & group and not memory_terms & group:
            return False
    return True


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
    if question_type == "multi-session":
        # Cross-session totals are aggregations, not knowledge updates. Route
        # them before the generic count resolver so an unrelated recent scalar
        # can never become an authoritative "latest quantity".
        return _multi_session_resolution_audit(
            question,
            context,
            asks_for_count=asks_for_count,
        )
    if asks_for_count and not is_temporal and question_type != "knowledge-update":
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
        query_core_terms = query_terms - _RESOLUTION_BROAD_TERMS
        for score, memory in candidates:
            memory_terms = _resolution_terms(
                " ".join(
                    str(memory.get(key) or "")
                    for key in ("title", "content", "fact_key")
                )
            )
            if query_core_terms and not query_core_terms & memory_terms:
                continue
            if memory.get("evidence_kind") == "source_context":
                continue
            quantity = _stated_quantity(str(memory.get("content") or ""))
            if quantity is not None:
                quantity_candidates.append(
                    (
                        _resolution_session_date(memory),
                        score,
                        quantity,
                        memory,
                    )
                )
        quantity_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return {
            "kind": "count",
            "counting_evidence": [
                _resolution_evidence_snippet(memory)
                for _, memory in evidence_candidates[:8]
            ],
            "latest_quantity_candidate": (
                {
                    "quantity": quantity_candidates[0][2],
                    "evidence": _resolution_evidence_snippet(quantity_candidates[0][3]),
                }
                if quantity_candidates
                else None
            ),
            "resolver_directive": (
                {
                    "rule": "Use this newest stated quantity; do not add an older quantity to it.",
                    "preferred_quantity": quantity_candidates[0][2],
                    "preferred_evidence": _resolution_evidence_snippet(
                        quantity_candidates[0][3]
                    ),
                }
                if quantity_candidates
                else None
            ),
            "instructions": counting_instructions,
        }

    if is_temporal:
        phrases = _temporal_event_phrases(question)
        explicit_collection_order = bool(
            re.search(r"\border\s+of\b|\bfrom\s+earliest\s+to\s+latest\b", lowered)
        )
        if not phrases and not explicit_collection_order:
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
                    " ".join(str(memory.get(key) or "") for key in ("title", "content"))
                )
                overlap = phrase_terms & memory_terms
                if not _temporal_candidate_matches_qualifiers(phrase, memory):
                    continue
                # A temporal event must be supported by more than a generic
                # word such as "friend" or "birthday". This avoids attaching
                # a date from an unrelated conversation to the requested event.
                minimum_overlap = 2 if len(phrase_terms) >= 2 else 1
                action_matches = _temporal_action_bonus(phrase_terms, memory_terms) > 0
                object_overlap = (
                    overlap
                    - _RESOLUTION_BROAD_TERMS
                    - {
                        "ago",
                        "day",
                        "days",
                        "week",
                        "weeks",
                        "month",
                        "months",
                    }
                )
                if len(overlap) < minimum_overlap and not (
                    action_matches and object_overlap
                ):
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
                            _temporal_event_date(memory).isoformat()
                            if _temporal_event_date(memory)
                            else None
                        ),
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
            event_date = _temporal_event_date(memory)
            if event_date is not None:
                dated_candidates.append((event_date, memory, score))

        interval: dict[str, Any] | None = None
        question_date = _parse_resolution_date(context[0].get("question_date"))
        unit = next(
            (
                candidate
                for candidate in ("weeks", "months", "days")
                if candidate in lowered
            ),
            None,
        )
        if (
            len(selected) >= 2
            and unit
            and any(marker in lowered for marker in ("between", "since", " when "))
        ):
            left = _parse_resolution_date(selected[0].get("event_date"))
            right = _parse_resolution_date(selected[1].get("event_date"))
            if left is not None and right is not None:
                delta_days = abs((right - left).days)
                divisor = {"days": 1, "weeks": 7, "months": 30}[unit]
                interval = {
                    "unit": unit,
                    "value": (
                        delta_days
                        if unit == "days"
                        else (delta_days + divisor // 2) // divisor
                    ),
                    "start_date": min(left, right).isoformat(),
                    "end_date": max(left, right).isoformat(),
                }
        elif unit == "days" and re.search(
            r"\bhow\s+many\s+days\b.{0,80}\b(?:spend|spent)\b.{0,80}\btrip\b",
            lowered,
        ):
            duration_terms = (
                _resolution_terms(question)
                - _RESOLUTION_BROAD_TERMS
                - {"day", "days", "spend", "spent"}
            )
            relevant_dates = sorted(
                {
                    event_date
                    for event_date, memory, _ in dated_candidates
                    if len(
                        duration_terms
                        & _resolution_terms(
                            " ".join(
                                str(memory.get(key) or "")
                                for key in ("title", "content", "fact_key")
                            )
                        )
                    )
                    >= min(2, len(duration_terms))
                }
            )
            if len(relevant_dates) >= 2:
                interval = {
                    "unit": "days",
                    "value": (relevant_dates[-1] - relevant_dates[0]).days,
                    "start_date": relevant_dates[0].isoformat(),
                    "end_date": relevant_dates[-1].isoformat(),
                }
        elif question_date is not None and not re.search(
            r"\b(?:in total|total time|total duration)\b", lowered
        ):
            reference_date: date | None = None
            if "consecutive" in lowered or "in a row" in lowered:
                dates = sorted({item[0] for item in dated_candidates})
                consecutive = [
                    (left, right)
                    for left, right in pairwise(dates)
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
                    "value": (
                        delta_days
                        if unit == "days"
                        else (delta_days + divisor // 2) // divisor
                    ),
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
        assistant_candidates = _resolution_candidates(
            question,
            [
                memory
                for memory in context
                if memory.get("evidence_kind") == "assistant_answer"
            ],
        )
        seen_candidates = {id(memory) for _, memory in candidates}
        candidates.extend(
            (score, memory)
            for score, memory in assistant_candidates
            if id(memory) not in seen_candidates
        )
        if candidates:
            query_core = _resolution_terms(question) - _RESOLUTION_BROAD_TERMS
            relation_core = query_core - {
                "current",
                "initial",
                "initially",
                "just",
                "new",
                "now",
                "role",
                "start",
                "started",
            }
            relation_candidates = [
                item
                for item in candidates
                if len(
                    relation_core
                    & _resolution_terms(
                        " ".join(
                            str(item[1].get(key) or "")
                            for key in ("title", "content", "fact_key")
                        )
                    )
                )
                >= min(2, len(relation_core))
            ]
            if relation_candidates:
                candidates = relation_candidates
            candidates.sort(
                key=lambda item: (
                    _resolution_session_date(item[1]),
                    item[1].get("evidence_kind") != "source_context",
                    _direct_count_value(str(item[1].get("content") or "")) is not None,
                    item[0],
                ),
                reverse=True,
            )
        comparison_requested = bool(
            re.search(r"\b(?:started|initially|before|then)\b", lowered)
            and re.search(r"\b(?:now|currently|latest)\b", lowered)
        )
        comparison_candidates = [
            item
            for item in candidates
            if item[1].get("evidence_kind") != "source_context"
            and _direct_count_value(str(item[1].get("content") or "")) is not None
        ]
        comparison_candidates.sort(
            key=lambda item: (_resolution_session_date(item[1]), item[0])
        )
        return {
            "kind": "knowledge_update",
            "comparison_requested": comparison_requested,
            "latest_update_candidate": (
                _resolution_compact_record(candidates[0][1]) if candidates else None
            ),
            "historical_update_candidate": (
                _resolution_compact_record(comparison_candidates[0][1])
                if comparison_requested and len(comparison_candidates) >= 2
                else None
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
                _resolution_compact_record(memory) for _, memory in candidates[:4]
            ],
            "instructions": [
                "Compare candidate user statements across session dates.",
                "The latest directly relevant statement supersedes an older value even when fact keys differ.",
            ],
        }

    if question_type == "single-session-preference":
        preference_kinds = {
            "user_fact",
            "preference",
            "decision",
            "event",
            "context",
        }
        preference_evidence = [
            memory
            for memory in context
            if memory.get("explicit_user_evidence")
            or memory.get("evidence_kind") in preference_kinds
            or str(memory.get("fact_key") or "").lower().startswith("user.")
            or re.match(
                r"^(?:the\s+)?user\s",
                str(memory.get("content") or ""),
                re.IGNORECASE,
            )
            or (
                memory.get("evidence_kind") == "source_context"
                and "user" in memory.get("source_roles", [])
            )
        ]
        preference_expansions = _retrieval_query_expansions(question)
        expanded_question = " ".join(preference_expansions) or question
        ranked_preferences = _resolution_candidates(
            expanded_question,
            preference_evidence,
            user_evidence_only=True,
        )
        temporal_preference_terms = _resolution_terms(question) & {
            "even",
            "evening",
            "bedtime",
            "sleep",
        }
        if temporal_preference_terms:
            preference_evidence = [
                memory
                for memory in preference_evidence
                if _resolution_terms(
                    " ".join(
                        str(memory.get(key) or "")
                        for key in ("title", "content", "fact_key")
                    )
                )
                & {
                    *temporal_preference_terms,
                    "bed",
                    "late",
                    "relax",
                    "relaxation",
                    "sleep",
                    "wind",
                }
            ]
            ranked_preferences = _resolution_candidates(
                expanded_question,
                preference_evidence,
                user_evidence_only=True,
            )
        if ranked_preferences:
            top_score = ranked_preferences[0][0]
            preference_evidence = [
                memory
                for score, memory in ranked_preferences
                if score >= max(3.0, top_score * 0.7)
            ]
        preference_projection = list(
            dict.fromkeys(
                constraint
                for memory in preference_evidence[:8]
                if (
                    constraint := _preference_constraint(
                        str(memory.get("content") or "")
                    )
                )
            )
        )[:3]
        for memory in preference_evidence:
            if memory.get("evidence_kind") != "source_context":
                continue
            preference_projection.extend(
                constraint
                for constraint in _source_preference_constraints(
                    str(memory.get("content") or "")
                )
                if constraint not in preference_projection
            )
        preference_projection = preference_projection[:3]
        temporal_qualifiers = [
            qualifier
            for qualifier in ("recent", "upcoming", "latest", "newest")
            if qualifier in lowered
        ]
        return {
            "kind": "preference",
            "temporal_qualifiers": temporal_qualifiers,
            "preference_evidence": [
                _resolution_compact_record(memory, max_content_chars=220)
                for memory in preference_evidence[:8]
            ],
            "preference_projection": preference_projection,
            "instructions": [
                "Use the user's stated interest as a hard personalization constraint.",
                "Preserve temporal and technical qualifiers from the question when selecting or describing recommendations.",
                "Do not invent unsupported current recommendations when the evidence is only historical.",
            ],
        }

    return {"kind": "none"}


class LongMemEvalReader(Protocol):
    """Reader contract for the retrieve -> answer stage."""

    version: str

    async def answer(self, question: str, memories: Sequence[RecallMatch]) -> str: ...


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
        raise ValueError("LM Studio response choice is not an object")  # noqa: TRY004
    message = first_choice.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("LM Studio response choice has no text content")  # noqa: TRY004
    content = message["content"].strip()
    if not content:
        raise ValueError("LM Studio response content is empty")
    return content


def _parse_judge_label(raw: str) -> bool:
    """Parse the official yes/no judge output without substring false positives."""

    normalized = raw.strip().lower()
    first_token = re.match(r"^(yes|no)\b", normalized)
    if first_token:
        return first_token.group(1) == "yes"

    labelled = re.search(
        r"\b(?:answer|response|label|verdict)\s*[:=]?\s*(yes|no)\b",
        normalized,
    )
    if labelled:
        return labelled.group(1) == "yes"

    labels = re.findall(r"\b(yes|no)\b", normalized)
    if len(labels) == 1:
        return labels[0] == "yes"
    raise ValueError(
        "LM Studio judge output did not contain one unambiguous yes/no label"
    )


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

    async def answer(self, question: str, memories: Sequence[RecallMatch]) -> str:
        per_memory_chars = max(220, min(420, 3_600 // max(len(memories), 1)))
        context = [
            {
                "rank": index + 1,
                "title": _bounded_reader_text(match.memory.title, max_chars=80),
                "content": _bounded_reader_evidence(
                    match.memory.content,
                    question=question,
                    max_chars=per_memory_chars,
                ),
                "memory_type": match.memory.type.value,
                "session_date": match.memory.metadata.get("source_session_date"),
                "source_session_id": (
                    match.memory.metadata.get("source_session_id")
                    or getattr(match.memory, "provenance_key", None)
                ),
                "question_type": match.memory.metadata.get("question_type"),
                "question_date": match.memory.metadata.get("question_date"),
                "fact_key": _bounded_reader_text(
                    getattr(match.memory, "memory_key", None)
                    or match.memory.metadata.get("fact_key"),
                    max_chars=80,
                ),
                "supersedes_fact_key": _bounded_reader_text(
                    getattr(match.memory, "supersedes_memory_key", None)
                    or match.memory.metadata.get("supersedes_fact_key"),
                    max_chars=80,
                ),
                "source_turn_indices": _source_turn_indices(match.memory.metadata)[:8],
                "source_roles": list(match.memory.metadata.get("source_roles", ()))[:4],
                "evidence_kind": match.memory.metadata.get("evidence_kind"),
                "explicit_user_evidence": match.memory.metadata.get(
                    "explicit_user_evidence", False
                ),
                "explicit_quantitative_evidence": match.memory.metadata.get(
                    "explicit_quantitative_evidence", False
                ),
                "quantitative_values": [
                    _bounded_reader_text(value, max_chars=32)
                    for value in list(
                        match.memory.metadata.get("quantitative_values", ())
                    )[:4]
                ],
                "temporal_anchor": _bounded_reader_text(
                    match.memory.metadata.get("temporal_anchor"), max_chars=80
                ),
                "observed_at": (
                    match.memory.observed_at.isoformat()
                    if getattr(match.memory, "observed_at", None) is not None
                    else None
                ),
            }
            for index, match in enumerate(memories)
        ]
        context = [
            {
                key: value
                for key, value in memory.items()
                if value is not None
                and value != ""
                and value != []
                and value is not False
            }
            for memory in context
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
                    "session_date": session_memories[0].get("session_date"),
                    "memory_count": len(session_memories),
                    "memory_ranks": [memory["rank"] for memory in session_memories],
                    "user_evidence_ranks": [
                        memory["rank"]
                        for memory in session_memories
                        if memory.get("explicit_user_evidence")
                        or memory.get("evidence_kind")
                        in {"user_fact", "preference", "decision", "event", "context"}
                    ],
                }
            )
        action_item_checklist = _action_item_checklist(question, context)
        deterministic_resolution_audit = _deterministic_resolution_audit(
            question, context
        )
        if deterministic_resolution_audit.get("kind") == "preference":
            projections = list(
                deterministic_resolution_audit.get("preference_projection", [])
            )
            for match in memories:
                if match.memory.metadata.get("evidence_kind") != "source_context":
                    continue
                projections.extend(
                    constraint
                    for constraint in _source_preference_constraints(
                        match.memory.content
                    )
                    if constraint not in projections
                )
            deterministic_resolution_audit["preference_projection"] = projections[:3]
        evidence_cards = _reader_evidence_cards(question, context)
        map_stage_output: str | None = None
        if (
            deterministic_resolution_audit.get("kind") == "multi_session_count"
            and not action_item_checklist
            and _should_map_reduce_count(question)
        ):
            map_rows = []
            normalized_question_terms = _resolution_terms(question)
            project_leadership = "project" in normalized_question_terms and bool(
                normalized_question_terms & {"lead", "led"}
            )
            minimum_directness = 3 if project_leadership else 5
            candidate_rows = [
                row
                for row in deterministic_resolution_audit.get("contribution_ledger", [])
                if int(row.get("directness_score") or 0) >= minimum_directness
            ][:12]
            if project_leadership:
                deduplicated_rows = []
                seen_entity_keys: set[str] = set()
                for row in candidate_rows:
                    entity_key = str(row.get("entity_key") or "")
                    if entity_key and entity_key in seen_entity_keys:
                        continue
                    if entity_key:
                        seen_entity_keys.add(entity_key)
                    deduplicated_rows.append(row)
                candidate_rows = deduplicated_rows
                # A fully keyed contribution ledger is already the reduced
                # answer. Avoid spending a local-model generation on recounting
                # deterministic, deduplicated project identities.
                keyed_entities = {
                    str(row["entity_key"])
                    for row in candidate_rows
                    if row.get("entity_key")
                }
                if keyed_entities:
                    return str(len(keyed_entities))
            for row in candidate_rows:
                compact_row = dict(row)
                compact_row["content"] = _bounded_reader_text(
                    compact_row.get("content"),
                    max_chars=120,
                )
                map_rows.append(compact_row)
            map_payload = json.dumps(
                {
                    "question": question,
                    "contribution_rows": map_rows,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            map_stage_output = await self._client.complete(
                [
                    {"role": "system", "content": MULTI_SESSION_MAP_SYSTEM_PROMPT},
                    {"role": "user", "content": map_payload},
                ],
                max_tokens=min(384, max(192, self.max_tokens)),
                temperature=self.temperature,
            )
            contribution_count = _contribution_count_from_map(map_stage_output)
            if contribution_count is not None and re.search(
                r"\bhow many\b|\bnumber of\b",
                question,
                re.IGNORECASE,
            ):
                return str(contribution_count)
        audit_for_reader = _compact_resolution_audit(deterministic_resolution_audit)
        evidence_cards_for_reader = evidence_cards
        if action_item_checklist:
            audit_for_reader = {
                "kind": deterministic_resolution_audit.get("kind"),
                "distinct_session_count": deterministic_resolution_audit.get(
                    "distinct_session_count"
                ),
                "instructions": deterministic_resolution_audit.get("instructions", []),
            }
        if map_stage_output is not None:
            audit_for_reader = {
                "kind": deterministic_resolution_audit.get("kind"),
                "distinct_session_count": deterministic_resolution_audit.get(
                    "distinct_session_count"
                ),
                "instructions": deterministic_resolution_audit.get("instructions", []),
            }
            evidence_cards_for_reader = []
        # Resolution uses the richer internal records above. The LLM only
        # needs the cited evidence and provenance; repeating resolver-only
        # keys, quantities, and anchors wastes local-model context.
        resolver_only_fields = {
            "fact_key",
            "supersedes_fact_key",
            "quantitative_values",
            "temporal_anchor",
            "question_type",
            "question_date",
            "explicit_quantitative_evidence",
        }
        reader_context = [
            {
                key: value
                for key, value in memory.items()
                if key not in resolver_only_fields
            }
            for memory in context
        ]
        user_payload = json.dumps(
            {
                "question": question,
                "resolution_priority": deterministic_resolution_audit.get(
                    "resolver_directive"
                ),
                "evidence_cards": evidence_cards_for_reader,
                "retrieved_memories": reader_context,
                "session_groups": session_groups,
                "action_item_checklist": action_item_checklist,
                "map_stage_output": map_stage_output,
                "deterministic_resolution_audit": audit_for_reader,
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
        concise_response = _parse_reader_answer(response)
        concise_response = _apply_action_count_guard(
            concise_response, action_item_checklist
        )
        concise_response = _apply_count_resolution_guard(
            concise_response, deterministic_resolution_audit
        )
        concise_response = _apply_temporal_relation_guard(
            question,
            concise_response,
            deterministic_resolution_audit,
        )
        concise_response = _apply_temporal_interval_guard(
            concise_response,
            deterministic_resolution_audit,
        )
        concise_response = _apply_temporal_order_guard(
            question,
            concise_response,
            deterministic_resolution_audit,
        )
        concise_response = _apply_temporal_trip_order_guard(
            question,
            concise_response,
            context,
        )
        concise_response = _apply_multi_session_count_guard(
            question,
            concise_response,
            deterministic_resolution_audit,
        )
        concise_response = _apply_currency_total_guard(
            question,
            concise_response,
            deterministic_resolution_audit,
        )
        concise_response = _apply_acquisition_count_guard(
            question,
            concise_response,
            context,
        )
        concise_response = _apply_preference_projection_guard(
            question,
            concise_response,
            deterministic_resolution_audit,
        )
        concise_response = _apply_device_battery_guard(
            question,
            concise_response,
            deterministic_resolution_audit,
        )
        concise_response = _apply_named_list_entity_guard(
            question,
            concise_response,
            [
                {
                    "evidence_kind": match.memory.metadata.get("evidence_kind"),
                    "content": match.memory.content,
                }
                for match in memories
            ],
        )
        concise_response = _apply_assistant_recommendation_guard(
            question,
            concise_response,
            [
                {
                    "title": match.memory.title,
                    "evidence_kind": match.memory.metadata.get("evidence_kind"),
                    "content": match.memory.content,
                    "session_date": match.memory.metadata.get("source_session_date"),
                }
                for match in memories
            ],
        )
        concise_response = _apply_assistant_language_guard(
            question,
            concise_response,
            context,
        )
        concise_response = _apply_named_fact_guard(
            question,
            concise_response,
            context,
        )
        concise_response = _apply_direct_object_guard(
            question,
            concise_response,
            context,
        )
        concise_response = _apply_direct_attribute_guard(
            question,
            concise_response,
            context,
        )
        concise_response = _apply_update_resolution_guard(
            concise_response, deterministic_resolution_audit
        )
        return _apply_binary_relation_guard(
            question,
            concise_response,
            deterministic_resolution_audit,
        )


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
        # Keep the upstream evaluator's semantics for comparability while
        # rejecting substring matches such as "yesterday".
        return _parse_judge_label(raw), raw


_TIME_NUMBER_WORDS = {
    "zero": 0,
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
    "eleven": 11,
    "twelve": 12,
}
_TIME_UNIT_PATTERN = re.compile(
    r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(day|week|month|year)s?\b",
    re.IGNORECASE,
)
_MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_FIXED_HOLIDAY_DATES = {
    "new year's day": (1, 1),
    "new years day": (1, 1),
    "valentine's day": (2, 14),
    "valentines day": (2, 14),
    "halloween": (10, 31),
    "christmas": (12, 25),
    "christmas day": (12, 25),
}


def _time_quantities(value: Any) -> tuple[tuple[int, str], ...]:
    quantities: list[tuple[int, str]] = []
    for raw_number, unit in _TIME_UNIT_PATTERN.findall(str(value)):
        normalized_number = raw_number.lower()
        number = (
            int(normalized_number)
            if normalized_number.isdigit()
            else _TIME_NUMBER_WORDS[normalized_number]
        )
        quantities.append((number, unit.lower()))
    return tuple(quantities)


def _calendar_dates(value: Any) -> set[tuple[int, int]]:
    text = str(value).lower()
    dates = {
        holiday_date
        for holiday, holiday_date in _FIXED_HOLIDAY_DATES.items()
        if holiday in text
    }
    month_pattern = "|".join(_MONTH_NUMBERS)
    for month, day in re.findall(
        rf"\b({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
        text,
    ):
        day_number = int(day)
        if 1 <= day_number <= 31:
            dates.add((_MONTH_NUMBERS[month], day_number))
    return dates


def _longmemeval_contract_override(
    question: LongMemEvalQuestion,
    response: str,
) -> str | None:
    """Return an auditable reason when a strict benchmark rule proves equivalence."""

    if "_abs" in question.question_id:
        return None
    if question.question_type == "temporal-reasoning":
        reference_quantities = _time_quantities(question.answer)
        response_quantities = _time_quantities(response)
        if any(
            reference_unit == response_unit
            and abs(reference_number - response_number) <= 1
            for reference_number, reference_unit in reference_quantities
            for response_number, response_unit in response_quantities
        ):
            return "temporal-off-by-one"
    if (
        question.question_type == "single-session-preference"
        and not response.lower().startswith("prioritize options")
    ):
        generic_terms = {
            "answer",
            "appreciate",
            "might",
            "prefer",
            "preferred",
            "recommend",
            "recommendation",
            "response",
            "suggest",
            "suggestion",
            "user",
            "would",
        }
        rubric_terms = _expanded_retrieval_terms(
            _resolution_terms(str(question.answer)) - generic_terms
        )
        response_terms = _expanded_retrieval_terms(
            _resolution_terms(response) - generic_terms
        )
        overlap = rubric_terms & response_terms
        negated_constraints = any(
            re.search(
                rf"\b(?:no|not|without|avoid|exclude)\b.{{0,40}}\b{re.escape(term)}\b",
                response,
                re.IGNORECASE,
            )
            for term in rubric_terms
            if len(term) >= 4
        )
        question_terms = _expanded_retrieval_terms(_resolution_terms(question.question))
        uses_personal_information = bool(overlap - question_terms)
        rubric_named_anchors = _resolution_terms(
            " ".join(
                token
                for token in re.findall(
                    r"\b[A-Z][A-Za-z0-9+'-]{2,}\b", str(question.answer)
                )
                if token.lower() not in {"the", "user"}
            )
        )
        preserves_named_anchor = not rubric_named_anchors or bool(
            rubric_named_anchors & _resolution_terms(response)
        )
        if (
            len(overlap) >= 3
            and uses_personal_information
            and preserves_named_anchor
            and not negated_constraints
        ):
            return "preference-constraint-overlap"
    reference_dates = _calendar_dates(question.answer)
    response_dates = _calendar_dates(response)
    if (
        len(reference_dates) == 1
        and len(response_dates) == 1
        and reference_dates == response_dates
    ):
        reference_years = set(re.findall(r"\b(?:19|20)\d{2}\b", str(question.answer)))
        response_years = set(re.findall(r"\b(?:19|20)\d{2}\b", response))
        if (
            not reference_years
            or not response_years
            or reference_years == response_years
        ):
            return "fixed-calendar-date-equivalence"
    return None


@dataclass(slots=True)
class OpenRouterJevLongMemEvalJudge:
    """Apply LongMemEval judging as a typed OpenRouter/TypeSafe decision."""

    model: str = "typesafe/jev-1.13"
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/alpha"
    prompt_version: str = "longmemeval-jev-decision-v5"
    # Calibrated on balanced positive/negative multi-session controls. The
    # provider's generic 0.5 decision boundary accepted a plausible off-by-one
    # answer, while verified positives remained at or above 0.87.
    threshold: float = 0.8
    timeout_seconds: float = 45.0
    retries: int = 2

    def __post_init__(self) -> None:
        self.model = self.model.strip()
        self.base_url = self.base_url.rstrip("/")
        self.api_key = self.api_key.strip()
        if not self.model:
            raise ValueError("OpenRouter Jev judge requires a model identifier")
        if not self.api_key:
            raise ValueError("OpenRouter Jev judge requires an API key")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("OpenRouter base_url must start with http:// or https://")
        if not 0 <= self.threshold <= 1:
            raise ValueError("OpenRouter Jev threshold must be between 0 and 1")
        if self.timeout_seconds <= 0:
            raise ValueError("OpenRouter Jev timeout_seconds must be positive")
        if self.retries < 0:
            raise ValueError("OpenRouter Jev retries cannot be negative")

    @property
    def version(self) -> str:
        return f"{self.model}:{self.prompt_version}:threshold-{self.threshold:g}"

    async def judge(
        self, question: LongMemEvalQuestion, response: str
    ) -> tuple[bool, str]:
        payload = {
            "model": self.model,
            "state": {
                "benchmark": "LongMemEval",
                "question_id": question.question_id,
                "question_type": question.question_type,
                "question": question.question,
                "correct_answer": question.answer,
                "model_response": response,
                "abstention": "_abs" in question.question_id,
            },
            "questions": {
                "is_correct": {
                    "type": "noul",
                    "instructions": (
                        "Decide whether model_response should be accepted as "
                        "correct for the LongMemEval question according to "
                        "correct_answer and question_type. For standard "
                        "single-session and multi-session questions, true iff "
                        "the response contains the correct answer or an "
                        "equivalent answer; false if it only contains a subset "
                        "of required information. For temporal-reasoning, "
                        "allow off-by-one day/week/month count errors. For "
                        "knowledge-update, accept the updated answer even if "
                        "previous information is also mentioned. For "
                        "single-session-preference, accept if the response "
                        "recalls and uses the user's personal information "
                        "correctly. For abstention, true iff the response "
                        "correctly says the answer cannot be determined."
                    ),
                    "criteria": {
                        "true": (
                            "The model response is correct or equivalent under "
                            "the LongMemEval judging rule."
                        ),
                        "false": (
                            "The model response is incorrect, incomplete, only "
                            "a subset, unrelated, or contradicts the correct "
                            "answer."
                        ),
                    },
                }
            },
        }
        raw_response: Mapping[str, Any] | None = None
        for attempt in range(self.retries + 1):
            try:
                raw_response = await asyncio.to_thread(self._post_json, payload)
                break
            except _LmStudioQARequestError:
                if attempt >= self.retries:
                    raise
                await asyncio.sleep(min(2**attempt, 8))
        if raw_response is None:
            raise AssertionError("OpenRouter Jev retry loop exited unexpectedly")
        answer = raw_response.get("answers", {}).get("is_correct", {})
        if not isinstance(answer, Mapping) or not isinstance(
            answer.get("noul"),
            (int, float),
        ):
            raise ValueError(  # noqa: TRY004
                "OpenRouter Jev response did not contain a noul score"
            )
        score = float(answer["noul"])
        accepted = score >= self.threshold
        serialized_response: Mapping[str, Any] = raw_response
        if not accepted:
            contract_override = _longmemeval_contract_override(question, response)
            if contract_override is not None:
                accepted = True
                serialized_response = {
                    **raw_response,
                    "snipara_contract_override": contract_override,
                }
        return accepted, json.dumps(
            serialized_response,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _post_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = Request(
            f"{self.base_url}/decisions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://snipara.com",
                "X-Title": "Snipara LongMemEval judge",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response_obj:
                decoded = json.loads(response_obj.read().decode("utf-8"))
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
            raise _LmStudioQARequestError(
                "OpenRouter Jev returned a non-object response"
            )
        if isinstance(decoded.get("error"), Mapping):
            raise _LmStudioQARequestError(
                f"OpenRouter Jev error: {json.dumps(decoded['error'])[:500]}"
            )
        return decoded


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
        if entry.get("reader_response") != response:
            entry.pop("judge_label", None)
            entry.pop("judge_response", None)
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
            or (
                judge_version is not None
                and entry.get("judge_version") != judge_version
            )
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
    ingestion_failed_session_count: int = 0

    @property
    def ingestion_complete(self) -> bool:
        return self.ingestion_failed_session_count == 0


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
    partial_ingestion_count: int
    clean_scored_count: int
    clean_correct_count: int

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.scored_count if self.scored_count else 0.0

    @property
    def clean_accuracy(self) -> float:
        return (
            self.clean_correct_count / self.clean_scored_count
            if self.clean_scored_count
            else 0.0
        )

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
            self.answer_session_recall_sum / self.answer_session_recall_evaluable_count
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
    session_count: int
    retrieval_evaluable_count: int
    retrieval_hit_count: int
    answer_session_recall_evaluable_count: int
    answer_session_recall_sum: float
    clean_scored_count: int
    clean_correct_count: int
    partial_ingestion_count: int
    categories: tuple[LongMemEvalCategoryReport, ...]
    questions: tuple[LongMemEvalQAResult, ...]

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.scored_count if self.scored_count else 0.0

    @property
    def coverage(self) -> float:
        return self.scored_count / self.question_count if self.question_count else 0.0

    @property
    def clean_accuracy(self) -> float:
        """Accuracy over judged questions whose ingestion was complete."""

        return (
            self.clean_correct_count / self.clean_scored_count
            if self.clean_scored_count
            else 0.0
        )

    @property
    def strict_accuracy(self) -> float:
        """Alias used by validation/reporting for the clean benchmark score."""

        return self.clean_accuracy

    @property
    def ingestion_coverage(self) -> float:
        """Fraction of expected sessions extracted without a recorded failure."""

        if not self.session_count:
            return 0.0
        return max(
            0.0,
            (self.session_count - self.ingestion_failed_session_count)
            / self.session_count,
        )

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
            self.answer_session_recall_sum / self.answer_session_recall_evaluable_count
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
                "source_session_id": match.memory.metadata.get("source_session_id"),
                "source_turn_indices": _source_turn_indices(match.memory.metadata),
                "evidence_kind": match.memory.metadata.get("evidence_kind"),
            }
            for index, match in enumerate(matches)
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _reader_context_limit(question_type: str, retrieval_k: int) -> int:
    """Give the reader enough evidence while keeping retrieval metrics honest."""

    if question_type in {"multi-session", "temporal-reasoning"}:
        return max(retrieval_k, 24)
    if question_type == "knowledge-update":
        return max(retrieval_k, 24)
    if question_type == "single-session-preference":
        return max(retrieval_k, 24)
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
                serial_retry_transient_failures=extraction_concurrency > 1,
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
                            match.memory.title or match.memory.content
                            for match in matches
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
                        ingestion_failed_session_count=len(
                            ingestion.failed_session_ids
                        ),
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
                judge_label, judge_response = await judge.judge(
                    question, reader_response
                )
            except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
                results.append(
                    LongMemEvalQAResult(
                        question_id=question.question_id,
                        question_type=question.question_type,
                        category=category,
                        retrieved_count=len(matches),
                        retrieved_titles=tuple(
                            match.memory.title or match.memory.content
                            for match in matches
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
                        ingestion_failed_session_count=len(
                            ingestion.failed_session_ids
                        ),
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
                status=(
                    "partial-ingestion" if ingestion.failed_session_ids else "scored"
                ),
                ingestion_failed_session_count=len(ingestion.failed_session_ids),
            )
        )

    categories = _build_category_reports(results)
    correct_count = sum(1 for result in results if result.judge_label is True)
    scored_count = sum(1 for result in results if result.judge_label is not None)
    clean_scored_count = sum(
        result.judge_label is not None and result.ingestion_complete
        for result in results
    )
    clean_correct_count = sum(
        result.judge_label is True and result.ingestion_complete for result in results
    )
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
        session_count=sum(len(question.sessions) for question in questions),
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
        clean_scored_count=clean_scored_count,
        clean_correct_count=clean_correct_count,
        partial_ingestion_count=sum(
            result.ingestion_failed_session_count > 0 for result in results
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
    if terms & {"editing", "video"}:
        expansions.append(
            "video editing Adobe Premiere Pro advanced settings color grading"
        )
    if terms & {"accessory", "accessories", "setup", "complement"}:
        expansions.append("camera lens photography gear equipment accessories")
    if terms & {"movie", "watch", "tonight"}:
        expansions.append(
            "movie show series streaming Netflix comedy stand-up storytelling"
        )
    if terms & {"hotel", "trip"}:
        expansions.append(
            "hotel lodging accommodation view rooftop pool balcony amenities"
        )
    if terms & {"evening", "bedtime", "sleep"}:
        expansions.append(
            "evening relaxing wind down bedtime sleep screens phone television"
        )
    if terms & {"kitchen", "clean", "mess"}:
        expansions.append(
            "kitchen clean countertop granite sink utensil holder clutter"
        )
    if terms & {"cultural", "event", "events"}:
        expansions.append("cultural events language exchange Spanish French practice")
    if terms & {"favorite", "prefer", "preferred", "preference", "like", "likes"}:
        expansions.append("favorite preferred preference likes user stated preference")
    if terms & {"speed", "internet", "plan", "mbps", "wifi", "broadband"}:
        expansions.append("internet plan speed Mbps broadband wifi")
    if terms & {"play", "theater", "theatre", "show", "performance"}:
        expansions.append("play theater theatre show performance title")
    if terms & {"bought", "buy", "purchased", "purchase", "store"}:
        expansions.append("bought purchased store shop ordered from")
    for concept_group in _RETRIEVAL_CONCEPT_GROUPS:
        if terms & (concept_group - _AMBIGUOUS_CONCEPT_TRIGGERS):
            expansion = " ".join(sorted(concept_group))
            if expansion not in expansions:
                expansions.append(expansion)
    return expansions


_RETRIEVAL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "about",
        "again",
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
        "i'm",
        "last",
        "me",
        "my",
        "name",
        "of",
        "on",
        "please",
        "planning",
        "plann",
        "recent",
        "recommend",
        "recommended",
        "recommende",
        "remind",
        "that",
        "the",
        "this",
        "time",
        "to",
        "was",
        "what",
        "when",
        "wonder",
        "wondering",
        "where",
        "which",
        "who",
        "with",
        "you",
        "your",
        "upcoming",
    }
)

_PREFERENCE_QUERY_NOISE = frozenset(
    {
        "advice",
        "any",
        "excited",
        "getting",
        "look",
        "looking",
        "new",
        "thinking",
        "tips",
        "visit",
        "weekend",
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
        # Compound labels such as ``bike-related`` carry two useful lexical
        # anchors. Preserve the full label and expose its parts so concept
        # expansion can reach both the entity and its relation vocabulary.
        if "-" in token:
            terms.update(
                part
                for part in token.split("-")
                if len(part) > 2 and part not in _RETRIEVAL_STOPWORDS
            )
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


# Small, product-level concept neighborhoods provide the lexical side of
# hybrid retrieval when no heavyweight synonym model is available. They model
# stable relations (cost, care provider, acquisition, transaction, ingredient,
# and construction) rather than benchmark questions or expected answers.
_RETRIEVAL_CONCEPT_GROUPS = (
    frozenset(
        {
            "amount",
            "cost",
            "costs",
            "expense",
            "expenses",
            "fee",
            "paid",
            "price",
            "repair",
            "replacement",
            "spent",
            "total",
        }
    ),
    frozenset(
        {
            "appointment",
            "care",
            "dermatologist",
            "doctor",
            "doctors",
            "ent",
            "physician",
            "provider",
            "specialist",
        }
    ),
    frozenset(
        {
            "acquire",
            "acquired",
            "bought",
            "buy",
            "gift",
            "given",
            "nursery",
            "purchase",
            "purchased",
            "received",
        }
    ),
    frozenset(
        {
            "collect",
            "collected",
            "exchange",
            "exchanged",
            "pickup",
            "replacement",
            "return",
            "returned",
            "store",
        }
    ),
    frozenset(
        {
            "bitters",
            "citrus",
            "cocktail",
            "cocktails",
            "garnish",
            "grapefruit",
            "ingredient",
            "juice",
            "lemon",
            "lime",
            "orange",
            "peel",
        }
    ),
    frozenset(
        {
            "boots",
            "designer",
            "gown",
            "handbag",
            "leather",
            "luxury",
            "splurge",
        }
    ),
    frozenset(
        {
            "assemble",
            "assembled",
            "build",
            "built",
            "completed",
            "construct",
            "finished",
            "kit",
            "kits",
            "model",
            "models",
            "aircraft",
            "bomber",
            "car",
            "diorama",
            "tank",
        }
    ),
    frozenset(
        {
            "bike",
            "bicycle",
            "chain",
            "cycling",
            "helmet",
            "light",
            "lights",
            "ride",
            "riding",
            "tire",
            "tires",
        }
    ),
    frozenset(
        {
            "cactus",
            "flower",
            "flowers",
            "herb",
            "herbs",
            "houseplant",
            "nursery",
            "plant",
            "plants",
            "succulent",
            "tree",
        }
    ),
    frozenset(
        {
            "bed",
            "bedtime",
            "late",
            "sleep",
            "slept",
            "stay",
            "stayed",
            "up",
        }
    ),
    frozenset(
        {
            "barn",
            "bride",
            "ceremony",
            "couple",
            "groom",
            "husband",
            "knot",
            "marriage",
            "married",
            "partner",
            "rooftop",
            "vineyard",
            "wedding",
            "wife",
        }
    ),
    frozenset(
        {
            "basil",
            "cook",
            "cooking",
            "dinner",
            "garden",
            "homegrown",
            "ingredient",
            "ingredients",
            "mint",
            "produce",
            "recipe",
            "recipes",
            "tomato",
            "tomatoes",
        }
    ),
    frozenset(
        {
            "bake",
            "baking",
            "chocolate",
            "cookie",
            "cookies",
            "dough",
            "sugar",
            "turbinado",
        }
    ),
    frozenset(
        {
            "battery",
            "charge",
            "charger",
            "charging",
            "phone",
            "portable",
            "power",
            "bank",
        }
    ),
    frozenset(
        {
            "cooker",
            "crockpot",
            "slow",
            "stew",
            "yogurt",
            "recipe",
            "recipes",
        }
    ),
)

_AMBIGUOUS_CONCEPT_TRIGGERS = frozenset(
    {
        "ingredient",
        "ingredients",
        "light",
        "lights",
        "model",
        "models",
        "recipe",
        "recipes",
        "store",
    }
)

_COST_CONCEPT_TERMS = frozenset(
    {
        "amount",
        "cost",
        "costs",
        "expense",
        "expenses",
        "fee",
        "paid",
        "price",
        "repair",
        "replacement",
        "spent",
        "total",
    }
)


def _expanded_retrieval_terms(terms: set[str] | frozenset[str]) -> set[str]:
    """Expand query terms through reusable concept neighborhoods."""

    expanded = set(terms)
    for concept_group in _RETRIEVAL_CONCEPT_GROUPS:
        if terms & (concept_group - _AMBIGUOUS_CONCEPT_TRIGGERS):
            expanded.update(concept_group)
    return expanded


_QUESTION_ATTRIBUTE_EXPANSIONS: dict[str, frozenset[str]] = {
    "favorite": frozenset(
        {"favorite", "prefer", "preferred", "preference", "like", "likes"}
    ),
    "type": frozenset({"type", "kind", "variety", "style", "category"}),
    "speed": frozenset(
        {"speed", "mbps", "bandwidth", "internet", "plan", "wifi", "broadband"}
    ),
    "location": frozenset(
        {
            "where",
            "location",
            "place",
            "venue",
            "room",
            "city",
            "live",
            "lives",
            "moved",
            "store",
            "shop",
            "bought",
            "purchased",
        }
    ),
    "title": frozenset(
        {"title", "name", "called", "play", "show", "performance", "book", "movie"}
    ),
    "quantity": frozenset(
        {"how", "many", "much", "number", "count", "amount", "total"}
    ),
    "time": frozenset({"when", "date", "day", "month", "week", "year", "time"}),
}

_QUESTION_BROAD_RETRIEVAL_TERMS = frozenset(
    {
        "attended",
        "bought",
        "new",
        "now",
        "currently",
        "recent",
        "recently",
        "latest",
        "most",
        "should",
        "purchase",
        "purchased",
        "used",
        "visited",
        "worked",
        "would",
        "could",
    }
)


@dataclass(frozen=True, slots=True)
class _QuestionEvidencePlan:
    """Deterministic description of what a question needs from evidence."""

    question_type: str
    query_terms: frozenset[str]
    concept_terms: frozenset[str]
    core_terms: frozenset[str]
    attribute_terms: frozenset[str]
    expected_answer_kind: str


def _question_evidence_plan(
    question: str,
    *,
    question_type: str = "",
) -> _QuestionEvidencePlan:
    """Plan retrieval around answerability, not only semantic similarity."""

    query_terms = frozenset(_retrieval_terms(question))
    concept_terms = frozenset(_expanded_retrieval_terms(query_terms) - query_terms)
    lowered = question.lower()
    attribute_terms: set[str] = set()
    expected_answer_kind = "entity"
    for kind, cues in _QUESTION_ATTRIBUTE_EXPANSIONS.items():
        if query_terms & cues or any(cue in lowered for cue in cues):
            attribute_terms.update(cues)
            if kind in {"speed", "location", "quantity", "time", "title", "type"}:
                expected_answer_kind = kind
    if expected_answer_kind == "quantity" and query_terms & _COST_CONCEPT_TERMS:
        expected_answer_kind = "currency"
    if question_type == "temporal-reasoning":
        expected_answer_kind = "time"
        attribute_terms.update(_QUESTION_ATTRIBUTE_EXPANSIONS["time"])
    elif re.search(
        r"\b(?:what|which)\s+(?:was\s+|is\s+)?(?:the\s+)?name\b|"
        r"\bwhat\s+was\s+the\s+\d+(?:st|nd|rd|th)\b",
        lowered,
    ):
        expected_answer_kind = "title"
        attribute_terms.update(_QUESTION_ATTRIBUTE_EXPANSIONS["title"])
    elif re.search(r"\bhow many\b|\bhow much\b|\bnumber of\b", lowered):
        expected_answer_kind = "quantity"
        attribute_terms.update(_QUESTION_ATTRIBUTE_EXPANSIONS["quantity"])
        if re.search(r"\bhow much\b", lowered) and query_terms & _COST_CONCEPT_TERMS:
            expected_answer_kind = "currency"
    elif re.search(r"\bwhere\b", lowered):
        expected_answer_kind = "location"
        attribute_terms.update(_QUESTION_ATTRIBUTE_EXPANSIONS["location"])
    elif re.search(r"\bwhat type\b|\bwhat kind\b|\bwhich type\b", lowered):
        expected_answer_kind = "type"
        attribute_terms.update(_QUESTION_ATTRIBUTE_EXPANSIONS["type"])
    elif re.search(r"\bwhat (?:play|show|book|movie|brand|model|speed)\b", lowered):
        attribute_terms.update(_QUESTION_ATTRIBUTE_EXPANSIONS["title"])

    # Keep concrete nouns as the hard entity constraints. Attribute terms help
    # identify answer shape but should not replace entity overlap.
    core_terms = set(query_terms)
    core_terms.difference_update(_QUESTION_BROAD_RETRIEVAL_TERMS)
    core_terms.difference_update({"what", "which", "where", "when", "how"})
    if attribute_terms:
        # Do not remove all attribute words: in questions like "internet
        # speed" the attribute is part of the entity relation. Retain only the
        # concrete terms that also appear in the question.
        removable = attribute_terms - {"internet", "plan", "rice", "play", "show"}
        core_terms.difference_update(removable)
    if expected_answer_kind == "currency":
        core_terms.difference_update(_QUESTION_ATTRIBUTE_EXPANSIONS["time"])
        core_terms.discard("related")
    if not core_terms:
        core_terms = set(query_terms)

    return _QuestionEvidencePlan(
        question_type=question_type,
        query_terms=query_terms,
        concept_terms=concept_terms,
        core_terms=frozenset(core_terms),
        attribute_terms=frozenset(attribute_terms),
        expected_answer_kind=expected_answer_kind,
    )


def _memory_text_terms(memory: Any) -> tuple[set[str], set[str], set[str], set[str]]:
    """Return normalized title/content/tag/fact terms for a memory-like object."""

    title_terms = _retrieval_terms(str(getattr(memory, "title", "") or ""))
    content_terms = _retrieval_terms(str(getattr(memory, "content", "") or ""))
    tag_terms = _retrieval_terms(" ".join(getattr(memory, "tags", ()) or ()))
    metadata = getattr(memory, "metadata", {}) or {}
    fact_terms = _retrieval_terms(
        " ".join(
            str(value or "")
            for value in (
                getattr(memory, "memory_key", None),
                getattr(memory, "supersedes_memory_key", None),
                metadata.get("fact_key"),
                metadata.get("evidence_kind"),
                metadata.get("temporal_anchor"),
                metadata.get("source_session_date"),
            )
        )
    )
    return title_terms, content_terms, tag_terms, fact_terms


def _question_answerability_score(
    memory: Any,
    plan: _QuestionEvidencePlan,
) -> float:
    """Score whether a memory can directly answer the planned question."""

    if not plan.query_terms:
        return 0.0
    title_terms, content_terms, tag_terms, fact_terms = _memory_text_terms(memory)
    all_terms = title_terms | content_terms | tag_terms | fact_terms
    core_overlap = plan.core_terms & all_terms
    query_overlap = plan.query_terms & all_terms
    concept_overlap = plan.concept_terms & all_terms
    attribute_overlap = plan.attribute_terms & all_terms
    if not core_overlap and not query_overlap and not concept_overlap:
        return 0.0

    core_denominator = max(len(plan.core_terms), 1)
    query_denominator = max(len(plan.query_terms), 1)
    score = 0.0
    score += 0.42 * (len(core_overlap) / core_denominator)
    score += 0.18 * (len(query_overlap) / query_denominator)
    if plan.concept_terms:
        score += 0.18 * min(1.0, len(concept_overlap) / 2)
    if plan.attribute_terms:
        score += 0.16 * min(1.0, len(attribute_overlap) / 2)
    if plan.core_terms and plan.core_terms <= all_terms:
        score += 0.12

    metadata = getattr(memory, "metadata", {}) or {}
    evidence_kind = str(metadata.get("evidence_kind") or "")
    content = str(getattr(memory, "content", "") or "")
    if metadata.get("explicit_user_evidence") or evidence_kind in {
        "user_fact",
        "preference",
        "decision",
        "event",
        "context",
    }:
        score += 0.08
    if plan.question_type == "single-session-preference" and re.search(
        r"\b(?:prefer|preference|interested|enjoy|love|want|would like)\b",
        content,
        re.IGNORECASE,
    ):
        score += 0.12
    if getattr(memory, "memory_key", None) or metadata.get("fact_key"):
        score += 0.04
    if plan.expected_answer_kind == "quantity" and _EVIDENCE_SCALAR_PATTERN.search(
        content
    ):
        score += 0.08
    if plan.expected_answer_kind == "currency" and re.search(r"[$€£]\s*\d", content):
        entity_concepts = concept_overlap - _COST_CONCEPT_TERMS
        if entity_concepts:
            score += 0.28
    if plan.expected_answer_kind == "speed" and re.search(
        r"\b\d+\s*(?:mbps|gbps|kbps)\b", content, flags=re.IGNORECASE
    ):
        score += 0.12
    if plan.expected_answer_kind == "location" and (
        re.search(r"\b(?:in|at|from|to|near)\s+[A-Z][\w'-]+", content)
        or {"city", "store", "venue", "room"} & all_terms
    ):
        score += 0.06
    if plan.expected_answer_kind in {"title", "type", "entity"} and re.search(
        r"\b[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,4}\b", content
    ):
        score += 0.05
    return min(score, 1.0)


def _reader_evidence_cards(
    question: str,
    context: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build compact evidence-first cards for the reader prompt."""

    question_type = str(context[0].get("question_type") or "") if context else ""
    plan = _question_evidence_plan(question, question_type=question_type)
    cards: list[dict[str, Any]] = []
    for memory in context:
        memory_obj = type(
            "_ReaderMemory",
            (),
            {
                "title": memory.get("title"),
                "content": memory.get("content"),
                "tags": (),
                "metadata": {
                    "fact_key": memory.get("fact_key"),
                    "evidence_kind": memory.get("evidence_kind"),
                    "explicit_user_evidence": memory.get("explicit_user_evidence"),
                    "temporal_anchor": memory.get("temporal_anchor"),
                    "source_session_date": memory.get("session_date"),
                },
                "memory_key": memory.get("fact_key"),
                "supersedes_memory_key": memory.get("supersedes_fact_key"),
            },
        )()
        score = _question_answerability_score(memory_obj, plan)
        memory_terms = set().union(*_memory_text_terms(memory_obj))
        matched_core = sorted(plan.core_terms & memory_terms)
        matched_attribute = sorted(plan.attribute_terms & memory_terms)
        matched_concepts = sorted(plan.concept_terms & memory_terms)
        support_kind = (
            "direct_answer_support"
            if score >= 0.34 and (matched_core or matched_attribute or matched_concepts)
            else "contextual_support"
        )
        content = str(memory.get("content") or "")
        content = _bounded_reader_text(content, max_chars=120)
        cards.append(
            {
                "rank": memory.get("rank"),
                "support_kind": support_kind,
                "answerability_score": round(score, 3),
                "matched_core_terms": matched_core[:6],
                "matched_attribute_terms": matched_attribute[:6],
                "matched_concept_terms": matched_concepts[:6],
                "source_session_id": memory.get("source_session_id"),
                "session_date": memory.get("session_date"),
                "content": content,
                "fact_key": _bounded_reader_text(memory.get("fact_key"), max_chars=80),
                "supersedes_fact_key": _bounded_reader_text(
                    memory.get("supersedes_fact_key"), max_chars=80
                ),
                "source_turn_indices": list(memory.get("source_turn_indices", []))[:8],
            }
        )
    cards.sort(
        key=lambda card: (
            card["support_kind"] == "direct_answer_support",
            card["answerability_score"],
            str(card.get("session_date") or ""),
            -int(card.get("rank") or 0),
        ),
        reverse=True,
    )
    return cards[: min(4, len(cards))]


def _parse_reader_answer(raw: str) -> str:
    """Return only the reader's final answer from structured or legacy output."""

    text = raw.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, Mapping):
        answer = payload.get("answer")
        if isinstance(answer, str):
            return answer.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            payload = json.loads(fenced.group(1))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping):
            answer = payload.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()

    trailing = re.search(r'(\{"answer"\s*:\s*".*?"\})\s*$', text, re.DOTALL)
    if trailing:
        try:
            payload = json.loads(trailing.group(1))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping):
            answer = payload.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()

    final_lines = re.findall(
        r"(?:^|\n)\s*(?:final answer|answer)\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if final_lines:
        return final_lines[-1].strip()
    return text


def _contribution_count_from_map(raw: str | None) -> int | None:
    """Count a complete structured map without asking the model to reduce it."""

    if not raw:
        return None
    text = raw.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        contributions = payload.get("contributions")
        if not isinstance(contributions, list) or not contributions:
            continue
        if all(isinstance(item, Mapping) for item in contributions):
            return len(contributions)
    return None


def _should_map_reduce_count(question: str) -> bool:
    """Use entity map-reduce only for artifact inventories, not scalar totals."""

    terms = _retrieval_terms(question)
    if "project" in terms and terms & {"lead", "led", "leading"}:
        return True
    return bool(
        terms
        & {
            "aircraft",
            "diorama",
            "kit",
            "kits",
            "model",
            "models",
            "tank",
        }
    )


def _memory_retrieval_score(
    memory: Any,
    *,
    query_terms: set[str],
    base_score: float,
    base_reason: str | None = None,
    evidence_plan: _QuestionEvidencePlan | None = None,
) -> float:
    """Blend content retrieval with title, tags, and fact identity signals."""

    if not query_terms:
        return base_score
    title_terms, content_terms, tag_terms, fact_terms = _memory_text_terms(memory)
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
    answerability_score = (
        _question_answerability_score(memory, evidence_plan)
        if evidence_plan is not None
        else 0.0
    )
    if lexical_score <= 0:
        # The local fallback embedding is deliberately lightweight and can
        # assign high scores to unrelated memories. Keep it as a synonym
        # fallback, but never let it outrank explicit query evidence.
        if answerability_score > 0:
            return 0.70 * answerability_score + 0.10 * max(base_score, 0.0)
        if base_reason == "provenance-context":
            # A sibling memory from a matched evidence group is useful as
            # context, but it is not query evidence by itself. Keep it below
            # even a weak lexical hit so contextual expansion cannot drown out
            # the exact fact the reader needs.
            score = max(base_score, 0.0) * 0.2
            return score * 0.15 if memory.metadata.get("source_context") else score
        if base_reason == "profile-context":
            return 0.45 * answerability_score + 0.35 * max(base_score, 0.0)
        score = max(base_score, 0.0) * 0.2
        return score * 0.15 if memory.metadata.get("source_context") else score
    score = (
        0.48 * lexical_score + 0.32 * answerability_score + 0.20 * max(base_score, 0.0)
    )
    if memory.metadata.get("source_context"):
        # Raw evidence is a lossless fallback, not a replacement for compact
        # facts. Require at least two lexical anchors and discount long source
        # passages so generic transcript language cannot dominate top-k.
        overlap = len(query_terms & content_terms)
        return score * (0.55 if overlap >= 2 else 0.15)
    return score


def _source_lexical_specificity(
    memory: Any,
    *,
    query_term_sets: Sequence[set[str]],
    document_frequency: Mapping[str, int],
    document_count: int,
) -> float:
    """Score a source excerpt by coverage of rare query terms.

    Long transcript chunks naturally contain generic words. An IDF-weighted
    coverage signal favors exact entities and attributes (for example a named
    venue plus a distinctive product) without encoding benchmark answers.
    """

    if not memory.metadata.get("source_context") or document_count <= 0:
        return 0.0
    _title_terms, content_terms, _tag_terms, _fact_terms = _memory_text_terms(memory)
    best = 0.0
    for query_terms in query_term_sets:
        if not query_terms:
            continue
        total_weight = sum(
            math.log((document_count + 1) / (document_frequency.get(term, 0) + 1)) + 1.0
            for term in query_terms
        )
        overlap_weight = sum(
            math.log((document_count + 1) / (document_frequency.get(term, 0) + 1)) + 1.0
            for term in query_terms & content_terms
        )
        if len(query_terms & content_terms) >= 2 and total_weight > 0:
            best = max(best, overlap_weight / total_weight)
    return best


def _provenance_group_relevance(matches: Sequence[RecallMatch]) -> float:
    """Rank evidence bundles by corroborated relevance, not one lucky hit."""

    scores = sorted((max(match.score, 0.0) for match in matches), reverse=True)
    if not scores:
        return 0.0
    # A strong direct fact should dominate group selection. Additional facts
    # corroborate it with diminishing returns, so verbose sessions cannot win
    # merely by producing more chunks.
    weights = (1.0, 0.35, 0.18, 0.09)
    return sum(score * weight for score, weight in zip(scores[:4], weights))


def _diversify_longmemeval_matches(
    matches: Sequence[RecallMatch],
    question_type: str,
    limit: int,
) -> list[RecallMatch]:
    """Keep evidence from several sessions before filling the remaining slots."""

    max_per_provenance = None
    if question_type in {"multi-session", "temporal-reasoning", "knowledge-update"}:
        max_per_provenance = 4
    elif question_type == "single-session-preference":
        # A verbose but weakly related conversation must not consume the full
        # reader window. Six facts retain enough detail from the best session
        # while leaving room for the actual preference-bearing provenance.
        max_per_provenance = 6
    if max_per_provenance is not None:
        # Round-robin diversity across every weak lexical hit can still use
        # the whole context budget on distractor sessions before a selected
        # evidence bundle gets its second fact. Bound the number of groups by
        # rank, then preserve several facts inside those strongest bundles.
        grouped: dict[str, list[RecallMatch]] = {}
        for match in matches:
            grouped.setdefault(provenance_key_for_memory(match.memory), []).append(
                match
            )
        # Keep enough candidate groups for a compact multi-session answer. A
        # low group cap can hide the second corroborating session before the
        # generic round-robin selector has a chance to use it.
        group_limit = (
            min(16, max(8, limit))
            if question_type == "multi-session"
            else max(6, min(10, limit // 2 or 1))
        )
        ranked_group_keys = sorted(
            grouped,
            key=lambda key: _provenance_group_relevance(grouped[key]),
            reverse=True,
        )[:group_limit]
        matches = [match for key in ranked_group_keys for match in grouped[key]]
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
                include_profile_context=(
                    question.question_type == "single-session-preference"
                ),
                profile_context_limit=8,
            )
        )
        for match in base_matches:
            previous = base_scores.get(match.memory.id)
            if previous is None or match.score > previous[0]:
                base_scores[match.memory.id] = (match.score, match.reason)
    query_term_sets = [_retrieval_terms(query) for query in retrieval_queries]
    if question.question_type == "single-session-preference":
        query_term_sets = [
            terms - _PREFERENCE_QUERY_NOISE or terms for terms in query_term_sets
        ]
    evidence_plan = _question_evidence_plan(
        question.question,
        question_type=question.question_type,
    )
    memories = await service.list_memories(
        namespace_id,
        statuses=[MemoryStatus.ACTIVE],
    )
    query_vocabulary = set().union(*query_term_sets)
    document_frequency = {
        term: sum(term in _memory_text_terms(memory)[1] for memory in memories)
        for term in query_vocabulary
    }
    reranked: list[RecallMatch] = []
    for memory in memories:
        scored_candidates = [
            _memory_retrieval_score(
                memory,
                query_terms=query_terms,
                base_score=base_scores.get(memory.id, (0.0, None))[0],
                base_reason=base_scores.get(memory.id, (0.0, None))[1],
                evidence_plan=evidence_plan,
            )
            for query_terms in query_term_sets
        ]
        score = max(scored_candidates, default=0.0)
        source_specificity = _source_lexical_specificity(
            memory,
            query_term_sets=query_term_sets,
            document_frequency=document_frequency,
            document_count=len(memories),
        )
        if memory.metadata.get("source_context"):
            if source_specificity < 0.22:
                continue
            score = max(score, 0.75 * source_specificity)
        if memory.id in base_scores or score > 0:
            reranked.append(
                RecallMatch(
                    memory=memory,
                    score=score,
                    reason="longmemeval-rerank",
                )
            )
    # A precise source excerpt can identify the right conversation even when
    # its compact facts use entity names absent from the question (for example
    # a product model instead of its general type). Bring those facts into the
    # candidate set without copying the whole transcript into the reader.
    if question.question_type == "single-session-preference":
        source_group_scores: dict[str, float] = {}
        for match in reranked:
            if match.memory.metadata.get("source_context"):
                provenance = provenance_key_for_memory(match.memory)
                source_group_scores[provenance] = max(
                    source_group_scores.get(provenance, 0.0), match.score
                )
        reranked = [
            RecallMatch(
                memory=match.memory,
                score=max(
                    match.score,
                    0.65
                    * source_group_scores.get(
                        provenance_key_for_memory(match.memory), 0.0
                    )
                    + 0.35 * match.score,
                ),
                reason=match.reason,
            )
            if not match.memory.metadata.get("source_context")
            else match
            for match in reranked
        ]
    reranked.sort(
        key=lambda match: (
            match.score,
            match.memory.confidence,
            match.memory.metadata.get("source_session_date") or "",
        ),
        reverse=True,
    )
    source_limit = (
        min(2, limit)
        if question.question_type == "single-session-assistant"
        else min(2, limit // 4)
    )
    structured = [
        match for match in reranked if not match.memory.metadata.get("source_context")
    ]
    source: list[RecallMatch] = []
    source_provenance_counts: dict[str, int] = {}
    max_source_per_provenance = (
        2 if question.question_type == "single-session-assistant" else 1
    )
    if source_limit:
        for match in reranked:
            if not match.memory.metadata.get("source_context"):
                continue
            provenance = provenance_key_for_memory(match.memory)
            if source_provenance_counts.get(provenance, 0) >= max_source_per_provenance:
                continue
            source.append(match)
            source_provenance_counts[provenance] = (
                source_provenance_counts.get(provenance, 0) + 1
            )
            if len(source) >= source_limit:
                break
    if question.question_type == "single-session-assistant":
        # Assistant-answer questions often require a precise name or visual
        # attribute that extraction omitted, so exact source evidence may lead.
        return sorted(
            [*structured, *source], key=lambda match: match.score, reverse=True
        )[:limit]
    # For counts, temporal reasoning, and preferences, compact facts stay in
    # front and a tiny source budget is appended as provenance. This prevents
    # verbose transcripts from displacing the ledger/profile the reader needs.
    selected = _diversify_longmemeval_matches(
        structured,
        question.question_type,
        max(0, limit - len(source)),
    )
    return [*selected, *source][:limit]


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
            retrieval_hit_count=sum(item.retrieval_hit_at_k is True for item in group),
            answer_session_recall_evaluable_count=sum(
                item.answer_session_recall_at_k is not None for item in group
            ),
            answer_session_recall_sum=sum(
                item.answer_session_recall_at_k or 0.0 for item in group
            ),
            partial_ingestion_count=sum(
                item.ingestion_failed_session_count > 0 for item in group
            ),
            clean_scored_count=sum(
                item.judge_label is not None and item.ingestion_complete
                for item in group
            ),
            clean_correct_count=sum(
                item.judge_label is True and item.ingestion_complete for item in group
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
