"""LongMemEval answer generation and official judge integration.

The ingestion adapter deliberately stops at structured memories.  This module
adds the separate QA stage: retrieve active memories, ask a reader to answer
from that evidence, then apply the official LongMemEval yes/no judge prompt.
"""

from __future__ import annotations

import asyncio
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
from .domain import MemoryService, RecallMatch, RecallQuery
from .longmemeval import (
    ExtractionCache,
    FactExtractor,
    LongMemEvalQuestion,
    ingest_longmemeval_question,
    load_longmemeval_instances,
)

LONGMEMEVAL_QA_CACHE_SCHEMA = "snipara.longmemeval.qa-cache.v1"
LONGMEMEVAL_READER_PROMPT_VERSION = "lmstudio-longmemeval-reader-v1"
LONGMEMEVAL_JUDGE_PROMPT_VERSION = "longmemeval-official-judge-v1"

READER_SYSTEM_PROMPT = """You answer a LongMemEval question using only the retrieved memories below.

Give the shortest direct answer that is supported by the memories. Resolve
updates by preferring the active/latest fact shown in the context. Do not
invent details, do not use outside knowledge, and do not mention the memory
system or the retrieval process. If the context does not support the answer,
say clearly that the information cannot be determined from the available
memories.
"""


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
                "title": match.memory.title,
                "content": match.memory.content,
                "memory_type": match.memory.type.value,
                "session_date": match.memory.metadata.get("source_session_date"),
            }
            for match in memories
        ]
        user_payload = json.dumps(
            {"question": question, "retrieved_memories": context},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return await self._client.complete(
            [
                {"role": "system", "content": READER_SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
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

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.scored_count if self.scored_count else 0.0


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
    categories: tuple[LongMemEvalCategoryReport, ...]
    questions: tuple[LongMemEvalQAResult, ...]

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.scored_count if self.scored_count else 0.0

    @property
    def coverage(self) -> float:
        return self.scored_count / self.question_count if self.question_count else 0.0


def _question_category(question: LongMemEvalQuestion) -> str:
    return "abstention" if "_abs" in question.question_id else question.question_type


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
) -> LongMemEvalQAReport:
    """Run LongMemEval ingestion, retrieval, reader generation, and judging."""

    if retrieval_k <= 0:
        raise ValueError("retrieval_k must be positive")
    questions = load_longmemeval_instances(dataset_path, limit=limit)
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
            )
            ingestion_failed_session_count += len(ingestion.failed_session_ids)
            matches = await service.semantic_recall(
                RecallQuery(
                    namespace_id=ingestion.namespace_id,
                    query=_normalize_retrieval_query(question.question),
                    limit=retrieval_k,
                )
            )
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
            results.append(
                LongMemEvalQAResult(
                    question_id=question.question_id,
                    question_type=question.question_type,
                    category=category,
                    retrieved_count=0,
                    retrieved_titles=(),
                    reader_response=None,
                    judge_response=None,
                    judge_label=None,
                    status="failed",
                    failed_stage="ingestion-or-retrieval",
                    failure_message=f"{type(error).__name__}: {str(error)[:500]}",
                )
            )
            continue

        input_hash = _retrieval_input_hash(question, matches, retrieval_k)
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
                reader_response = await reader.answer(question.question, matches)
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
        categories=tuple(categories),
        questions=tuple(results),
    )


def _model_name(component: object) -> str:
    model = getattr(component, "model", None)
    return str(model) if model else str(getattr(component, "version", "unknown"))


def _normalize_retrieval_query(query: str) -> str:
    """Keep punctuation from becoming a retrieval token in the local adapter."""

    return re.sub(r"[^\w]+", " ", query, flags=re.UNICODE).strip()


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
