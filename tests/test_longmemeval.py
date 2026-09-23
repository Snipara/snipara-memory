from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from snipara_memory import (
    ExtractedFact,
    ExtractionCache,
    InMemoryMemoryStore,
    LmStudioBatchFactExtractor,
    LmStudioFactExtractor,
    LmStudioLongMemEvalJudge,
    LmStudioLongMemEvalReader,
    LongMemEvalQACache,
    LongMemEvalQuestion,
    LongMemEvalSession,
    LongMemEvalTurn,
    MemoryService,
    MemoryStatus,
    MemoryType,
    OpenRouterJevLongMemEvalJudge,
    ingest_longmemeval_dataset,
    ingest_longmemeval_question,
    load_longmemeval_instances,
    official_longmemeval_judge_prompt,
    run_longmemeval_qa,
)
from snipara_memory.longmemeval import (
    _LmStudioRequestError,
    _augment_high_signal_evidence,
    _facts_from_lm_studio_batch_response,
    _facts_from_lm_studio_response,
)


def test_extractor_recovers_failed_chunk_without_losing_turn_indices() -> None:
    class BoundedExtractor(LmStudioFactExtractor):
        async def _post_json_async(self, payload):
            session = json.loads(payload["messages"][1]["content"])
            turns = session["turns"]
            if sum(len(turn["content"]) for turn in turns) > 12:
                if failure_mode == "context":
                    raise _LmStudioRequestError("HTTP 400: context overflow")
                return {"choices": [{"message": {"content": '{"facts":[{"content":'}}]}
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "facts": [
                                        {
                                            "content": turn["content"],
                                            "memory_type": "FACT",
                                            "source_turn_indices": [index],
                                        }
                                        for index, turn in enumerate(turns)
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

    session = LongMemEvalSession(
        session_id="split-test",
        date="2024-01-01",
        turns=(
            LongMemEvalTurn(role="user", content="first turn"),
            LongMemEvalTurn(role="assistant", content="second turn"),
            LongMemEvalTurn(role="user", content="third turn"),
        ),
    )
    for failure_mode in ("context", "invalid_json"):
        extractor = BoundedExtractor(model="test", retries=0)
        facts = asyncio.run(extractor.extract(session))
        assert [fact.source_turn_indices for fact in facts] == [(0,), (1,), (2,)]
        assert [fact.content for fact in facts] == [
            "first turn",
            "second turn",
            "third turn",
        ]


from snipara_memory.qa import (
    READER_SYSTEM_PROMPT,
    _action_item_checklist,
    _apply_acquisition_count_guard,
    _apply_action_count_guard,
    _apply_assistant_language_guard,
    _apply_assistant_recommendation_guard,
    _apply_binary_relation_guard,
    _apply_count_resolution_guard,
    _apply_currency_total_guard,
    _apply_device_battery_guard,
    _apply_direct_attribute_guard,
    _apply_direct_object_guard,
    _apply_multi_session_count_guard,
    _apply_named_fact_guard,
    _apply_named_list_entity_guard,
    _apply_preference_projection_guard,
    _apply_temporal_interval_guard,
    _apply_temporal_order_guard,
    _apply_temporal_relation_guard,
    _apply_temporal_trip_order_guard,
    _apply_update_resolution_guard,
    _bounded_reader_evidence,
    _compact_resolution_audit,
    _contribution_count_from_map,
    _deterministic_resolution_audit,
    _expanded_retrieval_terms,
    _longmemeval_contract_override,
    _named_event_mentions,
    _parse_judge_label,
    _parse_reader_answer,
    _project_identity,
    _question_answerability_score,
    _question_evidence_plan,
    _reader_context_limit,
    _reader_evidence_cards,
    _resolution_terms,
    _retrieval_terms,
    _should_map_reduce_count,
    _source_lexical_specificity,
    _temporal_event_phrases,
)


def _question_payload() -> dict[str, object]:
    return {
        "question_id": "q-1",
        "question_type": "knowledge-update",
        "question": "Where do I live now?",
        "answer": "Zurich",
        "question_date": "2024-06-01",
        "haystack_session_ids": ["session-1", "session-2"],
        "haystack_dates": ["2024-01-01", "2024-05-01"],
        "haystack_sessions": [
            [{"role": "user", "content": "I live in Paris."}],
            [{"role": "user", "content": "I now live in Zurich."}],
        ],
        "answer_session_ids": ["session-2"],
    }


class CountingExtractor:
    def __init__(self, version: str = "test-extractor-v1") -> None:
        self.version = version
        self.calls = 0

    async def extract(self, session) -> list[ExtractedFact]:
        self.calls += 1
        if session.session_id == "session-1":
            return [
                ExtractedFact(
                    content="The user lives in Paris.",
                    memory_type=MemoryType.FACT,
                    fact_key="user-city",
                )
            ]
        return [
            ExtractedFact(
                content="The user now lives in Zurich.",
                memory_type=MemoryType.FACT,
                fact_key="user-city",
                supersedes_fact_key="user-city",
            )
        ]


def test_load_longmemeval_instances_validates_parallel_session_fields(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([_question_payload()]), encoding="utf-8")

    questions = load_longmemeval_instances(dataset, limit=1)

    assert len(questions) == 1
    assert questions[0].sessions[0].session_id == "session-1"
    assert (
        questions[0].sessions[0].content_hash != questions[0].sessions[1].content_hash
    )


def test_load_longmemeval_instances_skips_blank_turn_placeholders(
    tmp_path: Path,
) -> None:
    payload = _question_payload()
    payload["haystack_sessions"][0].append({"role": "user", "content": ""})
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([payload]), encoding="utf-8")

    questions = load_longmemeval_instances(dataset, limit=1)

    assert len(questions[0].sessions[0].turns) == 1


async def test_ingestion_uses_cache_and_invalidates_by_extractor_version(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([_question_payload()]), encoding="utf-8")
    cache_path = tmp_path / "extractions.json"

    first_extractor = CountingExtractor()
    first_report = await ingest_longmemeval_dataset(
        MemoryService(InMemoryMemoryStore()),
        dataset,
        first_extractor,
        cache_path=cache_path,
    )
    second_extractor = CountingExtractor()
    second_report = await ingest_longmemeval_dataset(
        MemoryService(InMemoryMemoryStore()),
        dataset,
        second_extractor,
        cache_path=cache_path,
    )
    third_extractor = CountingExtractor(version="test-extractor-v2")
    third_report = await ingest_longmemeval_dataset(
        MemoryService(InMemoryMemoryStore()),
        dataset,
        third_extractor,
        cache_path=cache_path,
    )

    assert first_report.cache_misses == 2
    assert first_extractor.calls == 2
    assert second_report.cache_hits == 2
    assert second_extractor.calls == 0
    assert third_report.cache_misses == 2
    assert third_extractor.calls == 2
    assert (
        ExtractionCache(cache_path).get(
            "q-1:session-1",
            session_hash=questions_hash(dataset, 0),
            extractor_version="test-extractor-v1",
        )
        is None
    )


def test_extraction_cache_reuses_success_by_content_hash(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path / "extractions.json")
    fact = ExtractedFact(content="The user lives in Paris.")
    cache.put(
        "q-1:session-1",
        session_hash="same-content",
        extractor_version="extractor-v1",
        facts=[fact],
    )

    reused = cache.get(
        "q-2:session-9",
        session_hash="same-content",
        extractor_version="extractor-v1",
    )

    assert reused == [fact]


async def test_ingestion_graveyards_superseded_facts_and_keeps_provenance(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([_question_payload()]), encoding="utf-8")
    store = InMemoryMemoryStore()
    service = MemoryService(store)

    report = await ingest_longmemeval_dataset(service, dataset, CountingExtractor())
    result = report.questions[0]
    active = await service.list_memories(
        result.namespace_id, statuses=[MemoryStatus.ACTIVE]
    )
    all_memories = await service.list_memories(result.namespace_id)
    city_memories = [
        memory for memory in all_memories if memory.memory_key == "user-city"
    ]
    old_memory = next(
        memory for memory in city_memories if memory.status is MemoryStatus.GRAVEYARD
    )
    new_memory = next(
        memory for memory in city_memories if memory.status is MemoryStatus.ACTIVE
    )

    assert [
        memory.content for memory in active if not memory.metadata.get("source_context")
    ] == ["The user now lives in Zurich."]
    assert old_memory.status is MemoryStatus.GRAVEYARD
    assert old_memory.memory_key == "user-city"
    assert new_memory.memory_key == "user-city"
    assert new_memory.supersedes_memory_key == "user-city"
    assert new_memory.provenance_key == "session-2"
    assert new_memory.observed_at is not None
    assert new_memory.observed_at.isoformat() == "2024-05-01T00:00:00+00:00"
    assert new_memory.metadata["source_session_id"] == "session-2"
    assert "answer-session" in new_memory.tags
    assert result.superseded_memory_ids == (old_memory.id,)


async def test_ingestion_handles_repeated_supersession_without_crashing(
    tmp_path: Path,
) -> None:
    payload = _question_payload()
    payload["haystack_session_ids"] = ["session-1", "session-2", "session-3"]
    payload["haystack_dates"] = ["2024-01-01", "2024-05-01", "2024-06-01"]
    payload["haystack_sessions"] = [
        *payload["haystack_sessions"],
        [{"role": "user", "content": "I still live in Zurich."}],
    ]
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([payload]), encoding="utf-8")
    store = InMemoryMemoryStore()

    report = await ingest_longmemeval_dataset(
        MemoryService(store),
        dataset,
        CountingExtractor(),
    )

    result = report.questions[0]
    all_memories = await store.list_memories("longmemeval:q-1")
    superseded_city_ids = {
        memory.id
        for memory in all_memories
        if memory.memory_key == "user-city" and memory.status is MemoryStatus.GRAVEYARD
    }
    assert set(result.superseded_memory_ids) == superseded_city_ids
    active = await store.list_memories(
        "longmemeval:q-1", statuses=[MemoryStatus.ACTIVE]
    )
    assert [
        memory.content for memory in active if not memory.metadata.get("source_context")
    ] == ["The user now lives in Zurich."]


async def test_ingestion_keeps_cumulative_todo_evidence_active(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([_question_payload()]), encoding="utf-8")

    class CumulativeActionExtractor:
        version = "cumulative-action-v1"

        async def extract(self, session) -> list[ExtractedFact]:
            if session.session_id == "session-1":
                return [
                    ExtractedFact(
                        content="The user needs to return the exchanged boots.",
                        memory_type=MemoryType.TODO,
                        fact_key="boots.return",
                        metadata={"evidence_kind": "user_fact"},
                    )
                ]
            return [
                ExtractedFact(
                    content="The user needs to pick up the exchanged boots.",
                    memory_type=MemoryType.TODO,
                    fact_key="boots.pickup",
                    supersedes_fact_key="boots.return",
                    metadata={"evidence_kind": "assistant_answer"},
                )
            ]

    service = MemoryService(InMemoryMemoryStore())
    report = await ingest_longmemeval_dataset(
        service, dataset, CumulativeActionExtractor()
    )

    result = report.questions[0]
    active = await service.list_memories(
        result.namespace_id, statuses=[MemoryStatus.ACTIVE]
    )
    assert {
        memory.metadata["fact_key"]
        for memory in active
        if not memory.metadata.get("source_context")
    } == {
        "boots.return",
        "boots.pickup",
    }
    assert result.superseded_memory_ids == ()


async def test_ingestion_flushes_extraction_cache_after_each_session(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([_question_payload()]), encoding="utf-8")
    cache_path = tmp_path / "extractions.json"
    second_session_started = asyncio.Event()
    release_second_session = asyncio.Event()

    class FailOnSecondSession:
        version = "failing-extractor-v1"

        async def extract(self, session) -> list[ExtractedFact]:
            if session.session_id == "session-2":
                second_session_started.set()
                await release_second_session.wait()
                raise RuntimeError("synthetic extraction failure")
            return [ExtractedFact(content="The user lives in Paris.")]

    task = asyncio.create_task(
        ingest_longmemeval_dataset(
            MemoryService(InMemoryMemoryStore()),
            dataset,
            FailOnSecondSession(),
            cache_path=cache_path,
        )
    )
    await asyncio.wait_for(second_session_started.wait(), timeout=2)

    question = load_longmemeval_instances(dataset)[0]
    cached_before_second_finishes = ExtractionCache(cache_path).get(
        "q-1:session-1",
        session_hash=question.sessions[0].content_hash,
        extractor_version="failing-extractor-v1",
    )
    assert cached_before_second_finishes is not None
    release_second_session.set()
    report = await task

    result = report.questions[0]
    assert report.failed_session_count == 1
    assert result.failed_session_ids == ("session-2",)
    assert "synthetic extraction failure" in result.failure_messages[0]
    cached = ExtractionCache(cache_path).get(
        "q-1:session-1",
        session_hash=questions_hash(dataset, 0),
        extractor_version="failing-extractor-v1",
    )
    assert cached is not None
    assert cached[0].content == "The user lives in Paris."
    question = load_longmemeval_instances(dataset)[0]
    failure = ExtractionCache(cache_path).get_failure(
        "q-1:session-2",
        session_hash=question.sessions[1].content_hash,
        extractor_version="failing-extractor-v1",
    )
    assert failure is not None
    assert "synthetic extraction failure" in failure

    replay = await ingest_longmemeval_dataset(
        MemoryService(InMemoryMemoryStore()),
        dataset,
        FailOnSecondSession(),
        cache_path=cache_path,
    )
    assert replay.questions[0].failed_session_ids == ("session-2",)

    class RecoveringExtractor(FailOnSecondSession):
        async def extract(self, session) -> list[ExtractedFact]:
            if session.session_id == "session-2":
                return [ExtractedFact(content="The user now lives in Zurich.")]
            return await super().extract(session)

    recovered = await ingest_longmemeval_dataset(
        MemoryService(InMemoryMemoryStore()),
        dataset,
        RecoveringExtractor(),
        cache_path=cache_path,
        retry_failed=True,
    )
    assert recovered.questions[0].failed_session_ids == ()
    assert recovered.questions[0].cache_misses == 1


async def test_ingestion_retries_new_parallel_failures_serially(tmp_path: Path) -> None:
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([_question_payload()]), encoding="utf-8")
    cache_path = tmp_path / "extractions.json"

    class TransientExtractor:
        version = "transient-extractor-v1"

        def __init__(self) -> None:
            self.calls: dict[str, int] = {}

        async def extract(self, session) -> list[ExtractedFact]:
            call_count = self.calls.get(session.session_id, 0) + 1
            self.calls[session.session_id] = call_count
            if session.session_id == "session-2" and call_count == 1:
                raise RuntimeError("synthetic transient failure")
            return [ExtractedFact(content=f"Fact from {session.session_id}.")]

    extractor = TransientExtractor()
    question = load_longmemeval_instances(dataset)[0]
    result = await ingest_longmemeval_question(
        MemoryService(InMemoryMemoryStore()),
        question,
        extractor,
        cache=ExtractionCache(cache_path),
        extraction_concurrency=2,
        serial_retry_transient_failures=True,
    )

    assert result.failed_session_ids == ()
    assert result.cache_misses == 2
    assert extractor.calls == {"session-1": 1, "session-2": 2}
    recovered = ExtractionCache(cache_path).get(
        "q-1:session-2",
        session_hash=question.sessions[1].content_hash,
        extractor_version="transient-extractor-v1",
    )
    assert recovered is not None


def questions_hash(dataset: Path, index: int) -> str:
    question = load_longmemeval_instances(dataset)[index]
    return question.sessions[0].content_hash


def test_qa_cache_invalidates_judge_when_reader_response_changes(
    tmp_path: Path,
) -> None:
    cache = LongMemEvalQACache(tmp_path / "qa-cache.json")
    cache.put_reader(
        "q-1",
        input_hash="input-1",
        reader_version="reader-v1",
        judge_version="judge-v1",
        response="old response",
    )
    cache.put_judge(
        "q-1",
        input_hash="input-1",
        reader_version="reader-v1",
        judge_version="judge-v1",
        label=False,
        response="old judge response",
    )

    cache.put_reader(
        "q-1",
        input_hash="input-1",
        reader_version="reader-v2",
        judge_version="judge-v1",
        response="new response",
    )

    assert (
        cache.get_judge(
            "q-1",
            input_hash="input-1",
            reader_version="reader-v2",
            judge_version="judge-v1",
        )
        is None
    )


@contextmanager
def _json_server(response_payload: dict[str, Any]) -> Iterator[ThreadingHTTPServer]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"] or 0)
            body = self.rfile.read(length)
            self.server.requests.append(json.loads(body))
            response = json.dumps(response_payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


async def test_lm_studio_extractor_uses_structured_output_without_ground_truth() -> (
    None
):
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "facts": [
                                {
                                    "content": "The user prefers Zurich.",
                                    "title": "City preference",
                                    "memory_type": "PREFERENCE",
                                    "confidence": 0.91,
                                    "fact_key": "user.city",
                                    "supersedes_fact_key": None,
                                    "source_turn_indices": [0],
                                    "tags": ["profile"],
                                }
                            ]
                        }
                    )
                }
            }
        ]
    }
    session = LongMemEvalSession(
        session_id="session-1",
        date="2024-06-01",
        turns=(
            LongMemEvalTurn(
                role="user",
                content="I prefer Zurich.",
                has_answer=True,
            ),
        ),
    )

    with _json_server(response) as server:
        extractor = LmStudioFactExtractor(
            model="local-test-model",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            reasoning_effort="low",
            retries=0,
        )
        facts = await extractor.extract(session)

        assert len(facts) == 1
        assert facts[0].content == "The user prefers Zurich."
        assert facts[0].memory_type is MemoryType.PREFERENCE
        assert len(server.requests) == 1
        payload = server.requests[0]
        assert payload["model"] == "local-test-model"
        assert payload["response_format"]["type"] == "json_schema"
        assert (
            payload["response_format"]["json_schema"]["schema"]["properties"]["facts"][
                "maxItems"
            ]
            == 12
        )
        assert payload["reasoning_effort"] == "low"
        assert "Extraction priority is important" in payload["messages"][0]["content"]
        user_prompt = payload["messages"][1]["content"]
        assert "has_answer" not in user_prompt
        assert "I prefer Zurich." in user_prompt


async def test_lm_studio_extractor_chunks_large_sessions_and_maps_turn_indices() -> (
    None
):
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "facts": [
                                {
                                    "content": "A durable fact.",
                                    "title": "Fact",
                                    "memory_type": "FACT",
                                    "confidence": 0.8,
                                    "fact_key": None,
                                    "supersedes_fact_key": None,
                                    "source_turn_indices": [0],
                                    "tags": [],
                                }
                            ]
                        }
                    )
                }
            }
        ]
    }
    session = LongMemEvalSession(
        session_id="large-session",
        date="2024-06-01",
        turns=(
            LongMemEvalTurn(role="user", content="abcdef"),
            LongMemEvalTurn(role="assistant", content="ghijkl"),
        ),
    )

    with _json_server(response) as server:
        extractor = LmStudioFactExtractor(
            model="local-test-model",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            max_session_chars=8,
            retries=0,
        )
        facts = await extractor.extract(session)

    assert len(server.requests) == 2
    assert [fact.source_turn_indices for fact in facts] == [(0,), (1,)]


async def test_lm_studio_batch_extractor_chunks_and_remaps_turn_indices() -> None:
    class FakeChunkBatchExtractor(LmStudioFactExtractor):
        async def _extract_batch_request(self, sessions):
            return {
                session.session_id: [
                    ExtractedFact(
                        content=f"fact-{session.session_id}",
                        source_turn_indices=(0,),
                    )
                ]
                for session in sessions
            }

    session = LongMemEvalSession(
        session_id="large-session",
        date="2024-06-01",
        turns=(
            LongMemEvalTurn(role="user", content="abcdef"),
            LongMemEvalTurn(role="assistant", content="ghijkl"),
        ),
    )
    extractor = FakeChunkBatchExtractor(
        model="local-test-model",
        max_session_chars=8,
        batch_request_size=2,
        retries=0,
    )

    extracted = await extractor.extract_batch([session])

    assert [fact.source_turn_indices for fact in extracted["large-session"]] == [
        (0,),
        (1,),
    ]


def test_lm_studio_version_changes_when_model_or_prompt_changes() -> None:
    base = LmStudioFactExtractor(model="model-a")

    assert base.version == "model-a:lmstudio-fact-extractor-v4"
    assert base.max_session_chars == 12000
    assert LmStudioFactExtractor(model="model-b").version != base.version
    assert (
        LmStudioFactExtractor(
            model="model-a", prompt_version="lmstudio-fact-extractor-v5"
        ).version
        != base.version
    )
    assert (
        LmStudioFactExtractor(model="model-a", reasoning_effort="low").version
        != base.version
    )


def test_lm_studio_salvages_complete_facts_before_truncated_json() -> None:
    complete_fact = {
        "content": "The user prefers Zurich.",
        "title": "City preference",
        "memory_type": "PREFERENCE",
        "confidence": 0.91,
        "fact_key": "user.city",
        "supersedes_fact_key": None,
        "source_turn_indices": [0],
        "tags": ["profile"],
    }
    truncated_content = (
        '{"facts":['
        + json.dumps(complete_fact)
        + ',{"content":"The second fact is truncated'
    )

    facts = _facts_from_lm_studio_response(
        {"choices": [{"message": {"content": truncated_content}}]}
    )

    assert len(facts) == 1
    assert facts[0].content == "The user prefers Zurich."
    assert facts[0].metadata["lm_studio_parse"] == "salvaged_json_prefix"


def test_lm_studio_preserves_evidence_metadata() -> None:
    facts = _facts_from_lm_studio_response(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "facts": [
                                    {
                                        "content": "The assistant recommended Miss Bee Providore.",
                                        "title": "Restaurant recommendation",
                                        "memory_type": "FACT",
                                        "confidence": 0.9,
                                        "fact_key": "restaurant.miss_bee_providore",
                                        "supersedes_fact_key": None,
                                        "source_turn_indices": [2, 3],
                                        "tags": ["restaurant"],
                                        "evidence_kind": "assistant_answer",
                                        "temporal_anchor": "2024-06-01",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
    )

    assert facts[0].metadata["evidence_kind"] == "assistant_answer"
    assert facts[0].metadata["temporal_anchor"] == "2024-06-01"


def test_high_signal_augmentation_preserves_schedule_table_cells() -> None:
    session = LongMemEvalSession(
        session_id="schedule-1",
        date="2024-06-01",
        turns=(
            LongMemEvalTurn(
                role="assistant",
                content=(
                    "|  | 8 am - 4 pm (Day Shift) | 12 pm - 8 pm |\n"
                    "| --- | --- | --- |\n"
                    "| Sunday | Admon | Magdy |"
                ),
            ),
        ),
    )

    facts = _augment_high_signal_evidence(session, [])

    assert any(
        fact.title == "Shift rotation Sunday"
        and "Admon is assigned to 8 am - 4 pm" in fact.content
        for fact in facts
    )


def test_high_signal_augmentation_preserves_explicit_museum_visit() -> None:
    session = LongMemEvalSession(
        session_id="museum-1",
        date="2024-06-01",
        turns=(
            LongMemEvalTurn(
                role="user",
                content="I just got back from a guided tour at the Museum of Modern Art.",
            ),
        ),
    )

    facts = _augment_high_signal_evidence(session, [])

    assert facts[0].fact_key == "user.visit.museum_of_modern_art"
    assert "Museum of Modern Art" in facts[0].content


def test_lm_studio_bounds_confidence_schema_drift() -> None:
    facts = _facts_from_lm_studio_response(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "facts": [
                                    {
                                        "content": "The user prefers Zurich.",
                                        "title": "City preference",
                                        "memory_type": "PREFERENCE",
                                        "confidence": 1.04,
                                        "fact_key": "user.city",
                                        "supersedes_fact_key": None,
                                        "source_turn_indices": [0],
                                        "tags": [],
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
    )

    assert len(facts) == 1
    assert facts[0].confidence == 1.0


def test_lm_studio_batch_response_preserves_session_ids() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "sessions": [
                                {
                                    "session_id": "session-1",
                                    "facts": [
                                        {
                                            "content": "The user likes tea.",
                                            "title": "Tea preference",
                                            "memory_type": "PREFERENCE",
                                            "confidence": 0.8,
                                            "fact_key": "user.drink",
                                            "supersedes_fact_key": None,
                                            "source_turn_indices": [0],
                                            "tags": [],
                                        }
                                    ],
                                },
                                {"session_id": "session-2", "facts": []},
                            ]
                        }
                    )
                }
            }
        ]
    }

    facts = _facts_from_lm_studio_batch_response(response)

    assert list(facts) == ["session-1", "session-2"]
    assert facts["session-1"][0].content == "The user likes tea."
    assert facts["session-2"] == []


def test_lm_studio_batch_payload_requires_every_session_result() -> None:
    extractor = LmStudioFactExtractor(model="local-test-model")
    sessions = [
        LongMemEvalSession(
            session_id=f"session-{index}",
            date="2024-01-01",
            turns=(LongMemEvalTurn(role="user", content="hello"),),
        )
        for index in range(3)
    ]

    payload = extractor._build_batch_payload(sessions)
    schema = payload["response_format"]["json_schema"]["schema"]

    assert schema["properties"]["sessions"]["minItems"] == 3
    assert schema["properties"]["sessions"]["maxItems"] == 3
    assert (
        schema["properties"]["sessions"]["items"]["properties"]["facts"]["maxItems"]
        == 12
    )


async def test_lm_studio_batch_extractor_coalesces_requests() -> None:
    class FakeBatchExtractor:
        version = "fake-batch-v1"

        def __init__(self) -> None:
            self.calls = 0

        async def extract_batch(self, sessions):
            self.calls += 1
            return {
                session.session_id: [
                    ExtractedFact(content=f"fact-{session.session_id}")
                ]
                for session in sessions
            }

    inner = FakeBatchExtractor()
    extractor = LmStudioBatchFactExtractor(
        inner, batch_size=2, dispatch_delay_seconds=0
    )
    sessions = [
        LongMemEvalSession(
            session_id=f"session-{index}",
            date="2024-01-01",
            turns=(LongMemEvalTurn(role="user", content="hello"),),
        )
        for index in range(2)
    ]

    facts = await asyncio.gather(*(extractor.extract(session) for session in sessions))

    assert inner.calls == 1
    assert [item[0].content for item in facts] == [
        "fact-session-0",
        "fact-session-1",
    ]


def test_official_judge_prompt_uses_task_specific_rules() -> None:
    update_prompt = official_longmemeval_judge_prompt(
        "knowledge-update",
        "Where do I live now?",
        "Zurich",
        "You now live in Zurich.",
    )
    temporal_prompt = official_longmemeval_judge_prompt(
        "temporal-reasoning",
        "How many days?",
        "18",
        "19 days.",
    )
    abstention_prompt = official_longmemeval_judge_prompt(
        "unknown",
        "What is my passport number?",
        "The information is unavailable.",
        "I cannot determine that.",
        abstention=True,
    )

    assert "updated answer" in update_prompt
    assert "off-by-one" in temporal_prompt
    assert "unanswerable" in abstention_prompt


def test_reader_prompt_counts_action_records_across_venues() -> None:
    reader = LmStudioLongMemEvalReader(model="local-test-model")

    assert ":lmstudio-longmemeval-reader-v59" in reader.version
    assert "dry-cleaning pickup" in READER_SYSTEM_PROMPT
    assert "silently enumerate" in READER_SYSTEM_PROMPT
    assert "niece" in READER_SYSTEM_PROMPT
    assert "vintage\n  cameras" in READER_SYSTEM_PROMPT
    assert "unrelated\n  recommendations" in READER_SYSTEM_PROMPT
    assert "scale vehicle or aircraft diorama" in READER_SYSTEM_PROMPT
    assert '{"answer":"..."}' in READER_SYSTEM_PROMPT


def test_action_item_checklist_keeps_exchange_actions_distinct() -> None:
    checklist = _action_item_checklist(
        "How many items of clothing do I need to pick up or return from a store?",
        [
            {
                "source_session_id": "session-1",
                "title": "Zara boot pickup",
                "content": "Pick up new boots from Zara.",
            },
            {
                "source_session_id": "session-2",
                "title": "Dry cleaning pickup",
                "content": "Pick up the navy blazer from dry cleaning.",
            },
            {
                "source_session_id": "session-3",
                "title": "Boot return",
                "content": "Return boots to Zara and pick up the exchanged pair.",
            },
        ],
    )

    assert checklist == [
        {
            "source_session_id": "session-1",
            "action": "pickup",
            "item": "boots",
            "venue": "zara",
        },
        {
            "source_session_id": "session-2",
            "action": "pickup",
            "item": "blazer",
            "venue": "dry cleaning",
        },
        {
            "source_session_id": "session-3",
            "action": "return",
            "item": "boots",
            "venue": "zara",
        },
    ]
    assert _apply_action_count_guard("2", checklist) == "3"


def test_action_item_checklist_excludes_personal_returns_without_store() -> None:
    checklist = _action_item_checklist(
        "How many items of clothing do I need to pick up or return from a store?",
        [
            {
                "source_session_id": "family",
                "content": "My sister will return my green sweater next week.",
            },
            {
                "source_session_id": "shop",
                "content": "I need to return my jacket to North Market.",
            },
        ],
    )

    assert checklist == [
        {
            "source_session_id": "shop",
            "action": "return",
            "item": "jacket",
            "venue": "north market",
        }
    ]


def test_action_item_checklist_rejects_colors_and_address_substrings() -> None:
    checklist = _action_item_checklist(
        "How many items of clothing do I need to pick up or return from a store?",
        [
            {
                "source_session_id": "family",
                "title": "Green sweater return",
                "content": "My sister will return my green sweater next week.",
            },
            {
                "source_session_id": "commerce",
                "title": "Returns process details",
                "content": "The return address and refund method are not finalized.",
            },
        ],
    )

    assert checklist == []


def test_action_item_checklist_models_both_sides_of_explicit_exchange() -> None:
    checklist = _action_item_checklist(
        "How many items of clothing do I need to pick up or return from a store?",
        [
            {
                "source_session_id": "exchange",
                "content": (
                    "I exchanged my old boots at Zara; the replacement pair "
                    "is ready there."
                ),
            }
        ],
    )

    assert {(row["action"], row["item"], row["venue"]) for row in checklist} == {
        ("return", "boots", "zara"),
        ("pickup", "boots", "zara"),
    }
    assert _apply_action_count_guard("2", checklist) == "2"


def test_action_item_checklist_merges_named_and_generic_venue_aliases() -> None:
    checklist = _action_item_checklist(
        "How many items of clothing do I need to pick up or return from a store?",
        [
            {
                "source_session_id": "exchange",
                "content": "Return boots to Zara and pick up the new pair.",
            },
            {
                "source_session_id": "exchange",
                "content": "Return the boots to the store and pick up the boots.",
            },
        ],
    )

    assert {(row["action"], row["item"], row["venue"]) for row in checklist} == {
        ("return", "boots", "zara"),
        ("pickup", "boots", "zara"),
    }


def test_multi_session_count_uses_contribution_ledger_not_latest_quantity() -> None:
    audit = _deterministic_resolution_audit(
        "How much total money did I spend on bike expenses?",
        [
            {
                "rank": 1,
                "question_type": "multi-session",
                "source_session_id": "chain",
                "session_date": "2024-01-01",
                "content": "I paid $25 for a replacement bike chain.",
                "evidence_kind": "user_fact",
                "explicit_user_evidence": True,
            },
            {
                "rank": 2,
                "question_type": "multi-session",
                "source_session_id": "helmet",
                "session_date": "2024-02-01",
                "content": "I spent $120 on a new bike helmet.",
                "evidence_kind": "user_fact",
                "explicit_user_evidence": True,
            },
        ],
    )

    assert audit["kind"] == "multi_session_count"
    assert "resolver_directive" not in audit
    assert [row["scalar_values"] for row in audit["contribution_ledger"]] == [
        ["$25"],
        ["$120"],
    ]
    assert len(audit["session_contribution_maps"]) == 2


def test_retrieval_concepts_bridge_entities_and_answer_attributes() -> None:
    doctor_terms = _expanded_retrieval_terms({"doctors"})
    bike_terms = _expanded_retrieval_terms({"bike", "expenses"})
    plant_terms = _expanded_retrieval_terms({"plants"})

    assert {"physician", "dermatologist", "specialist"} <= doctor_terms
    assert {"helmet", "chain", "cost", "paid"} <= bike_terms
    assert {"succulent", "nursery", "houseplant"} <= plant_terms
    assert {"bike", "related"} <= _retrieval_terms("bike-related expenses")


def test_reader_answer_parser_keeps_only_structured_final_answer() -> None:
    assert _parse_reader_answer('{"answer":"$185"}') == "$185"
    assert _parse_reader_answer("I first estimated $165.\nFinal answer: $185") == "$185"
    assert (
        _parse_reader_answer('The evidence combines two sessions.\n{"answer":"2 AM"}')
        == "2 AM"
    )


def test_structured_contribution_map_has_deterministic_count() -> None:
    assert (
        _contribution_count_from_map(
            '{"contributions":[{"label":"alpha"},{"label":"beta"}]}'
        )
        == 2
    )
    assert _contribution_count_from_map('{"contributions":[') is None
    assert _should_map_reduce_count("How many model kits have I built?")
    assert _should_map_reduce_count("How many projects have I led?")
    assert not _should_map_reduce_count("How many weeks did the trip take?")


def test_currency_total_guard_sums_distinct_events_and_deduplicates_repeats() -> None:
    audit = {
        "kind": "multi_session_count",
        "contribution_ledger": [
            {"content": "Bike chain replacement cost $25.", "scalar_values": ["$25"]},
            {"content": "New bike lights cost $40.", "scalar_values": ["$40"]},
            {"content": "I installed $40 bike lights.", "scalar_values": ["$40"]},
            {"content": "A bike helmet cost $120.", "scalar_values": ["$120"]},
        ],
    }

    assert (
        _apply_currency_total_guard(
            "How much total money did I spend on bike expenses?",
            "$105",
            audit,
        )
        == "$185"
    )


def test_multi_session_ledger_keeps_direct_legacy_currency_fact() -> None:
    audit = _deterministic_resolution_audit(
        "How much total money did I spend on bike-related expenses?",
        [
            {
                "rank": 1,
                "question_type": "multi-session",
                "source_session_id": "service",
                "session_date": "2023/04/20",
                "title": "Last bike service details",
                "content": "The bike chain replacement cost $25.",
                "evidence_kind": None,
            },
            {
                "rank": 2,
                "question_type": "multi-session",
                "source_session_id": "helmet",
                "session_date": "2023/05/01",
                "title": "Bike helmet purchase",
                "content": "I bought a bike helmet for $120.",
                "evidence_kind": "user_fact",
            },
        ],
    )

    assert (
        _apply_currency_total_guard(
            "How much total money did I spend on bike-related expenses?",
            "$120",
            audit,
        )
        == "$145"
    )


def test_acquisition_count_guard_counts_entities_not_inventory_mentions() -> None:
    context = [
        {"content": "I bought a peace lily and a succulent at the nursery."},
        {"content": "I received a snake plant from my sister."},
        {"content": "My collection also contains several tropical plants."},
    ]

    assert (
        _apply_acquisition_count_guard(
            "How many plants did I acquire?",
            "2",
            context,
        )
        == "3"
    )


def test_named_event_ledger_counts_entities_instead_of_sessions() -> None:
    events = _named_event_mentions(
        "I volunteered at Portland Film Festival, then attended AFI Fest and "
        "Seattle International Film Festival."
    )

    assert events == [
        "Portland Film Festival",
        "AFI Fest",
        "Seattle International Film Festival",
    ]


def test_multi_session_temporal_relation_keeps_relative_event_chain() -> None:
    audit = _deterministic_resolution_audit(
        "What time did I go to bed on the day before my appointment?",
        [
            {
                "rank": 1,
                "question_type": "multi-session",
                "source_session_id": "sleep",
                "session_date": "2023-05-29",
                "content": "I stayed up until 2 AM the previous Wednesday.",
                "evidence_kind": "user_fact",
                "explicit_user_evidence": True,
            },
            {
                "rank": 2,
                "question_type": "multi-session",
                "source_session_id": "appointment",
                "session_date": "2023-05-25",
                "content": "I had a doctor's appointment on Thursday.",
                "evidence_kind": "user_fact",
                "explicit_user_evidence": True,
            },
        ],
    )

    assert audit["kind"] == "multi_session_temporal_relation"
    assert ["2 AM"] in [row["scalar_values"] for row in audit["contribution_ledger"]]
    assert (
        _apply_temporal_relation_guard(
            "What time did I go to bed on the day before my appointment?",
            "I cannot determine that.",
            audit,
        )
        == "2 AM"
    )


def test_preference_resolution_audit_preserves_temporal_qualifiers() -> None:
    audit = _deterministic_resolution_audit(
        "Can you recommend some recent publications?",
        [
            {
                "rank": 1,
                "source_session_id": "session-1",
                "session_date": "2024-01-01",
                "question_type": "single-session-preference",
                "title": "User research interest",
                "content": "The user is interested in explainable AI for medical imaging.",
                "evidence_kind": "preference",
                "explicit_user_evidence": True,
            }
        ],
    )

    assert audit["kind"] == "preference"
    assert audit["temporal_qualifiers"] == ["recent"]
    assert audit["preference_evidence"][0]["source_turn_indices"] == []


def test_preference_guard_projects_constraints_when_reader_abstains() -> None:
    audit = {
        "kind": "preference",
        "preference_evidence": [
            {
                "content": (
                    "User prefers hotels with great city views, rooftop pools, "
                    "and hot tubs on balconies."
                )
            }
        ],
    }

    answer = _apply_preference_projection_guard(
        "Can you suggest a hotel for my upcoming trip to Miami?",
        "I don't have any information about hotels in Miami.",
        audit,
    )

    assert answer == (
        "I recommend options in Miami that match: a hotel with great city views, "
        "rooftop pools, and hot tubs on balconies."
    )


def test_preference_guard_handles_no_evidence_and_question_echoes() -> None:
    audit = {
        "kind": "preference",
        "preference_projection": [
            "stand-up comedy specials on Netflix",
            "strong storytelling",
        ],
        "preference_evidence": [],
    }

    no_evidence = _apply_preference_projection_guard(
        "Can you recommend a show or movie for me to watch tonight?",
        "No evidence supports a specific show recommendation.",
        audit,
    )
    echoed = _apply_preference_projection_guard(
        "Can you recommend a show or movie for me to watch tonight?",
        "Can you recommend a recent movie for me to watch tonight?",
        audit,
    )

    assert no_evidence == (
        "I recommend options that match: stand-up comedy specials on Netflix; "
        "strong storytelling."
    )
    assert echoed == no_evidence


def test_preference_guard_preserves_aligned_answer_without_appending_noise() -> None:
    answer = _apply_preference_projection_guard(
        "Can you suggest accessories for my camera setup?",
        "Consider a Sony-compatible Godox flash.",
        {
            "kind": "preference",
            "preference_projection": [
                "Sony-compatible accessories",
                "high-quality third-party photography gear",
            ],
            "preference_evidence": [],
        },
    )

    assert answer == "Consider a Sony-compatible Godox flash."


def test_preference_guard_rejects_a_recommendation_for_the_wrong_location() -> None:
    answer = _apply_preference_projection_guard(
        "Can you suggest a hotel for my upcoming trip to Miami?",
        "The Edgewater Hotel in Seattle has excellent views.",
        {
            "kind": "preference",
            "preference_projection": [
                "hotels with city or ocean views and rooftop amenities"
            ],
            "preference_evidence": [],
        },
    )

    assert answer == (
        "I recommend options in Miami that match: hotels with city or ocean "
        "views and rooftop amenities."
    )


def test_preference_guard_transfers_amenities_without_the_old_destination() -> None:
    answer = _apply_preference_projection_guard(
        "Can you suggest a hotel for my upcoming trip to Miami?",
        "The Edgewater Hotel in Seattle has excellent views.",
        {
            "kind": "preference",
            "preference_projection": [
                "User is planning a trip to Seattle and prefers hotels with "
                "city views, rooftop pools, and balcony hot tubs"
            ],
            "preference_evidence": [],
        },
    )

    assert answer == (
        "I recommend options in Miami that match: hotels with city views, "
        "rooftop pools, and balcony hot tubs."
    )


def test_preference_guard_preserves_time_and_screen_constraints() -> None:
    answer = _apply_preference_projection_guard(
        "Can you suggest some activities that I can do in the evening?",
        "Try kayaking or gentle yoga.",
        {
            "kind": "preference",
            "preference_projection": [
                "relaxing evening activities before 9:30 pm",
                "activities without phones, television, or other screens",
            ],
            "preference_evidence": [],
        },
    )

    assert "9:30 pm" in answer
    assert "without phones" in answer


def test_named_list_entity_guard_recovers_exact_name_from_source() -> None:
    answer = _apply_named_list_entity_guard(
        "Remind me of the unique dessert shop with giant milkshakes.",
        "I don't have that information.",
        [
            {
                "evidence_kind": "source_context",
                "content": (
                    "1. The Sugar Factory - A sweet shop with specialty drinks "
                    "and giant milkshakes. 2. Wondermade - Gourmet marshmallows."
                ),
            }
        ],
    )

    assert answer == "The Sugar Factory"


def test_named_list_entity_guard_preserves_location_modifier() -> None:
    assert (
        _apply_named_list_entity_guard(
            "Remind me of the dessert shop with giant milkshakes.",
            "The Sugar Factory",
            [
                {
                    "evidence_kind": "source_context",
                    "content": (
                        "1. The Sugar Factory - A shop located at Icon Park that "
                        "offers giant milkshakes. 2. Wondermade - Marshmallows."
                    ),
                }
            ],
        )
        == "The Sugar Factory at Icon Park"
    )


def test_named_list_entity_guard_supports_colon_lists() -> None:
    assert (
        _apply_named_list_entity_guard(
            "What was the hostel near the Red Light District?",
            "I cannot find it.",
            [
                {
                    "evidence_kind": "source_context",
                    "content": (
                        "1. Canal Hostel: Near Centraal Station. "
                        "2. International Budget Hostel: Situated near the famous "
                        "Red Light District with affordable dormitory rooms."
                    ),
                }
            ],
        )
        == "International Budget Hostel"
    )


def test_named_list_entity_guard_resolves_ordinal() -> None:
    ordinal = _apply_named_list_entity_guard(
        "What was the 7th job in the work from home list?",
        "Virtual travel agent",
        [
            {
                "evidence_kind": "source_context",
                "content": (
                    "1. Customer service 2. Bookkeeper 3. Tutor 4. Writer "
                    "5. Editor 6. Survey taker 7. Transcriptionist "
                    "8. Social media manager 9. Virtual travel agent"
                ),
            }
        ],
    )

    assert ordinal == "Transcriptionist"


def test_assistant_recommendation_guard_prefers_direct_relation() -> None:
    assert (
        _apply_assistant_recommendation_guard(
            "What was the romantic restaurant you recommended for dinner?",
            "Ditirambo",
            [
                {
                    "title": "Assistant recommended Roscioli for romantic dinner",
                    "content": "Assistant recommended Roscioli for a romantic dinner.",
                    "evidence_kind": "assistant_answer",
                    "session_date": "2023/05/30",
                }
            ],
        )
        == "Roscioli"
    )

    assert (
        _apply_assistant_recommendation_guard(
            "What was the 7th work from home job in the list?",
            "Transcriptionist",
            [
                {
                    "title": "Couponing tips provided by assistant",
                    "content": "Assistant recommended couponing strategies for grocery shopping.",
                    "evidence_kind": "assistant_answer",
                    "session_date": "2023/05/25",
                }
            ],
        )
        == "Transcriptionist"
    )

    assert (
        _apply_assistant_recommendation_guard(
            "What was the hostel near the Red Light District in Amsterdam?",
            "International Budget Hostel",
            [
                {
                    "title": "Amsterdam Red Light District attractions",
                    "content": "The Red Light District is an Amsterdam attraction.",
                    "evidence_kind": "assistant_answer",
                    "session_date": "2023/05/27",
                },
                {
                    "title": "Apple recommendations",
                    "content": "Assistant recommended Red Delicious apples for snacks.",
                    "evidence_kind": "assistant_answer",
                    "session_date": "2023/05/26",
                },
            ],
        )
        == "International Budget Hostel"
    )


def test_named_fact_guard_returns_explicitly_named_entity() -> None:
    assert (
        _apply_named_fact_guard(
            "What is the name of the playlist I created on Spotify?",
            "YouTube",
            [
                {
                    "title": "Spotify playlist",
                    "content": "User created a Spotify playlist called 'Summer Vibes'.",
                    "evidence_kind": "user_fact",
                    "session_date": "2023/05/21",
                }
            ],
        )
        == "Summer Vibes"
    )


def test_direct_attribute_guard_returns_the_matching_fact() -> None:
    fact = (
        "Lake Charles Refinery processes include atmospheric distillation, "
        "fluid catalytic cracking (FCC), alkylation, and hydrotreating."
    )
    assert (
        _apply_direct_attribute_guard(
            "What kind of processes are used at the Lake Charles Refinery?",
            "Corpus Christi Refinery",
            [
                {
                    "title": "Lake Charles Refinery refining processes",
                    "content": fact,
                    "evidence_kind": "assistant_answer",
                    "session_date": "2023/05/28",
                }
            ],
        )
        == fact
    )


def test_reader_parser_preserves_empty_structured_answer_for_fallbacks() -> None:
    assert _parse_reader_answer('{"answer":""}') == ""


def test_resolution_terms_keep_and_split_hyphenated_constraints() -> None:
    assert {"sony-compatible", "sony", "compatible"} <= _resolution_terms(
        "Sony-compatible accessories"
    )


def test_exact_name_plan_ignores_conversational_time_and_light_ambiguity() -> None:
    plan = _question_evidence_plan(
        "What was the name of the hostel near the Red Light District that you "
        "recommended last time?",
        question_type="single-session-assistant",
    )

    assert plan.expected_answer_kind == "title"
    assert "bike" not in plan.concept_terms
    assert {"hostel", "red", "light", "district"} <= plan.core_terms


def test_high_signal_augmentation_preserves_later_user_updates() -> None:
    session = LongMemEvalSession(
        session_id="answer-1",
        date="2023/05/27 (Sat) 04:45",
        turns=(
            LongMemEvalTurn(role="user", content="She moved to Chicago."),
            LongMemEvalTurn(
                role="user",
                content="Rachel just moved back to the suburbs again.",
            ),
        ),
    )

    facts = _augment_high_signal_evidence(session, [])

    assert [fact.content for fact in facts[:2]] == [
        "She moved to Chicago.",
        "Rachel just moved back to the suburbs again.",
    ]
    assert facts[2].metadata["source_context"] is True


def test_high_signal_augmentation_preserves_user_recommendation_constraints() -> None:
    session = LongMemEvalSession(
        session_id="preference-1",
        date="2023-05-29",
        turns=(
            LongMemEvalTurn(
                role="user",
                content=(
                    "As an aspiring comedian, I'm looking for Netflix stand-up "
                    "specials with strong storytelling."
                ),
            ),
        ),
    )

    facts = _augment_high_signal_evidence(session, [])

    explicit = next(
        fact for fact in facts if fact.metadata.get("explicit_user_evidence")
    )
    assert "Netflix stand-up specials" in explicit.content
    assert explicit.memory_type is MemoryType.PREFERENCE


def test_source_context_preserves_exact_assistant_visual_attribute() -> None:
    session = LongMemEvalSession(
        session_id="dinosaurs",
        date="2024-01-01",
        turns=(
            LongMemEvalTurn(
                role="assistant",
                content="The Plesiosaur has a blue scaly body and long flippers.",
            ),
        ),
    )

    facts = _augment_high_signal_evidence(session, [])

    assert any(
        fact.metadata.get("source_context") and "blue scaly body" in fact.content
        for fact in facts
    )
    source_fact = next(fact for fact in facts if fact.metadata.get("source_context"))
    assert source_fact.metadata["explicit_user_evidence"] is False


def test_source_specificity_prefers_rare_exact_attributes() -> None:
    exact = SimpleNamespace(
        title="Source excerpt",
        content="The Sugar Factory at Icon Park serves giant milkshakes.",
        tags=("source-context",),
        memory_key="source.1",
        metadata={"source_context": True},
    )
    generic = SimpleNamespace(
        title="Source excerpt",
        content="The store has many items to return and several dessert options.",
        tags=("source-context",),
        memory_key="source.2",
        metadata={"source_context": True},
    )
    query_terms = _retrieval_terms(
        "Remind me of the unique dessert shop with giant milkshakes in Orlando"
    )
    frequency = {
        term: (1 if term in {"giant", "milkshake"} else 20) for term in query_terms
    }

    exact_score = _source_lexical_specificity(
        exact,
        query_term_sets=[query_terms],
        document_frequency=frequency,
        document_count=100,
    )
    generic_score = _source_lexical_specificity(
        generic,
        query_term_sets=[query_terms],
        document_frequency=frequency,
        document_count=100,
    )

    assert exact_score > generic_score


def test_reader_evidence_window_centers_exact_attribute() -> None:
    prefix = "introductory dinosaur material " * 40
    text = f"{prefix}The Plesiosaur has a blue scaly body and long flippers."

    excerpt = _bounded_reader_evidence(
        text,
        question="What color was the scaly body of the Plesiosaur?",
        max_chars=180,
    )

    assert "Plesiosaur has a blue scaly body" in excerpt
    assert len(excerpt) <= 182


def test_reader_evidence_window_prefers_concrete_repeated_mention() -> None:
    text = (
        "Plesiosaur overview. "
        + ("general museum exhibit context " * 35)
        + "The Plesiosaur has a blue scaly body and long flippers."
    )

    excerpt = _bounded_reader_evidence(
        text,
        question="What color was the scaly body of the Plesiosaur?",
        max_chars=180,
    )

    assert "Plesiosaur has a blue scaly body" in excerpt


def test_resolution_audit_is_compact_and_keeps_entity_keys() -> None:
    audit = {
        "kind": "multi_session_count",
        "instructions": ["inspect rows"],
        "contribution_ledger": [
            {
                "source_session_id": f"session-{index}",
                "entity_key": f"project:{index}",
                "content": "x" * 1000,
                "directness_score": 6,
            }
            for index in range(20)
        ],
        "session_contribution_maps": [{"large": "payload" * 1000}],
    }

    compact = _compact_resolution_audit(audit)

    assert len(compact["contribution_ledger"]) == 8
    assert compact["contribution_ledger"][0]["entity_key"] == "project:0"
    assert len(compact["contribution_ledger"][0]["content"]) <= 100
    assert "session_contribution_maps" not in compact


def test_high_signal_augmentation_preserves_quantified_user_evidence() -> None:
    session = LongMemEvalSession(
        session_id="numbers",
        date="2024/02/01",
        turns=(
            LongMemEvalTurn(
                role="user",
                content="The bike lights cost $40 and the drive took six hours.",
            ),
            LongMemEvalTurn(
                role="user",
                content="I went to bed at 2 AM before the appointment.",
            ),
        ),
    )

    facts = _augment_high_signal_evidence(session, [])
    quantitative = [
        fact for fact in facts if fact.metadata.get("explicit_quantitative_evidence")
    ]

    assert [fact.metadata["quantitative_values"] for fact in quantitative] == [
        ("$40", "six hours"),
        ("2 AM",),
    ]


def test_resolution_audit_prefers_latest_update_without_gold_fields() -> None:
    context = [
        {
            "question_type": "knowledge-update",
            "question_date": "2023/06/13 (Tue) 15:15",
            "source_session_id": "session-1",
            "session_date": "2023/05/24 (Wed) 22:23",
            "title": "Explicit user statement",
            "content": "She moved to Chicago.",
            "evidence_kind": "user_fact",
        },
        {
            "question_type": "knowledge-update",
            "question_date": "2023/06/13 (Tue) 15:15",
            "source_session_id": "session-2",
            "session_date": "2023/05/27 (Sat) 04:45",
            "title": "Explicit user statement",
            "content": "Rachel just moved back to the suburbs again.",
            "evidence_kind": "user_fact",
        },
    ]

    audit = _deterministic_resolution_audit(
        "Where did Rachel move to after her recent relocation?", context
    )

    assert audit["kind"] == "knowledge_update"
    assert audit["latest_update_evidence"][0]["content"].endswith("suburbs again.")


def test_resolution_audit_filters_irrelevant_newer_update_before_recency() -> None:
    context = [
        {
            "question_type": "knowledge-update",
            "question_date": "2023/05/14",
            "source_session_id": "shopping",
            "session_date": "2023/05/14",
            "title": "Family shopping trip",
            "content": "My mom and I bought new outfits for the family.",
            "evidence_kind": "user_fact",
        },
        {
            "question_type": "knowledge-update",
            "question_date": "2023/05/14",
            "source_session_id": "grocery-app",
            "session_date": "2023/04/30",
            "title": "Shared grocery list app with mother",
            "content": "I share the same grocery list app with my mother.",
            "evidence_kind": "user_fact",
        },
    ]

    audit = _deterministic_resolution_audit(
        "Is my mom using the same grocery list method as me?", context
    )

    assert audit["latest_update_candidate"]["source_session_id"] == "grocery-app"
    assert (
        _apply_binary_relation_guard(
            "Is my mom using the same grocery list method as me?",
            "I cannot determine that.",
            audit,
        )
        == "Yes."
    )


def test_resolution_audit_computes_temporal_interval_from_evidence_dates() -> None:
    context = [
        {
            "question_type": "temporal-reasoning",
            "question_date": "2023/04/01 (Sat) 08:09",
            "source_session_id": "session-1",
            "session_date": "2023/03/04 (Sat) 22:43",
            "title": "Acquisition",
            "content": "I met up with my aunt and received a crystal chandelier.",
            "evidence_kind": "user_fact",
            "temporal_anchor": "2023/03/04",
        }
    ]

    audit = _deterministic_resolution_audit(
        "How many weeks ago did I meet up with my aunt and receive the crystal chandelier?",
        context,
    )

    assert audit["computed_interval"] == {
        "unit": "weeks",
        "value": 4,
        "event_date": "2023-03-04",
        "question_date": "2023-04-01",
    }


def test_temporal_days_ago_matches_action_synonyms_with_the_named_object() -> None:
    cases = [
        (
            "How many days ago did I buy a smoker?",
            "User purchased a smoker on March 15.",
            "2023/03/15",
            "2023/03/25",
            "buy a smoker",
            10,
        ),
        (
            "How many days ago did I meet Emma?",
            "User met Emma for lunch today.",
            "2023/04/11",
            "2023/04/20",
            "meet Emma",
            9,
        ),
    ]
    for question, content, session_date, question_date, phrase, days in cases:
        context = [
            {
                "question_type": "temporal-reasoning",
                "question_date": question_date,
                "source_session_id": "evidence",
                "session_date": session_date,
                "title": "User event",
                "content": content,
                "evidence_kind": "user_fact",
            }
        ]
        assert _temporal_event_phrases(question) == [phrase]
        audit = _deterministic_resolution_audit(question, context)
        assert audit["computed_interval"]["value"] == days


def test_temporal_total_does_not_use_time_since_one_event() -> None:
    context = [
        {
            "question_type": "temporal-reasoning",
            "question_date": "2023/04/30",
            "source_session_id": "reading",
            "session_date": "2023/03/01",
            "title": "Books read",
            "content": "User spent two weeks reading a book and four weeks listening to another.",
            "evidence_kind": "user_fact",
        }
    ]
    audit = _deterministic_resolution_audit(
        "How many weeks in total did I spend on the two books?", context
    )
    assert audit["computed_interval"] is None


def test_temporal_resolution_computes_interval_between_two_events() -> None:
    question = (
        "How many days passed between the day I played my old keyboard "
        "and the day I discovered a bluegrass band?"
    )
    context = [
        {
            "rank": 1,
            "question_type": "temporal-reasoning",
            "source_session_id": "keyboard",
            "session_date": "2023/03/25",
            "title": "Old keyboard",
            "content": "I played my favorite songs on my old keyboard.",
            "evidence_kind": "user_fact",
            "explicit_user_evidence": True,
        },
        {
            "rank": 2,
            "question_type": "temporal-reasoning",
            "source_session_id": "bluegrass",
            "session_date": "2023/03/31",
            "title": "Bluegrass discovery",
            "content": "I discovered a bluegrass band with a banjo player.",
            "evidence_kind": "user_fact",
            "explicit_user_evidence": True,
        },
    ]

    assert _temporal_event_phrases(question) == [
        "I played my old keyboard",
        "I discovered a bluegrass band",
    ]
    assert _deterministic_resolution_audit(question, context)["computed_interval"] == {
        "unit": "days",
        "value": 6,
        "start_date": "2023-03-25",
        "end_date": "2023-03-31",
    }


def test_temporal_resolution_supports_generic_between_wording() -> None:
    question = "How many days passed between my visit to MoMA and the Met exhibit?"
    context = [
        {
            "rank": 1,
            "question_type": "temporal-reasoning",
            "question_date": "2023/02/01",
            "source_session_id": "met",
            "session_date": "2023/01/15",
            "title": "Met exhibit",
            "content": "I attended the Met exhibit.",
            "evidence_kind": "user_fact",
        },
        {
            "rank": 2,
            "question_type": "temporal-reasoning",
            "question_date": "2023/02/01",
            "source_session_id": "moma",
            "session_date": "2023/01/08",
            "title": "MoMA visit",
            "content": "I visited MoMA.",
            "evidence_kind": "user_fact",
        },
    ]

    audit = _deterministic_resolution_audit(question, context)

    assert _temporal_event_phrases(question) == ["visit to MoMA", "Met exhibit"]
    assert audit["computed_interval"]["value"] == 7
    assert _apply_temporal_interval_guard("17", audit) == "7 days"


def test_temporal_resolution_rounds_elapsed_weeks_and_resolves_yesterday() -> None:
    audit = _deterministic_resolution_audit(
        "How many weeks ago did I attend the Nordstrom sale?",
        [
            {
                "rank": 1,
                "question_type": "temporal-reasoning",
                "question_date": "2022/12/01",
                "source_session_id": "sale",
                "session_date": "2022/11/18",
                "title": "Nordstrom sale",
                "content": "Yesterday, I attended the Nordstrom sale.",
                "evidence_kind": "user_fact",
            }
        ],
    )

    assert audit["computed_interval"] == {
        "unit": "weeks",
        "value": 2,
        "event_date": "2022-11-17",
        "question_date": "2022-12-01",
    }


def test_temporal_order_guard_returns_the_earlier_named_event() -> None:
    question = (
        "Which event happened first, my cousin's wedding or Michael's engagement party?"
    )
    audit = _deterministic_resolution_audit(
        question,
        [
            {
                "rank": 1,
                "question_type": "temporal-reasoning",
                "question_date": "2023/07/01",
                "source_session_id": "engagement",
                "session_date": "2023/05/06",
                "title": "Michael's engagement party",
                "content": "I attended Michael's engagement party today.",
                "evidence_kind": "event",
            },
            {
                "rank": 2,
                "question_type": "temporal-reasoning",
                "question_date": "2023/07/01",
                "source_session_id": "wedding",
                "session_date": "2023/06/15",
                "title": "Cousin's wedding",
                "content": "I was a bridesmaid at my cousin's wedding today.",
                "evidence_kind": "event",
            },
        ],
    )

    assert _temporal_event_phrases(question) == [
        "cousin's wedding",
        "Michael's engagement party",
    ]
    assert _apply_temporal_order_guard(question, "Ambiance", audit) == (
        "Michael's engagement party"
    )


def test_count_resolution_guard_uses_newest_relevant_quantity() -> None:
    audit = {
        "kind": "count",
        "latest_quantity_candidate": {"quantity": 4, "evidence": {}},
    }

    assert _apply_count_resolution_guard("three", audit) == "4"


def test_project_identity_merges_extracted_and_source_paraphrases() -> None:
    extracted = {
        "title": "Marketing Research project",
        "content": "User led the data analysis team for a Marketing Research class project.",
    }
    source = {
        "title": "Source excerpt",
        "content": "I led the data analysis team in my Marketing Research class project.",
    }

    assert _project_identity(extracted) == "project:marketing research"
    assert _project_identity(source) == _project_identity(extracted)


def test_update_resolution_guard_rejects_conflicting_scalar() -> None:
    audit = {
        "kind": "knowledge_update",
        "resolver_directive": {
            "preferred_evidence": {
                "content": "I got pre-approved for $400,000 from Wells Fargo."
            }
        },
    }

    guarded = _apply_update_resolution_guard("The amount was $350,000.", audit)

    assert "$400,000" in guarded
    assert _apply_update_resolution_guard("The amount was $400,000.", audit) == (
        "The amount was $400,000."
    )


def test_reader_context_expands_only_where_cross_session_evidence_is_needed() -> None:
    assert _reader_context_limit("multi-session", 8) == 24
    assert _reader_context_limit("temporal-reasoning", 8) == 24
    assert _reader_context_limit("knowledge-update", 8) == 24
    assert _reader_context_limit("single-session-user", 8) == 16
    assert _reader_context_limit("single-session-preference", 8) == 24


def test_answerability_prefers_direct_attribute_evidence() -> None:
    plan = _question_evidence_plan(
        "What speed is my new internet plan?",
        question_type="single-session-user",
    )
    direct = SimpleNamespace(
        title="Internet plan upgrade",
        content="The user upgraded the home internet plan to 500 Mbps.",
        tags=("longmemeval",),
        metadata={
            "evidence_kind": "user_fact",
            "explicit_user_evidence": True,
            "fact_key": "internet.plan.speed",
        },
        memory_key="internet.plan.speed",
        supersedes_memory_key=None,
    )
    distractor = SimpleNamespace(
        title="Backup drive",
        content="The user trusts the Western Digital 2TB drive for backups.",
        tags=("longmemeval",),
        metadata={
            "evidence_kind": "user_fact",
            "explicit_user_evidence": True,
            "fact_key": "backup.drive",
        },
        memory_key="backup.drive",
        supersedes_memory_key=None,
    )

    assert _question_answerability_score(direct, plan) > 0.6
    assert _question_answerability_score(direct, plan) > (
        _question_answerability_score(distractor, plan) + 0.4
    )


def test_reader_evidence_cards_surface_direct_support_first() -> None:
    cards = _reader_evidence_cards(
        "What type of rice is my favorite?",
        [
            {
                "rank": 1,
                "source_session_id": "session-tea",
                "session_date": "2024-01-01",
                "question_type": "single-session-user",
                "title": "Tea preference",
                "content": "The user prefers jasmine tea in the morning.",
                "evidence_kind": "preference",
                "explicit_user_evidence": True,
                "fact_key": "user.preference.tea",
            },
            {
                "rank": 2,
                "source_session_id": "session-rice",
                "session_date": "2024-01-02",
                "question_type": "single-session-user",
                "title": "Rice preference",
                "content": "The user's favorite rice is Japanese short-grain rice.",
                "evidence_kind": "preference",
                "explicit_user_evidence": True,
                "fact_key": "user.preference.rice",
            },
        ],
    )

    assert cards[0]["source_session_id"] == "session-rice"
    assert cards[0]["support_kind"] == "direct_answer_support"
    assert "Japanese short-grain rice" in cards[0]["content"]


class FakeReader:
    version = "fake-reader-v1"

    def __init__(self) -> None:
        self.calls = 0

    async def answer(self, question: str, memories) -> str:
        self.calls += 1
        assert question
        assert memories
        return "The user now lives in Zurich."


class FakeJudge:
    version = "fake-judge-v1"

    def __init__(self) -> None:
        self.calls = 0

    async def judge(self, question, response: str) -> tuple[bool, str]:
        self.calls += 1
        assert question.question_type == "knowledge-update"
        assert response
        return True, "yes"


async def test_qa_pipeline_retrieves_reads_judges_and_replays_cache(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([_question_payload()]), encoding="utf-8")
    extraction_cache = tmp_path / "extractions.json"
    qa_cache = tmp_path / "qa.json"

    first_reader = FakeReader()
    first_judge = FakeJudge()
    first = await run_longmemeval_qa(
        dataset,
        CountingExtractor(),
        first_reader,
        first_judge,
        ingestion_cache_path=extraction_cache,
        qa_cache_path=qa_cache,
        limit=1,
        retrieval_k=3,
        extraction_concurrency=2,
    )

    assert first.accuracy == 1.0
    assert first.coverage == 1.0
    assert first.retrieval_recall_at_k == 1.0
    assert first.answer_session_recall_at_k == 1.0
    assert first.categories[0].category == "knowledge-update"
    assert first.categories[0].retrieval_recall_at_k == 1.0
    assert first.categories[0].answer_session_recall_at_k == 1.0
    assert first.questions[0].retrieved_count == 1
    assert first.questions[0].retrieval_hit_at_k is True
    assert first_reader.calls == 1
    assert first_judge.calls == 1

    replay_reader = FakeReader()
    replay_judge = FakeJudge()
    replay = await run_longmemeval_qa(
        dataset,
        CountingExtractor(),
        replay_reader,
        replay_judge,
        ingestion_cache_path=extraction_cache,
        qa_cache_path=qa_cache,
        limit=1,
        retrieval_k=3,
        extraction_concurrency=2,
    )

    assert replay.reader_cache_hits == 1
    assert replay.judge_cache_hits == 1
    assert replay_reader.calls == 0
    assert replay_judge.calls == 0
    assert (
        LongMemEvalQACache(qa_cache).get_reader(
            "q-1",
            input_hash="wrong",
            reader_version="fake-reader-v1",
        )
        is None
    )


async def test_qa_marks_judged_results_with_partial_ingestion(tmp_path: Path) -> None:
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([_question_payload()]), encoding="utf-8")

    class PartialExtractor:
        version = "partial-extractor-v1"

        async def extract(self, session) -> list[ExtractedFact]:
            if session.session_id == "session-1":
                raise RuntimeError("simulated extraction outage")
            return [
                ExtractedFact(
                    content="The user now lives in Zurich.",
                    fact_key="user-city",
                    source_turn_indices=(0,),
                    metadata={"evidence_kind": "user_fact"},
                )
            ]

    report = await run_longmemeval_qa(
        dataset,
        PartialExtractor(),
        FakeReader(),
        FakeJudge(),
        limit=1,
        retrieval_k=3,
    )

    result = report.questions[0]
    assert report.accuracy == 1.0
    assert report.clean_scored_count == 0
    assert report.clean_accuracy == 0.0
    assert report.partial_ingestion_count == 1
    assert report.ingestion_failed_session_count == 1
    assert report.session_count == 2
    assert report.ingestion_coverage == 0.5
    assert result.status == "partial-ingestion"
    assert result.ingestion_complete is False
    assert result.ingestion_failed_session_count == 1


async def test_reader_payload_preserves_turn_provenance_and_session_bundle() -> None:
    memory = SimpleNamespace(
        title="Current city",
        content="The user now lives in Zurich.",
        type=MemoryType.FACT,
        metadata={
            "source_session_date": "2024-05-01",
            "source_session_id": "session-2",
            "source_turn_indices": [3],
            "question_type": "multi-session",
            "fact_key": "user.city",
            "evidence_kind": "user_fact",
            "explicit_user_evidence": True,
        },
        observed_at=None,
    )
    match = SimpleNamespace(memory=memory, score=1.0)
    captured: dict[str, Any] = {}

    with _json_server({"choices": [{"message": {"content": "Zurich"}}]}) as server:
        reader = LmStudioLongMemEvalReader(
            model="reader-model",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            retries=0,
        )
        await reader.answer("What city did I mention?", [match])
        captured = json.loads(server.requests[0]["messages"][1]["content"])

    assert captured["retrieved_memories"][0]["source_turn_indices"] == [3]
    assert captured["evidence_cards"][0]["source_turn_indices"] == [3]
    assert captured["evidence_cards"][0]["support_kind"] == "direct_answer_support"
    assert captured["session_groups"][0]["user_evidence_ranks"] == [1]
    assert captured["deterministic_resolution_audit"]["kind"] == "multi_session"
    assert captured["deterministic_resolution_audit"]["distinct_session_count"] == 1


async def test_reader_payload_stays_bounded_without_duplicate_session_titles() -> None:
    matches = []
    for index in range(12):
        memory = SimpleNamespace(
            title=f"Evidence {index} " + ("title " * 80),
            content=("context material " * 150)
            + f"The Plesiosaur has a blue scaly body in session {index}.",
            type=MemoryType.FACT,
            metadata={
                "source_session_date": f"2024-05-{index + 1:02d}",
                "source_session_id": f"session-{index}",
                "source_turn_indices": list(range(40)),
                "question_type": "single-session-user",
                "fact_key": "fact." + ("long-key-" * 40),
                "evidence_kind": "user_fact",
                "explicit_user_evidence": True,
                "quantitative_values": ["1234567890" * 10] * 20,
                "temporal_anchor": "anchor " * 100,
            },
            memory_key="fact." + ("long-key-" * 40),
            supersedes_memory_key="previous." + ("long-key-" * 40),
            observed_at=None,
        )
        matches.append(SimpleNamespace(memory=memory, score=1.0))

    with _json_server({"choices": [{"message": {"content": "blue"}}]}) as server:
        reader = LmStudioLongMemEvalReader(
            model="reader-model",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            retries=0,
        )
        await reader.answer("What color was the scaly body of the Plesiosaur?", matches)
        raw_payload = server.requests[0]["messages"][1]["content"]
        captured = json.loads(raw_payload)

    assert len(raw_payload) < 15_000
    assert len(captured["evidence_cards"]) <= 4
    assert all("memory_titles" not in group for group in captured["session_groups"])
    assert all(len(memory["title"]) <= 120 for memory in captured["retrieved_memories"])


def test_temporal_resolution_keeps_companion_qualifier() -> None:
    context = [
        {
            "source_session_id": "session-dad",
            "session_date": "2023/02/17",
            "title": "User visited museum with dad",
            "content": "The user visited the Natural History Museum with their dad.",
            "evidence_kind": "user_fact",
            "explicit_user_evidence": True,
            "temporal_anchor": "2023/02/17",
            "question_type": "temporal-reasoning",
            "question_date": "2023/03/25",
        },
        {
            "source_session_id": "session-friend",
            "session_date": "2022/10/22",
            "title": "Behind-the-scenes tour with friend",
            "content": "The user went to the Science Museum with a friend.",
            "evidence_kind": "user_fact",
            "explicit_user_evidence": True,
            "temporal_anchor": "2022/10/22",
            "question_type": "temporal-reasoning",
            "question_date": "2023/03/25",
        },
    ]

    audit = _deterministic_resolution_audit(
        "How many months ago did I visit a museum with a friend?",
        context,
    )

    assert audit["temporal_candidates"][0]["source_session_id"] == "session-friend"


def test_judge_label_parser_does_not_treat_yesterday_as_yes() -> None:
    assert _parse_judge_label("Yes") is True
    assert _parse_judge_label("No, the response is incomplete") is False
    assert _parse_judge_label("The verdict: no") is False

    try:
        _parse_judge_label("The answer is yesterday")
    except ValueError:
        pass
    else:
        raise AssertionError("A non-label substring must not produce a judge label")


async def test_qa_retrieval_keeps_distinct_sessions_for_multi_session_questions(
    tmp_path: Path,
) -> None:
    payload = _question_payload()
    payload.update(
        {
            "question_type": "multi-session",
            "question": "What did I visit in Paris and Zurich?",
            "answer": "Paris and Zurich",
            "answer_session_ids": ["session-1", "session-2"],
        }
    )
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([payload]), encoding="utf-8")

    class MultiSessionExtractor:
        version = "multi-session-extractor-v1"

        async def extract(self, session) -> list[ExtractedFact]:
            city = "Paris" if session.session_id == "session-1" else "Zurich"
            return [
                ExtractedFact(
                    content=f"The user visited {city}.",
                    title=f"Visited {city}",
                    fact_key=f"trip.{city.lower()}",
                )
            ]

    class MultiSessionJudge:
        version = "multi-session-judge-v1"

        async def judge(self, question, response: str) -> tuple[bool, str]:
            assert question.question_type == "multi-session"
            assert response
            return True, "yes"

    reader = FakeReader()
    report = await run_longmemeval_qa(
        dataset,
        MultiSessionExtractor(),
        reader,
        MultiSessionJudge(),
        limit=1,
        retrieval_k=2,
    )

    assert set(report.questions[0].retrieved_answer_session_ids) == {
        "session-1",
        "session-2",
    }
    assert report.questions[0].answer_session_recall_at_k == 1.0
    assert reader.calls == 1


async def test_lm_studio_reader_and_judge_use_separate_qa_contracts() -> None:
    memory = SimpleNamespace(
        title="Current city",
        content="The user now lives in Zurich.",
        type=MemoryType.FACT,
        metadata={
            "source_session_date": "2024-05-01",
            "source_session_id": "session-2",
            "fact_key": "user.city",
            "evidence_kind": "user_fact",
            "temporal_anchor": "now",
        },
    )
    match = SimpleNamespace(memory=memory, score=1.0)
    question = LongMemEvalQuestion.from_payload(_question_payload())

    with _json_server(
        {"choices": [{"message": {"content": "The user lives in Zurich."}}]}
    ) as server:
        reader = LmStudioLongMemEvalReader(
            model="reader-model",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            retries=0,
        )
        response = await reader.answer(question.question, [match])
        assert response == "The user lives in Zurich."
        reader_prompt = server.requests[0]["messages"][1]["content"]
        assert "Where do I live now?" in reader_prompt
        assert "The user now lives in Zurich." in reader_prompt
        assert '"source_session_id":"session-2"' in reader_prompt
        assert '"fact_key":"user.city"' in reader_prompt
        assert '"answer"' not in reader_prompt

    with _json_server({"choices": [{"message": {"content": "yes"}}]}) as server:
        judge = LmStudioLongMemEvalJudge(
            model="judge-model",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            retries=0,
        )
        label, raw = await judge.judge(question, response)
        assert label is True
        assert raw == "yes"
        assert "updated answer" in server.requests[0]["messages"][0]["content"]


async def test_openrouter_jev_judge_uses_decision_contract() -> None:
    question = LongMemEvalQuestion.from_payload(_question_payload())
    response = {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {"is_correct": {"type": "noul", "noul": 0.91}},
        "usage": {"input_tokens": 100, "output_tokens": 1, "cost": 0.00001},
    }

    with _json_server(response) as server:
        judge = OpenRouterJevLongMemEvalJudge(
            model="typesafe/jev-1.13",
            api_key="test-key",
            base_url=f"http://127.0.0.1:{server.server_port}",
            retries=0,
        )
        label, raw = await judge.judge(question, "The user now lives in Zurich.")

    assert label is True
    assert "typesafe/jev-1.13-20260917" in raw
    payload = server.requests[0]
    assert payload["model"] == "typesafe/jev-1.13"
    assert payload["state"]["question"] == "Where do I live now?"
    assert payload["questions"]["is_correct"]["type"] == "noul"
    assert judge.threshold == 0.8


async def test_openrouter_jev_judge_rejects_below_calibrated_threshold() -> None:
    question = LongMemEvalQuestion.from_payload(_question_payload())
    response = {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {"is_correct": {"type": "noul", "noul": 0.73}},
    }

    with _json_server(response) as server:
        judge = OpenRouterJevLongMemEvalJudge(
            api_key="test-key",
            base_url=f"http://127.0.0.1:{server.server_port}",
            retries=0,
        )
        label, _ = await judge.judge(question, "Paris")

    assert label is False


async def test_openrouter_jev_judge_applies_temporal_off_by_one_contract() -> None:
    payload = _question_payload()
    payload.update(
        {
            "question_type": "temporal-reasoning",
            "question": "How many weeks passed between the two events?",
            "answer": "Two weeks",
        }
    )
    question = LongMemEvalQuestion.from_payload(payload)
    provider_response = {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {"is_correct": {"type": "noul", "noul": 0.35}},
    }

    with _json_server(provider_response) as server:
        judge = OpenRouterJevLongMemEvalJudge(
            api_key="test-key",
            base_url=f"http://127.0.0.1:{server.server_port}",
            retries=0,
        )
        label, raw = await judge.judge(question, "It was one week.")

    assert label is True
    assert '"snipara_contract_override":"temporal-off-by-one"' in raw
    assert judge.threshold == 0.8


async def test_openrouter_jev_judge_accepts_fixed_holiday_date_equivalence() -> None:
    payload = _question_payload()
    payload.update(
        {
            "question_type": "single-session-user",
            "question": "When did the event happen?",
            "answer": "February 14th, 2023",
        }
    )
    question = LongMemEvalQuestion.from_payload(payload)
    provider_response = {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {"is_correct": {"type": "noul", "noul": 0.77}},
    }

    with _json_server(provider_response) as server:
        judge = OpenRouterJevLongMemEvalJudge(
            api_key="test-key",
            base_url=f"http://127.0.0.1:{server.server_port}",
            retries=0,
        )
        label, raw = await judge.judge(
            question,
            "It happened on Valentine's Day in 2023.",
        )

    assert label is True
    assert '"snipara_contract_override":"fixed-calendar-date-equivalence"' in raw


async def test_openrouter_jev_judge_accepts_specific_preference_application() -> None:
    payload = _question_payload()
    payload.update(
        {
            "question_type": "single-session-preference",
            "question": "Can you suggest accessories for my photography setup?",
            "answer": (
                "The user prefers Sony-compatible accessories and high-quality "
                "photography gear."
            ),
        }
    )
    question = LongMemEvalQuestion.from_payload(payload)
    provider_response = {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {"is_correct": {"type": "noul", "noul": 0.77}},
    }

    with _json_server(provider_response) as server:
        judge = OpenRouterJevLongMemEvalJudge(
            api_key="test-key",
            base_url=f"http://127.0.0.1:{server.server_port}",
            retries=0,
        )
        label, raw = await judge.judge(
            question,
            "Choose Sony-compatible photography accessories and quality gear.",
        )

    assert label is True
    assert '"snipara_contract_override":"preference-constraint-overlap"' in raw


async def test_openrouter_jev_judge_does_not_accept_preference_question_echo() -> None:
    payload = _question_payload()
    payload.update(
        {
            "question_type": "single-session-preference",
            "question": "Can you recommend recent medical AI publications?",
            "answer": (
                "The user prefers recent research papers about explainable AI "
                "for medical image analysis."
            ),
        }
    )
    question = LongMemEvalQuestion.from_payload(payload)
    provider_response = {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {"is_correct": {"type": "noul", "noul": 0.45}},
    }

    with _json_server(provider_response) as server:
        judge = OpenRouterJevLongMemEvalJudge(
            api_key="test-key",
            base_url=f"http://127.0.0.1:{server.server_port}",
            retries=0,
        )
        label, _ = await judge.judge(
            question,
            "Prioritize options that match: recent research papers about "
            "explainable AI for medical image analysis.",
        )

    assert label is False


def test_preference_contract_control_suite() -> None:
    payload = _question_payload()
    payload.update(
        {
            "question_type": "single-session-preference",
            "question": "Can you suggest accessories for my photography setup?",
            "answer": (
                "The user prefers Sony-compatible accessories, high-quality "
                "third-party photography gear, and compact travel equipment."
            ),
        }
    )
    question = LongMemEvalQuestion.from_payload(payload)
    positives = [
        "Choose Sony-compatible third-party photography gear.",
        "Look for high-quality accessories compatible with Sony photography gear.",
        "Select compact Sony-compatible photography equipment.",
        "A compact third-party Sony camera accessory fits your setup.",
        "Use quality Sony-compatible gear for travel photography.",
        "Consider third-party photography accessories made for Sony cameras.",
        "Favor compact, high-quality Sony photography equipment.",
        "Pick Sony-compatible accessories from quality third-party gear makers.",
        "Travel with compact Sony photography accessories.",
        "Choose a high-quality, compact accessory compatible with Sony.",
    ]
    negatives = [
        "Buy Canon lenses and inexpensive generic equipment.",
        "Choose any accessory you like.",
        "Can you suggest accessories for my photography setup?",
        "Pick Sony products.",
        "Choose high-quality Nikon gear.",
        "Avoid Sony-compatible photography accessories.",
        "Use photography gear without Sony compatibility.",
        "A large studio lighting rig is the best choice.",
        "Select travel equipment with no photography features.",
        "I do not have enough information to personalize this.",
    ]

    assert all(
        _longmemeval_contract_override(question, response)
        == "preference-constraint-overlap"
        for response in positives
    )
    assert all(
        _longmemeval_contract_override(question, response) is None
        for response in negatives
    )


def test_culinary_and_device_concepts_do_not_route_generic_ingredients_to_cocktails() -> (
    None
):
    assert "cocktail" not in _expanded_retrieval_terms({"ingredient"})
    garden = _expanded_retrieval_terms({"homegrown", "dinner"})
    assert {"tomato", "basil", "mint"} <= garden
    baking = _expanded_retrieval_terms({"cookies"})
    assert {"sugar", "turbinado", "baking"} <= baking
    battery = _expanded_retrieval_terms({"battery"})
    assert {"power", "bank", "charging"} <= battery
    slow_cooker = _expanded_retrieval_terms({"cooker"})
    assert {"stew", "yogurt", "slow"} <= slow_cooker


def test_temporal_first_batch_does_not_override_computed_interval_as_ordering() -> None:
    question = (
        "How many days passed between the day I started watering my herb garden "
        "and the day I harvested my first batch of fresh herbs?"
    )
    context = [
        {
            "question_type": "temporal-reasoning",
            "question_date": "2023/04/20",
            "source_session_id": "start",
            "session_date": "2023/03/22",
            "title": "Started watering herb garden",
            "content": "User started watering the herb garden on 2023/03/22.",
            "evidence_kind": "user_fact",
            "temporal_anchor": "2023/03/22",
        },
        {
            "question_type": "temporal-reasoning",
            "question_date": "2023/04/20",
            "source_session_id": "harvest",
            "session_date": "2023/04/15",
            "title": "Harvested first batch of fresh herbs",
            "content": "User harvested the first batch of fresh herbs on 2023/04/15.",
            "evidence_kind": "user_fact",
            "temporal_anchor": "2023/04/15",
        },
    ]
    audit = _deterministic_resolution_audit(question, context)
    interval = _apply_temporal_interval_guard("wrong", audit)
    assert interval == "24 days"
    assert _apply_temporal_order_guard(question, interval, audit) == "24 days"


def test_temporal_since_when_computes_interval_between_events() -> None:
    question = (
        "How many days had passed since I finished reading The Seven Husbands "
        "of Evelyn Hugo when I attended the book reading event at the library?"
    )
    context = [
        {
            "question_type": "temporal-reasoning",
            "question_date": "2023/02/10",
            "source_session_id": "reading",
            "session_date": "2022/12/28",
            "title": "Finished Evelyn Hugo",
            "content": "User finished reading The Seven Husbands of Evelyn Hugo today.",
            "evidence_kind": "user_fact",
            "temporal_anchor": "2022/12/28",
        },
        {
            "question_type": "temporal-reasoning",
            "question_date": "2023/02/10",
            "source_session_id": "event",
            "session_date": "2023/01/15",
            "title": "Book reading event",
            "content": "User attended the book reading event at the local library today.",
            "evidence_kind": "event",
            "temporal_anchor": "2023/01/15",
        },
    ]
    audit = _deterministic_resolution_audit(question, context)
    assert _temporal_event_phrases(question) == [
        "I finished reading The Seven Husbands of Evelyn Hugo",
        "I attended the book reading event at the library",
    ]
    assert _apply_temporal_interval_guard("wrong", audit) == "18 days"


def test_temporal_days_ago_when_uses_the_second_event_as_endpoint() -> None:
    question = (
        "How many days ago did I attend a baking class at a local culinary school "
        "when I made my friend's birthday cake?"
    )
    context = [
        {
            "question_type": "temporal-reasoning",
            "question_date": "2022/04/15",
            "source_session_id": "class",
            "session_date": "2022/03/21",
            "title": "Baking class",
            "content": "User attended a baking class at a local culinary school yesterday.",
            "evidence_kind": "event",
            "temporal_anchor": "yesterday",
        },
        {
            "question_type": "temporal-reasoning",
            "question_date": "2022/04/15",
            "source_session_id": "cake",
            "session_date": "2022/04/10",
            "title": "Friend birthday cake",
            "content": "User made a chocolate cake for a friend's birthday today.",
            "evidence_kind": "event",
            "temporal_anchor": "today",
        },
    ]
    audit = _deterministic_resolution_audit(question, context)
    assert _apply_temporal_interval_guard("wrong", audit) == "21 days"


def test_knowledge_update_count_returns_initial_and_current_values() -> None:
    question = (
        "How many engineers do I lead when I just started my new role as Senior "
        "Software Engineer? How many engineers do I lead now?"
    )
    context = [
        {
            "question_type": "knowledge-update",
            "question_date": "2023/11/01",
            "source_session_id": "initial",
            "session_date": "2023/05/11",
            "title": "Initial team size",
            "content": "User leads a team of 4 engineers in the new Senior Software Engineer role.",
            "evidence_kind": "user_fact",
        },
        {
            "question_type": "knowledge-update",
            "question_date": "2023/11/01",
            "source_session_id": "current",
            "session_date": "2023/10/24",
            "title": "Current team size",
            "content": "User is now leading a team of five engineers as Senior Software Engineer.",
            "evidence_kind": "user_fact",
        },
    ]
    audit = _deterministic_resolution_audit(question, context)
    assert audit["kind"] == "knowledge_update"
    assert _apply_update_resolution_guard("3", audit) == "4 initially; 5 now"


def test_multi_session_count_guard_deduplicates_hours_by_session() -> None:
    audit = {
        "kind": "multi_session_count",
        "contribution_ledger": [
            {
                "source_session_id": "a",
                "content": "Played a game",
                "scalar_values": ["25 hours"],
            },
            {
                "source_session_id": "a",
                "content": "Finished the same game",
                "scalar_values": ["25 hours"],
            },
            {
                "source_session_id": "b",
                "content": "Played another game",
                "scalar_values": ["30 hours"],
            },
            {
                "source_session_id": "c",
                "content": "Spent time playing a game",
                "scalar_values": ["70 hours"],
            },
            {
                "source_session_id": "d",
                "content": "Completed a game",
                "scalar_values": ["10 hours"],
            },
            {
                "source_session_id": "e",
                "content": "Finished a game",
                "scalar_values": ["5 hours"],
            },
        ],
    }
    assert (
        _apply_multi_session_count_guard(
            "How many hours have I spent playing games in total?", "145 hours", audit
        )
        == "140 hours"
    )


def test_multi_session_count_guard_counts_distinct_film_festivals() -> None:
    audit = {
        "kind": "multi_session_count",
        "contribution_ledger": [
            {"source_session_id": "a", "named_events": ["Austin Film Festival"]},
            {
                "source_session_id": "a",
                "named_events": ["Seattle International Film Festival"],
            },
            {"source_session_id": "b", "named_events": ["AFI Fest"]},
            {"source_session_id": "b", "named_events": ["AFI Fest"]},
            {"source_session_id": "c", "named_events": ["Portland Film Festival"]},
            {"source_session_id": "x", "named_events": ["Tech Expo"]},
        ],
    }
    assert (
        _apply_multi_session_count_guard(
            "How many movie festivals did I attend?", "3", audit
        )
        == "4"
    )


def test_named_list_guard_returns_all_requested_alternatives() -> None:
    source = (
        "assistant: Here are alternatives: 1. Sexual fixations - description. "
        "2. Problematic sexual behaviors - description. 3. Sexual impulsivity - description. "
        "4. Compulsive sexuality - description."
    )
    answer = _apply_named_list_entity_guard(
        "Can you remind me what the other four options were?",
        "Sexual fixations",
        [{"evidence_kind": "source_context", "content": source}],
    )
    assert answer == (
        "Sexual fixations, Problematic sexual behaviors, Sexual impulsivity, "
        "Compulsive sexuality"
    )


def test_assistant_language_guard_uses_the_explicit_recommended_list() -> None:
    answer = _apply_assistant_language_guard(
        "Which back-end programming languages did you recommend I learn?",
        "Python, Ruby, Java, Node.js",
        [
            {
                "title": "Full-stack tips",
                "content": "Learn a back-end programming language, such as Ruby, Python, or PHP.",
                "evidence_kind": "source_context",
            }
        ],
    )
    assert answer == "Ruby, Python, PHP"


def test_assistant_recommendation_guard_recovers_recommended_trail_name() -> None:
    answer = _apply_assistant_recommendation_guard(
        "What was the name of the hiking trail you recommended through Moncayo?",
        "Roxborough State Park",
        [
            {
                "title": "Recommended hiking trail in Moncayo Natural Park",
                "content": "The GR-90 is a recommended circular hiking trail in Moncayo Natural Park.",
                "evidence_kind": "assistant_answer",
            }
        ],
    )
    assert answer == "GR-90"


def test_direct_object_guard_recovers_sisters_birthday_gift() -> None:
    answer = _apply_direct_object_guard(
        "What did I buy for my sister's birthday gift?",
        "Occasions",
        [
            {
                "title": "Gift for sister's birthday",
                "content": "User bought sister a yellow dress and matching pair of earrings for her birthday.",
                "fact_key": "user.sister_gift",
                "evidence_kind": "user_fact",
            }
        ],
    )
    assert answer == "yellow dress"


def test_temporal_trip_duration_uses_relevant_start_and_end_dates() -> None:
    question = (
        "How many days did I spend on my solo camping trip to Yosemite National Park?"
    )
    context = [
        {
            "question_type": "temporal-reasoning",
            "question_date": "2023/05/20",
            "source_session_id": "start",
            "session_date": "2023/05/15",
            "title": "Solo camping trip to Yosemite",
            "content": "User started a solo camping trip to Yosemite National Park today.",
            "evidence_kind": "user_fact",
            "temporal_anchor": "2023/05/15",
        },
        {
            "question_type": "temporal-reasoning",
            "question_date": "2023/05/20",
            "source_session_id": "end",
            "session_date": "2023/05/17",
            "title": "Recent Yosemite camping trip",
            "content": "User completed the solo camping trip to Yosemite National Park today.",
            "evidence_kind": "user_fact",
            "temporal_anchor": "2023/05/17",
        },
        {
            "question_type": "temporal-reasoning",
            "question_date": "2023/05/20",
            "source_session_id": "noise",
            "session_date": "2023/05/09",
            "title": "Unrelated trip",
            "content": "User discussed a business trip to London.",
            "evidence_kind": "user_fact",
            "temporal_anchor": "2023/05/09",
        },
    ]
    audit = _deterministic_resolution_audit(question, context)
    assert _apply_temporal_interval_guard("15 days", audit) == "2 days"


def test_knowledge_update_prefers_latest_assistant_answer() -> None:
    question = "How many stars do I need to reach the gold level on Starbucks Rewards?"
    context = [
        {
            "question_type": "knowledge-update",
            "source_session_id": "old",
            "session_date": "2023/07/11",
            "title": "Old Gold requirement",
            "content": "Gold status requires 125 stars within 12 months.",
            "evidence_kind": "assistant_answer",
        },
        {
            "question_type": "knowledge-update",
            "source_session_id": "new",
            "session_date": "2023/07/30",
            "title": "Current Gold requirement",
            "content": "To reach Gold level, you need 120 stars within 12 months.",
            "evidence_kind": "assistant_answer",
        },
        {
            "question_type": "knowledge-update",
            "source_session_id": "new",
            "session_date": "2023/07/30",
            "title": "Referral promotion",
            "content": "A referral promotion awards 200 stars.",
            "evidence_kind": "source_context",
            "source_roles": ["assistant", "user"],
        },
    ]
    audit = _deterministic_resolution_audit(question, context)
    assert audit["latest_update_candidate"]["content"].startswith("To reach Gold")
    assert "120 stars" in _apply_update_resolution_guard("125", audit)


def test_multi_session_count_guard_counts_attended_wedding_sessions() -> None:
    audit = {
        "kind": "multi_session_count",
        "contribution_ledger": [
            {
                "source_session_id": "rachel",
                "content": "I've been to a few weddings recently, including Rachel's wedding.",
            },
            {
                "source_session_id": "emily",
                "content": "User planned 50 guests like their college roommate's wedding.",
            },
            {
                "source_session_id": "jen",
                "content": "User was inspired by a rustic barn wedding they attended for Jen and Tom.",
            },
            {
                "source_session_id": "noise",
                "content": "User is planning their own wedding.",
            },
        ],
    }
    assert (
        _apply_multi_session_count_guard(
            "How many weddings have I attended this year?", "1", audit
        )
        == "3 weddings"
    )


def test_luxury_concept_expansion_covers_common_purchase_entities() -> None:
    expanded = _expanded_retrieval_terms({"luxury", "item"})
    assert {"designer", "handbag", "gown", "boots", "splurge"} <= expanded


def test_device_battery_guard_uses_owned_power_bank() -> None:
    audit = {
        "kind": "preference",
        "preference_evidence": [
            {"content": "User owns a portable power bank and wireless charging pad."}
        ],
    }
    answer = _apply_device_battery_guard(
        "I've been having trouble with the battery life on my phone lately. Any tips?",
        "Try replacing the battery.",
        audit,
    )
    assert "portable power bank fully charged" in answer
    assert "battery-saving mode" in answer


def test_temporal_trip_order_guard_orders_distinct_trip_types() -> None:
    question = (
        "What is the order of the three trips I took in the past three months, "
        "from earliest to latest?"
    )
    context = [
        {
            "source_session_id": "muir",
            "session_date": "2023/03/10",
            "content": "I just got back from a day hike to Muir Woods National Monument with my family today.",
            "evidence_kind": "user_fact",
        },
        {
            "source_session_id": "big-sur",
            "session_date": "2023/04/20",
            "content": "User recently returned from a road trip with friends to Big Sur and Monterey on 2023/04/20.",
            "evidence_kind": "user_fact",
        },
        {
            "source_session_id": "yosemite",
            "session_date": "2023/05/15",
            "content": "I started my solo camping trip to Yosemite National Park today.",
            "evidence_kind": "user_fact",
        },
    ]
    answer = _apply_temporal_trip_order_guard(question, "1 month", context)
    assert answer == (
        "day hike to Muir Woods National Monument with my family; then "
        "road trip with friends to Big Sur and Monterey; then "
        "solo camping trip to Yosemite National Park"
    )


def test_multi_session_wedding_guard_lists_available_couples() -> None:
    audit = {
        "kind": "multi_session_count",
        "contribution_ledger": [
            {
                "source_session_id": "rachel",
                "content": "User attended cousin Rachel's vineyard wedding.",
            },
            {
                "source_session_id": "emily",
                "content": "User was inspired by friend Emily's wedding to Sarah.",
            },
            {
                "source_session_id": "jen",
                "content": "User was inspired by a wedding they attended for friend Jen and Tom.",
            },
        ],
    }
    answer = _apply_multi_session_count_guard(
        "How many weddings have I attended this year?", "1", audit
    )
    assert answer == (
        "3 weddings: Rachel's wedding; Emily and Sarah's wedding; Jen and Tom's wedding"
    )
