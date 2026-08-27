from __future__ import annotations

import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Iterator

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
    MemoryService,
    MemoryStatus,
    MemoryType,
    LongMemEvalSession,
    LongMemEvalTurn,
    ingest_longmemeval_dataset,
    load_longmemeval_instances,
    official_longmemeval_judge_prompt,
    run_longmemeval_qa,
)
from snipara_memory.longmemeval import (
    _augment_high_signal_evidence,
    _facts_from_lm_studio_batch_response,
    _facts_from_lm_studio_response,
)
from snipara_memory.qa import (
    READER_SYSTEM_PROMPT,
    _action_item_checklist,
    _apply_update_resolution_guard,
    _deterministic_resolution_audit,
    _reader_context_limit,
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
    old_memory = await store.get_memory(result.stored_memory_ids[0])
    new_memory = await store.get_memory(result.stored_memory_ids[1])

    assert len(active) == 1
    assert active[0].content == "The user now lives in Zurich."
    assert old_memory is not None
    assert old_memory.status is MemoryStatus.GRAVEYARD
    assert old_memory.memory_key == "user-city"
    assert new_memory is not None
    assert new_memory.memory_key == "user-city"
    assert new_memory.supersedes_memory_key == "user-city"
    assert new_memory.provenance_key == "session-2"
    assert new_memory.observed_at is not None
    assert new_memory.observed_at.isoformat() == "2024-05-01T00:00:00+00:00"
    assert new_memory.metadata["source_session_id"] == "session-2"
    assert "answer-session" in new_memory.tags
    assert result.superseded_memory_ids == (result.stored_memory_ids[0],)


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
    assert result.superseded_memory_ids == (
        result.stored_memory_ids[0],
        result.stored_memory_ids[1],
    )
    active = await store.list_memories(
        "longmemeval:q-1", statuses=[MemoryStatus.ACTIVE]
    )
    assert len(active) == 1
    assert active[0].content == "The user now lives in Zurich."


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
    assert {memory.metadata["fact_key"] for memory in active} == {
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


def questions_hash(dataset: Path, index: int) -> str:
    question = load_longmemeval_instances(dataset)[index]
    return question.sessions[0].content_hash


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
            payload["response_format"]["json_schema"]["schema"]["properties"][
                "facts"
                ]["maxItems"]
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
    assert schema["properties"]["sessions"]["items"]["properties"]["facts"][
        "maxItems"
        ] == 12


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

    assert ":lmstudio-longmemeval-reader-v24" in reader.version
    assert "dry-cleaning pickup" in READER_SYSTEM_PROMPT
    assert "silently enumerate" in READER_SYSTEM_PROMPT


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
        {"source_session_id": "session-1", "action": "pickup", "item": "boots"},
        {
            "source_session_id": "session-2",
            "action": "pickup",
            "item": "dry cleaning",
        },
        {"source_session_id": "session-3", "action": "pickup", "item": "boots"},
        {"source_session_id": "session-3", "action": "return", "item": "boots"},
    ]


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

    assert [fact.content for fact in facts] == [
        "She moved to Chicago.",
        "Rachel just moved back to the suburbs again.",
    ]
    assert all(fact.metadata["explicit_user_evidence"] for fact in facts)


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
    assert audit["latest_update_evidence"][0]["content"].endswith(
        "suburbs again."
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


def test_update_resolution_guard_rejects_conflicting_scalar() -> None:
    audit = {
        "kind": "knowledge_update",
        "resolver_directive": {
            "preferred_evidence": {
                "content": "I got pre-approved for $400,000 from Wells Fargo."
            }
        },
    }

    guarded = _apply_update_resolution_guard(
        "The amount was $350,000.", audit
    )

    assert "$400,000" in guarded
    assert _apply_update_resolution_guard("The amount was $400,000.", audit) == (
        "The amount was $400,000."
    )


def test_reader_context_expands_only_where_cross_session_evidence_is_needed() -> None:
    assert _reader_context_limit("multi-session", 8) == 24
    assert _reader_context_limit("temporal-reasoning", 8) == 18
    assert _reader_context_limit("knowledge-update", 8) == 24
    assert _reader_context_limit("single-session-user", 8) == 16


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
    assert LongMemEvalQACache(qa_cache).get_reader(
        "q-1",
        input_hash="wrong",
        reader_version="fake-reader-v1",
    ) is None


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

    with _json_server(
        {"choices": [{"message": {"content": "Zurich"}}]}
    ) as server:
        reader = LmStudioLongMemEvalReader(
            model="reader-model",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            retries=0,
        )
        await reader.answer("What city did I mention?", [match])
        captured = json.loads(server.requests[0]["messages"][1]["content"])

    assert captured["retrieved_memories"][0]["source_turn_indices"] == [3]
    assert captured["session_groups"][0]["user_evidence_ranks"] == [1]
    assert captured["deterministic_resolution_audit"]["kind"] == "multi_session"
    assert captured["deterministic_resolution_audit"]["distinct_session_count"] == 1


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
