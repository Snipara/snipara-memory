from __future__ import annotations

import json
from pathlib import Path

from snipara_memory import (
    ExtractedFact,
    ExtractionCache,
    InMemoryMemoryStore,
    MemoryService,
    MemoryStatus,
    MemoryType,
    ingest_longmemeval_dataset,
    load_longmemeval_instances,
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
    assert new_memory is not None
    assert new_memory.metadata["source_session_id"] == "session-2"
    assert "answer-session" in new_memory.tags
    assert result.superseded_memory_ids == (result.stored_memory_ids[0],)


def questions_hash(dataset: Path, index: int) -> str:
    question = load_longmemeval_instances(dataset)[index]
    return question.sessions[0].content_hash
