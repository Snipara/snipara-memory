"""Reproducible benchmark harness for snipara-memory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .adapters import InMemoryMemoryStore
from .domain import MemoryScope, MemoryService, MemoryType, RecallQuery, StoreMemoryRequest


@dataclass(slots=True)
class BenchmarkCase:
    id: str
    namespace_id: str
    query: str
    setup: list[dict[str, Any]]
    relevant_indices: list[int]
    limit: int = 5


@dataclass(slots=True)
class BenchmarkCaseResult:
    case_id: str
    hit_at_k: bool
    reciprocal_rank: float
    returned_titles: list[str]
    relevant_titles: list[str]


@dataclass(slots=True)
class BenchmarkReport:
    dataset: str
    case_count: int
    recall_at_k: float
    mean_reciprocal_rank: float
    top1_accuracy: float
    cases: list[BenchmarkCaseResult]


async def run_benchmark(dataset_path: str | Path, *, default_limit: int = 5) -> BenchmarkReport:
    dataset = Path(dataset_path)
    cases = load_benchmark_cases(dataset)
    results: list[BenchmarkCaseResult] = []

    for case in cases:
        service = MemoryService(store=InMemoryMemoryStore())
        created = await service.store_memories_bulk(
            [
                _request_from_setup_item(item, namespace_id=case.namespace_id)
                for item in case.setup
            ]
        )
        relevant_memory_ids = {
            created[index].id
            for index in case.relevant_indices
            if index < len(created)
        }
        relevant_titles = [
            created[index].title or created[index].content
            for index in case.relevant_indices
            if index < len(created)
        ]
        matches = await service.semantic_recall(
            RecallQuery(
                namespace_id=case.namespace_id,
                query=case.query,
                limit=case.limit or default_limit,
            )
        )
        returned_ids = [match.memory.id for match in matches]
        returned_titles = [match.memory.title or match.memory.content for match in matches]
        rank = next(
            (index + 1 for index, memory_id in enumerate(returned_ids) if memory_id in relevant_memory_ids),
            None,
        )
        results.append(
            BenchmarkCaseResult(
                case_id=case.id,
                hit_at_k=rank is not None,
                reciprocal_rank=0.0 if rank is None else 1.0 / rank,
                returned_titles=returned_titles,
                relevant_titles=relevant_titles,
            )
        )

    case_count = len(results)
    recall_at_k = sum(1 for result in results if result.hit_at_k) / case_count if case_count else 0.0
    mrr = sum(result.reciprocal_rank for result in results) / case_count if case_count else 0.0
    top1 = sum(1 for result in results if result.reciprocal_rank == 1.0) / case_count if case_count else 0.0
    return BenchmarkReport(
        dataset=str(dataset),
        case_count=case_count,
        recall_at_k=recall_at_k,
        mean_reciprocal_rank=mrr,
        top1_accuracy=top1,
        cases=results,
    )


def render_benchmark_report(report: BenchmarkReport) -> str:
    lines = [
        f"Dataset: {report.dataset}",
        f"Cases: {report.case_count}",
        f"Recall@k: {report.recall_at_k:.3f}",
        f"MRR: {report.mean_reciprocal_rank:.3f}",
        f"Top1 accuracy: {report.top1_accuracy:.3f}",
        "",
    ]
    for case in report.cases:
        status = "hit" if case.hit_at_k else "miss"
        lines.append(f"- {case.case_id}: {status} | relevant={case.relevant_titles} | returned={case.returned_titles[:3]}")
    return "\n".join(lines)


def benchmark_report_as_json(report: BenchmarkReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True)


def load_benchmark_cases(dataset_path: str | Path) -> list[BenchmarkCase]:
    dataset = Path(dataset_path)
    if dataset.suffix.lower() == ".jsonl":
        return [
            BenchmarkCase(**json.loads(line))
            for line in dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    payload = json.loads(dataset.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [BenchmarkCase(**item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return [BenchmarkCase(**item) for item in payload["cases"]]
    raise ValueError(f"Unsupported benchmark dataset format: {dataset}")


def _request_from_setup_item(item: dict[str, Any], *, namespace_id: str) -> StoreMemoryRequest:
    memory_type = MemoryType(item.get("memory_type", MemoryType.FACT.value))
    return StoreMemoryRequest(
        namespace_id=namespace_id,
        title=item.get("title"),
        content=item["content"],
        memory_type=memory_type,
        scope=MemoryScope(item.get("scope", MemoryScope.PROJECT.value)),
        category=item.get("category"),
        source=item.get("source", "benchmark"),
        tags=list(item.get("tags", [])),
        metadata=dict(item.get("metadata", {})),
        confidence=float(item.get("confidence", 0.8)),
    )
