"""Public package exports for snipara-memory."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .adapters import InMemoryMemoryStore, JsonFileMemoryStore, get_default_store_path
from .domain import (
    CompactionResult,
    Contradiction,
    ContradictionResolution,
    ContradictionStatus,
    GraveyardEntry,
    GraveyardReason,
    Memory,
    MemoryScope,
    MemoryService,
    MemoryStatus,
    MemoryTier,
    MemoryType,
    Namespace,
    NamespaceType,
    RecallMatch,
    RecallQuery,
    ResolveContradictionRequest,
    SessionMemoryBundle,
    StoreMemoryRequest,
    calculate_confidence_decay,
    classify_memory_tier,
    provenance_key_for_memory,
    select_diverse_matches,
)
from .ports import CacheStore, EmbeddingsProvider, MemoryStore
from .longmemeval import (
    ExtractedFact,
    ExtractionCache,
    FactExtractor,
    HeuristicFactExtractor,
    LmStudioBatchFactExtractor,
    LmStudioFactExtractor,
    LongMemEvalIngestionReport,
    LongMemEvalIngestionResult,
    LongMemEvalQuestion,
    LongMemEvalSession,
    LongMemEvalTurn,
    ingest_longmemeval_dataset,
    ingest_longmemeval_question,
    load_longmemeval_instances,
)
from .qa import (
    LONGMEMEVAL_JUDGE_PROMPT_VERSION,
    LONGMEMEVAL_READER_PROMPT_VERSION,
    LongMemEvalCategoryReport,
    LongMemEvalJudge,
    LongMemEvalQACache,
    LongMemEvalQAReport,
    LongMemEvalQAResult,
    LongMemEvalReader,
    LmStudioLongMemEvalJudge,
    LmStudioLongMemEvalReader,
    official_longmemeval_judge_prompt,
    run_longmemeval_qa,
    stratified_longmemeval_question_ids,
    write_longmemeval_hypotheses,
)


def create_app(service: MemoryService) -> Any:
    """Create the optional FastAPI application on demand.

    Core memory, ingestion, and QA consumers should not need to import the
    web framework just to import ``snipara_memory``.  The package still keeps
    FastAPI as an installation dependency for the API entry point, while this
    lazy boundary makes source checkouts and lightweight library use robust
    when only the core dependencies are available.
    """

    from .api import create_app as _create_app

    return _create_app(service)

try:
    __version__ = version("snipara-memory")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "CacheStore",
    "CompactionResult",
    "Contradiction",
    "ContradictionResolution",
    "ContradictionStatus",
    "EmbeddingsProvider",
    "ExtractedFact",
    "ExtractionCache",
    "FactExtractor",
    "GraveyardEntry",
    "HeuristicFactExtractor",
    "LmStudioBatchFactExtractor",
    "LmStudioFactExtractor",
    "GraveyardReason",
    "InMemoryMemoryStore",
    "JsonFileMemoryStore",
    "Memory",
    "MemoryScope",
    "MemoryService",
    "MemoryStatus",
    "MemoryStore",
    "MemoryTier",
    "MemoryType",
    "LongMemEvalCategoryReport",
    "LongMemEvalJudge",
    "LongMemEvalQACache",
    "LongMemEvalQAReport",
    "LongMemEvalQAResult",
    "LongMemEvalReader",
    "LongMemEvalIngestionReport",
    "LongMemEvalIngestionResult",
    "LongMemEvalQuestion",
    "LongMemEvalSession",
    "LongMemEvalTurn",
    "Namespace",
    "NamespaceType",
    "RecallMatch",
    "RecallQuery",
    "ResolveContradictionRequest",
    "SessionMemoryBundle",
    "StoreMemoryRequest",
    "__version__",
    "calculate_confidence_decay",
    "classify_memory_tier",
    "provenance_key_for_memory",
    "select_diverse_matches",
    "create_app",
    "get_default_store_path",
    "ingest_longmemeval_dataset",
    "ingest_longmemeval_question",
    "load_longmemeval_instances",
    "LmStudioLongMemEvalJudge",
    "LmStudioLongMemEvalReader",
    "LONGMEMEVAL_JUDGE_PROMPT_VERSION",
    "LONGMEMEVAL_READER_PROMPT_VERSION",
    "official_longmemeval_judge_prompt",
    "run_longmemeval_qa",
    "stratified_longmemeval_question_ids",
    "write_longmemeval_hypotheses",
]
