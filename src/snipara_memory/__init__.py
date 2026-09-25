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
from .evidence_graph import (
    AnswerabilityStatus,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceGraphStats,
    EvidenceNode,
    EvidenceNodeKind,
    EvidencePath,
    EvidenceRelation,
    NumericContribution,
    ReasoningResult,
    extract_numeric_contributions,
    reason_over_contributions,
)
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
from .holdout import freeze_longmemeval_holdout, validate_longmemeval_holdout
from .ports import CacheStore, EmbeddingsProvider, MemoryStore
from .qa import (
    LONGMEMEVAL_JUDGE_PROMPT_VERSION,
    LONGMEMEVAL_READER_PROMPT_VERSION,
    LmStudioLongMemEvalJudge,
    LmStudioLongMemEvalReader,
    LongMemEvalCategoryReport,
    LongMemEvalJudge,
    LongMemEvalQACache,
    LongMemEvalQAReport,
    LongMemEvalQAResult,
    LongMemEvalReader,
    OpenRouterJevLongMemEvalJudge,
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
    __version__ = "0.1.2"

__all__ = [
    "LONGMEMEVAL_JUDGE_PROMPT_VERSION",
    "LONGMEMEVAL_READER_PROMPT_VERSION",
    "AnswerabilityStatus",
    "CacheStore",
    "CompactionResult",
    "Contradiction",
    "ContradictionResolution",
    "ContradictionStatus",
    "EmbeddingsProvider",
    "EvidenceEdge",
    "EvidenceGraph",
    "EvidenceGraphStats",
    "EvidenceNode",
    "EvidenceNodeKind",
    "EvidencePath",
    "EvidenceRelation",
    "ExtractedFact",
    "ExtractionCache",
    "FactExtractor",
    "GraveyardEntry",
    "GraveyardReason",
    "HeuristicFactExtractor",
    "freeze_longmemeval_holdout",
    "InMemoryMemoryStore",
    "JsonFileMemoryStore",
    "LmStudioBatchFactExtractor",
    "LmStudioFactExtractor",
    "LmStudioLongMemEvalJudge",
    "LmStudioLongMemEvalReader",
    "LongMemEvalCategoryReport",
    "LongMemEvalIngestionReport",
    "LongMemEvalIngestionResult",
    "LongMemEvalJudge",
    "LongMemEvalQACache",
    "LongMemEvalQAReport",
    "LongMemEvalQAResult",
    "LongMemEvalQuestion",
    "LongMemEvalReader",
    "LongMemEvalSession",
    "LongMemEvalTurn",
    "Memory",
    "MemoryScope",
    "MemoryService",
    "MemoryStatus",
    "MemoryStore",
    "MemoryTier",
    "MemoryType",
    "Namespace",
    "NamespaceType",
    "NumericContribution",
    "OpenRouterJevLongMemEvalJudge",
    "ReasoningResult",
    "RecallMatch",
    "RecallQuery",
    "ResolveContradictionRequest",
    "SessionMemoryBundle",
    "StoreMemoryRequest",
    "__version__",
    "calculate_confidence_decay",
    "classify_memory_tier",
    "create_app",
    "extract_numeric_contributions",
    "get_default_store_path",
    "ingest_longmemeval_dataset",
    "ingest_longmemeval_question",
    "load_longmemeval_instances",
    "validate_longmemeval_holdout",
    "official_longmemeval_judge_prompt",
    "provenance_key_for_memory",
    "reason_over_contributions",
    "run_longmemeval_qa",
    "select_diverse_matches",
    "stratified_longmemeval_question_ids",
    "write_longmemeval_hypotheses",
]
