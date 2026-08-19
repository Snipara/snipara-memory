"""Public package exports for snipara-memory."""

from importlib.metadata import PackageNotFoundError, version

from .adapters import InMemoryMemoryStore, JsonFileMemoryStore, get_default_store_path
from .api import create_app
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
)
from .ports import CacheStore, EmbeddingsProvider, MemoryStore
from .longmemeval import (
    ExtractedFact,
    ExtractionCache,
    FactExtractor,
    HeuristicFactExtractor,
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
    "create_app",
    "get_default_store_path",
    "ingest_longmemeval_dataset",
    "ingest_longmemeval_question",
    "load_longmemeval_instances",
]
