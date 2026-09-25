"""Deterministic, disjoint LongMemEval holdout manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .longmemeval import load_longmemeval_instances


HOLDOUT_MANIFEST_SCHEMA = "snipara.longmemeval.holdout.v1"


def _category(question_id: str, question_type: str) -> str:
    return "abstention" if "_abs" in question_id else question_type


def _dataset_digest(dataset_path: str | Path) -> str:
    return hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()


def _manifest_question_ids(manifest_path: str | Path) -> set[str]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("question_ids"), list):
        raise ValueError("holdout manifest must contain a question_ids array")
    return {str(question_id) for question_id in payload["question_ids"]}


def freeze_longmemeval_holdout(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    per_category: int,
    exclude_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Freeze the first deterministic questions per category after exclusions."""

    if per_category <= 0:
        raise ValueError("per_category must be positive")
    excluded = {str(value) for value in (exclude_ids or set())}
    questions = load_longmemeval_instances(dataset_path)
    selected: list[str] = []
    category_counts: Counter[str] = Counter()
    available_ids = {question.question_id for question in questions}
    missing_exclusions = sorted(excluded - available_ids)
    if missing_exclusions:
        raise ValueError(
            "excluded question IDs are absent from dataset: "
            + ", ".join(missing_exclusions)
        )
    for question in questions:
        if question.question_id in excluded:
            continue
        category = _category(question.question_id, question.question_type)
        if category_counts[category] >= per_category:
            continue
        selected.append(question.question_id)
        category_counts[category] += 1
    if len(selected) < per_category:
        raise ValueError("dataset does not contain enough questions for one category")
    payload: dict[str, Any] = {
        "schema_version": HOLDOUT_MANIFEST_SCHEMA,
        "dataset_sha256": _dataset_digest(dataset_path),
        "question_ids": selected,
        "category_counts": dict(sorted(category_counts.items())),
        "excluded_question_ids": sorted(excluded),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def validate_longmemeval_holdout(
    dataset_path: str | Path,
    manifest_path: str | Path,
    *,
    excluded_manifest_paths: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Validate dataset identity, uniqueness, existence, and disjointness."""

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != HOLDOUT_MANIFEST_SCHEMA:
        raise ValueError("unsupported holdout manifest schema")
    question_ids = [str(value) for value in payload.get("question_ids", [])]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("holdout manifest contains duplicate question IDs")
    if payload.get("dataset_sha256") != _dataset_digest(dataset_path):
        raise ValueError("holdout manifest dataset digest does not match dataset")
    available = {
        question.question_id: _category(question.question_id, question.question_type)
        for question in load_longmemeval_instances(dataset_path)
    }
    missing = sorted(set(question_ids) - set(available))
    if missing:
        raise ValueError("holdout manifest contains unknown question IDs: " + ", ".join(missing))
    overlaps: set[str] = set()
    for excluded_path in excluded_manifest_paths:
        overlaps.update(set(question_ids) & _manifest_question_ids(excluded_path))
    if overlaps:
        raise ValueError(
            "holdout overlaps an excluded manifest: " + ", ".join(sorted(overlaps))
        )
    counts = Counter(available[question_id] for question_id in question_ids)
    return {
        "schema_version": HOLDOUT_MANIFEST_SCHEMA,
        "dataset_sha256": payload["dataset_sha256"],
        "question_count": len(question_ids),
        "category_counts": dict(sorted(counts.items())),
        "excluded_manifest_count": len(excluded_manifest_paths),
        "valid": True,
    }
