import json

import pytest

from snipara_memory.holdout import (
    freeze_longmemeval_holdout,
    validate_longmemeval_holdout,
)


def _dataset_payload() -> list[dict[str, object]]:
    payload = []
    for index, question_type in enumerate(("single-session-user", "multi-session")):
        for offset in range(3):
            payload.append(
                {
                    "question_id": f"q-{index}-{offset}",
                    "question_type": question_type,
                    "question": "What is remembered?",
                    "answer": "A",
                    "haystack_session_ids": [f"s-{index}-{offset}"],
                    "haystack_dates": ["2024-01-01"],
                    "haystack_sessions": [[{"role": "user", "content": "A"}]],
                }
            )
    payload.append(
        {
            "question_id": "q-abs_abs",
            "question_type": "single-session-user",
            "question": "What is not supported?",
            "answer": "I don't know",
            "haystack_session_ids": ["s-abs"],
            "haystack_dates": ["2024-01-01"],
            "haystack_sessions": [[{"role": "user", "content": "A"}]],
        }
    )
    return payload


def test_holdout_is_deterministic_and_validates_disjointness(tmp_path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(_dataset_payload()), encoding="utf-8")
    excluded = tmp_path / "excluded.json"
    excluded.write_text(
        json.dumps({"question_ids": ["q-0-0"]}),
        encoding="utf-8",
    )
    manifest = tmp_path / "holdout.json"

    frozen = freeze_longmemeval_holdout(
        dataset,
        manifest,
        per_category=2,
        exclude_ids={"q-0-0"},
    )

    assert frozen["question_ids"] == [
        "q-0-1",
        "q-0-2",
        "q-1-0",
        "q-1-1",
        "q-abs_abs",
    ]
    validation = validate_longmemeval_holdout(
        dataset,
        manifest,
        excluded_manifest_paths=(excluded,),
    )
    assert validation["valid"] is True
    assert validation["question_count"] == 5


def test_holdout_rejects_overlap(tmp_path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(_dataset_payload()), encoding="utf-8")
    manifest = tmp_path / "holdout.json"
    freeze_longmemeval_holdout(dataset, manifest, per_category=1)
    excluded = tmp_path / "excluded.json"
    excluded.write_text(
        json.dumps({"question_ids": ["q-0-0"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlaps"):
        validate_longmemeval_holdout(
            dataset,
            manifest,
            excluded_manifest_paths=(excluded,),
        )
