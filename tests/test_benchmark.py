from __future__ import annotations

from pathlib import Path

from snipara_memory.benchmark import run_benchmark


async def test_benchmark_reports_recall_hit(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"id":"jwt","namespace_id":"demo","query":"How do we handle JWT auth?",'
        '"setup":[{"title":"JWT convention","content":"JWT auth uses RS256 token pairs and refresh tokens.","memory_type":"DECISION"}],'
        '"relevant_indices":[0],"limit":5}\n',
        encoding="utf-8",
    )

    report = await run_benchmark(dataset)

    assert report.case_count == 1
    assert report.recall_at_k == 1.0
    assert report.top1_accuracy == 1.0
