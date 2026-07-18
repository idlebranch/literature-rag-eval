"""Tests for src.eval.export markdown / csv generators."""
import csv
from pathlib import Path

from src.eval.export import to_csv, to_index_md, to_markdown
from src.eval.io import compute_summary
from src.eval.schema import (
    EvalRun,
    JudgeScore,
    QuestionResult,
    RetrievedSource,
    RunConfig,
)


def _sample_run() -> EvalRun:
    run = EvalRun(
        run_id="sample",
        timestamp="2026-05-25T00:00:00",
        config=RunConfig(rag_model="gpt", rag_prompt_version="v3", top_k=5, judge_model="claude"),
        results=[
            QuestionResult(
                qid="q1",
                question="What is X?",
                ideal_answer="X is a mechanism.",
                model_answer="X involves Y and Z [S1].",
                answer_type="mechanism",
                retrieved=[
                    RetrievedSource(sid="S1", source="a.pdf", page=2, distance=0.41,
                                    chunk_text="Y reacts with Z to form …"),
                ],
                judge=JudgeScore(5, 4, 5, 4, "incomplete_answer", "missed factor F", judge_model="claude"),
            ),
            QuestionResult(
                qid="q2",
                question="Compare A and B",
                ideal_answer="A differs from B in ...",
                model_answer="A is fast, B is slow.",
                answer_type="comparison",
                retrieved=[],
                judge=JudgeScore(3, 2, 3, 2, "insufficient_context",
                                 "B not in sources", judge_model="claude"),
            ),
        ],
    )
    run.summary = compute_summary(run)
    return run


def test_to_markdown_writes_toc_and_per_qid_sections(tmp_path: Path):
    run = _sample_run()
    p = to_markdown(run, tmp_path / "out.md", badcase_threshold=4)
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "## Summary" in text
    assert "## 目录" in text
    assert "## q1" in text
    assert "## q2" in text
    # TOC anchor link present
    assert "(#q1)" in text
    # Badcase flag (q2 overall=2 < 4)
    assert "🔴" in text
    # Top-1 streak: faithfulness bar appears
    assert "█" in text


def test_to_csv_flat_schema(tmp_path: Path):
    run = _sample_run()
    p = to_csv(run, tmp_path / "out.csv")
    assert p.exists()
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["qid"] == "q1"
    assert rows[0]["overall"] == "4"
    assert rows[0]["n_retrieved"] == "1"
    assert rows[0]["top1_source"] == "a.pdf"
    assert rows[1]["overall"] == "2"
    assert rows[1]["n_retrieved"] == "0"
    # has_error is 0 for both since no .error set
    assert rows[0]["has_error"] == "0"


def test_to_index_md(tmp_path: Path):
    r1 = _sample_run()
    r2 = _sample_run()
    r2.run_id = "later"
    r2.timestamp = "2026-05-26T00:00:00"
    p = to_index_md([r1, r2], tmp_path / "INDEX.md")
    text = p.read_text(encoding="utf-8")
    assert "later" in text and "sample" in text
    # later should sort first (descending)
    assert text.index("later") < text.index("sample")
