"""Tests for src.eval.badcase report."""
from pathlib import Path

from src.eval.badcase import ERROR_GUIDE, write_badcase_report
from src.eval.io import compute_summary
from src.eval.schema import EvalRun, JudgeScore, QuestionResult, RunConfig


def _run_with_mixed_badcases() -> EvalRun:
    run = EvalRun(
        run_id="bad",
        timestamp="2026-05-25",
        config=RunConfig(judge_model="claude"),
        results=[
            QuestionResult(qid="q1", question="Q1", ideal_answer="", model_answer="",
                           judge=JudgeScore(5, 5, 5, 5, "none", "perfect")),
            QuestionResult(qid="q2", question="Q2", ideal_answer="", model_answer="",
                           judge=JudgeScore(3, 1, 1, 1, "generation_error", "refused")),
            QuestionResult(qid="q3", question="Q3", ideal_answer="", model_answer="",
                           judge=JudgeScore(4, 2, 4, 2, "insufficient_context", "no source")),
            QuestionResult(qid="q4", question="Q4", ideal_answer="", model_answer="",
                           judge=JudgeScore(5, 3, 4, 3, "incomplete_answer", "missed")),
        ],
    )
    run.summary = compute_summary(run)
    return run


def test_write_badcase_report_contains_all_error_type_sections(tmp_path: Path):
    run = _run_with_mixed_badcases()
    p = write_badcase_report(run, tmp_path / "bad.md", badcase_threshold=4)
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    for et in ERROR_GUIDE.keys():
        assert et in text, f"missing section for {et}"
    # q2/q3/q4 are badcases (overall <4) → must appear
    assert "q2" in text
    assert "q3" in text
    assert "q4" in text
    # q1 (overall=5) should not appear under badcase headings
    # but the count column may include it under 'none' line — check both don't conflict
    assert "本批共 **3** 个 badcase" in text


def test_error_guide_has_expected_keys():
    expected = {
        "retrieval_error", "generation_error", "citation_error",
        "incomplete_answer", "insufficient_context", "hallucination",
        "script_error",
    }
    assert set(ERROR_GUIDE.keys()) == expected
    for et, guide in ERROR_GUIDE.items():
        assert "definition" in guide and guide["definition"]
        assert "fixes" in guide and len(guide["fixes"]) >= 2
