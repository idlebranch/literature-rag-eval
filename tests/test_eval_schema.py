"""Tests for src.eval.schema dataclasses."""
from src.eval.schema import (
    EvalRun,
    HumanReview,
    JudgeScore,
    QuestionResult,
    RetrievedSource,
    RunConfig,
)


def test_retrieved_source_from_dict_handles_missing_fields():
    s = RetrievedSource.from_dict({"sid": "S1", "source": "x.pdf"})
    assert s.sid == "S1"
    assert s.source == "x.pdf"
    assert s.page == 0
    assert s.distance == 0.0
    assert s.chunk_text == ""


def test_judge_score_roundtrip():
    raw = {
        "faithfulness": 4,
        "completeness": 3,
        "citation": 5,
        "overall": 4,
        "error_type": "incomplete_answer",
        "reason": "missing X",
        "judge_model": "claude",
    }
    j = JudgeScore.from_dict(raw)
    assert j.faithfulness == 4 and j.overall == 4
    assert j.error_type == "incomplete_answer"
    assert j.judge_model == "claude"


def test_question_result_with_nested():
    raw = {
        "qid": "q001",
        "question": "Q",
        "ideal_answer": "I",
        "model_answer": "M",
        "answer_type": "mechanism",
        "retrieved": [{"sid": "S1", "source": "a.pdf", "page": 1, "distance": 0.5}],
        "judge": {
            "faithfulness": 5,
            "completeness": 5,
            "citation": 5,
            "overall": 5,
            "error_type": "none",
            "reason": "",
        },
        "human_review": {"score": 4, "notes": "ok"},
    }
    r = QuestionResult.from_dict(raw)
    assert len(r.retrieved) == 1
    assert r.judge.overall == 5
    assert r.human_review.score == 4


def test_eval_run_roundtrip_via_to_dict():
    r = EvalRun(
        run_id="test",
        timestamp="2026-01-01T00:00:00",
        config=RunConfig(rag_model="gpt", top_k=8),
        results=[
            QuestionResult(qid="q001", question="Q", ideal_answer="I", model_answer="M"),
        ],
    )
    d = r.to_dict()
    r2 = EvalRun.from_dict(d)
    assert r2.run_id == "test"
    assert r2.config.top_k == 8
    assert len(r2.results) == 1
    assert r2.results[0].qid == "q001"


def test_human_review_score_can_be_none_or_empty():
    h = HumanReview.from_dict({"score": "", "notes": "abc"})
    assert h.score is None
    assert h.notes == "abc"

    h2 = HumanReview.from_dict({"score": 3})
    assert h2.score == 3
