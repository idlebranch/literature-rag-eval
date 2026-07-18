"""Tests for src.eval.io migration + summary computation."""
import csv
from pathlib import Path

from src.eval.io import (
    compute_summary,
    load_run,
    migrate_from_csv,
    parse_retrieved_sources_field,
    save_run,
)
from src.eval.schema import EvalRun, JudgeScore, QuestionResult, RunConfig


def test_parse_retrieved_sources_field_handles_pipe_separators():
    raw = "[S1] a.pdf | page 2 | distance=0.4153 || [S2] b.pdf | page 1 | distance=0.5"
    out = parse_retrieved_sources_field(raw)
    assert len(out) == 2
    assert out[0].sid == "S1" and out[0].source == "a.pdf" and out[0].page == 2
    assert abs(out[0].distance - 0.4153) < 1e-6
    assert out[1].sid == "S2"


def test_parse_retrieved_sources_field_empty():
    assert parse_retrieved_sources_field("") == []


def test_compute_summary_averages_and_counts():
    run = EvalRun(
        run_id="t",
        timestamp="x",
        config=RunConfig(),
        results=[
            QuestionResult(
                qid="q1",
                question="",
                ideal_answer="",
                model_answer="",
                judge=JudgeScore(5, 5, 5, 5, "none", ""),
            ),
            QuestionResult(
                qid="q2",
                question="",
                ideal_answer="",
                model_answer="",
                judge=JudgeScore(3, 3, 3, 3, "incomplete_answer", ""),
            ),
            QuestionResult(
                qid="q3",
                question="",
                ideal_answer="",
                model_answer="",
                error="[ERROR] api",
            ),
        ],
    )
    s = compute_summary(run, badcase_threshold=4)
    assert s.n_questions == 3
    assert s.n_judged == 2
    assert s.n_errored == 1
    assert s.avg_overall == 4.0
    assert s.by_error_type == {"incomplete_answer": 1, "none": 1}
    assert s.badcase_count == 1  # q2 is overall=3 < 4


def test_migrate_from_csv_joins_eval_and_judge(tmp_path: Path):
    eval_csv = tmp_path / "eval.csv"
    judge_csv = tmp_path / "judge.csv"

    with eval_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id", "answer_type", "question", "ideal_answer",
                "model_answer", "retrieved_sources",
            ],
        )
        w.writeheader()
        w.writerow({
            "id": "q1", "answer_type": "mechanism", "question": "Q",
            "ideal_answer": "I", "model_answer": "M",
            "retrieved_sources": "[S1] x.pdf | page 1 | distance=0.4",
        })
        w.writerow({
            "id": "q2", "answer_type": "comparison", "question": "Q2",
            "ideal_answer": "I2", "model_answer": "[ERROR] bad key",
            "retrieved_sources": "",
        })

    with judge_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id", "question", "faithfulness_score", "completeness_score",
                "citation_score", "overall_score", "error_type", "judge_reason",
            ],
        )
        w.writeheader()
        w.writerow({
            "id": "q1", "question": "Q", "faithfulness_score": "5",
            "completeness_score": "4", "citation_score": "5",
            "overall_score": "4", "error_type": "incomplete_answer",
            "judge_reason": "missed pH",
        })

    run = migrate_from_csv(
        eval_csv=eval_csv,
        judge_csv=judge_csv,
        run_id="test",
        config=RunConfig(rag_model="gpt", top_k=5),
        judge_model="claude",
    )

    assert run.run_id == "test"
    assert len(run.results) == 2
    q1 = run.results[0]
    assert q1.qid == "q1"
    assert q1.judge is not None
    assert q1.judge.overall == 4
    assert q1.judge.judge_model == "claude"
    assert len(q1.retrieved) == 1
    assert q1.retrieved[0].source == "x.pdf"

    q2 = run.results[1]
    assert q2.error is not None and q2.error.startswith("[ERROR]")
    assert q2.judge is None

    assert run.summary is not None
    assert run.summary.n_judged == 1


def test_save_and_load_run_roundtrip(tmp_path: Path):
    run = EvalRun(
        run_id="rt",
        timestamp="x",
        config=RunConfig(rag_model="m", top_k=3),
        results=[
            QuestionResult(qid="q1", question="Q", ideal_answer="I", model_answer="M"),
        ],
    )
    path = tmp_path / "run.json"
    save_run(run, path)
    assert path.exists()

    loaded = load_run(path)
    assert loaded.run_id == "rt"
    assert loaded.config.rag_model == "m"
    assert loaded.results[0].qid == "q1"
