"""Tests for src.eval.runner pure functions (no LLM calls)."""
import json

import pytest

from src.eval.runner import (
    build_judge_prompt,
    load_groundtruth,
    parse_judge_json,
    prompt_hash,
)
from src.eval.schema import QuestionResult, RetrievedSource


def test_prompt_hash_is_deterministic_and_short():
    h1 = prompt_hash("system prompt")
    h2 = prompt_hash("system prompt")
    h3 = prompt_hash("system prompt v2")
    assert h1 == h2 and h1 != h3
    assert len(h1) == 12


def test_parse_judge_json_strict():
    raw = json.dumps({
        "faithfulness_score": 5,
        "completeness_score": 4,
        "citation_score": 5,
        "overall_score": 4,
        "error_type": "none",
        "judge_reason": "ok",
    })
    j = parse_judge_json(raw)
    assert j["overall_score"] == 4
    assert j["error_type"] == "none"


def test_parse_judge_json_with_surrounding_noise():
    raw = (
        "Sure, here is the JSON: "
        '{"faithfulness_score": 3, "completeness_score": 2, "citation_score": 4, '
        '"overall_score": 3, "error_type": "incomplete_answer", "judge_reason": "x"}\n'
        "(end)"
    )
    j = parse_judge_json(raw)
    assert j["overall_score"] == 3


def test_parse_judge_json_raises_when_no_json():
    with pytest.raises(ValueError):
        parse_judge_json("absolutely no JSON here")


def test_build_judge_prompt_includes_all_fields():
    qr = QuestionResult(
        qid="q1",
        question="Q?",
        ideal_answer="ideal",
        model_answer="model says X",
        answer_type="mechanism",
        retrieved=[RetrievedSource(sid="S1", source="a.pdf", page=1, distance=0.5)],
    )
    prompt = build_judge_prompt(qr, "[S1] a.pdf | page 1 | distance=0.5\nsnippet")
    assert "Q?" in prompt
    assert "ideal" in prompt
    assert "model says X" in prompt
    assert "S1" in prompt
    assert "faithfulness_score" in prompt
    assert "error_type" in prompt


def test_load_groundtruth_jsonl(tmp_path):
    p = tmp_path / "gt.jsonl"
    p.write_text(
        '{"id":"q1","question":"a"}\n'
        '\n'
        '{"id":"q2","question":"b"}\n',
        encoding="utf-8",
    )
    rows = load_groundtruth(p)
    assert len(rows) == 2
    assert rows[0]["id"] == "q1"


def test_load_groundtruth_bad_jsonl_raises(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not valid}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_groundtruth(p)
