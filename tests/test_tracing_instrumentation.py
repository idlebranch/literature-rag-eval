"""Tests for trace instrumentation: recorder fields + one-trace-per-request.

All hermetic — a fake ``answer_fn`` and explicit prompt identity are injected so
no LLM / Chroma / embedding model is imported or called.
"""
import json

import pytest

from src.tracing.instrumentation import TraceRecorder, traced_chat


def _fake_success(question, top_k):
    return {
        "question": question,
        "answer": "PAC 通过吸附与孔隙截留去除抗生素 [S1]。",
        "contexts": [
            {
                "text": "PAC adsorption removes antibiotics ...",
                "metadata": {"source": "pac.pdf", "page": 3, "chunk_index": 2},
                "distance": 0.4123,
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
        "model": "fake-llm",
    }


def _fake_raise(question, top_k):
    raise ValueError("boom")


# ── recorder unit level ────────────────────────────────────────────────

def test_record_success_captures_all_fields():
    rec = TraceRecorder("Q?", 5, prompt_version="v3", prompt_hash="deadbeef1234")
    rec.record_success(_fake_success("Q?", 5))
    d = rec.record.to_dict()

    assert d["status"] == "success"
    assert d["error"] is None
    assert d["question"] == "Q?"
    assert d["top_k"] == 5
    assert d["model"] == "fake-llm"
    assert d["prompt_version"] == "v3"
    assert d["prompt_hash"] == "deadbeef1234"
    assert d["token_usage"]["total_tokens"] == 160
    assert d["latency_ms"] >= 0
    assert len(d["retrieved"]) == 1
    s = d["retrieved"][0]
    assert (s["source"], s["page"], round(s["distance"], 4)) == ("pac.pdf", 3, 0.4123)
    assert s["text"].startswith("PAC adsorption")


def test_record_error_sets_status_and_message():
    rec = TraceRecorder("Q?", 7)
    rec.record_error(ValueError("boom"))
    d = rec.record.to_dict()

    assert d["status"] == "error"
    assert d["model_answer"] == ""
    assert d["retrieved"] == []
    assert d["token_usage"] is None
    assert "ValueError: boom" in d["error"]


def test_token_usage_null_when_missing():
    rec = TraceRecorder("Q?", 5)
    result = _fake_success("Q?", 5)
    result["usage"] = None
    rec.record_success(result)
    assert rec.record.to_dict()["token_usage"] is None


def test_emit_is_idempotent(tmp_path):
    target = tmp_path / "traces.jsonl"
    rec = TraceRecorder("Q?", 5)
    rec.record_success(_fake_success("Q?", 5))
    rec.emit(target)
    rec.emit(target)  # second call must be a no-op
    assert len(target.read_text(encoding="utf-8").splitlines()) == 1


# ── traced_chat orchestration level ────────────────────────────────────

def test_success_writes_exactly_one_trace_and_returns_id(tmp_path):
    target = tmp_path / "traces.jsonl"
    result, trace_id = traced_chat(
        "Q?", 5,
        answer_fn=_fake_success,
        store_path=target,
        prompt_version="v3",
        prompt_hash="abc123abc123",
    )
    assert trace_id
    assert result["answer"].startswith("PAC")

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["trace_id"] == trace_id
    assert row["status"] == "success"
    assert row["top_k"] == 5


def test_error_writes_exactly_one_error_trace_and_reraises(tmp_path):
    target = tmp_path / "traces.jsonl"
    with pytest.raises(ValueError, match="boom"):
        traced_chat(
            "Q?", 5,
            answer_fn=_fake_raise,
            store_path=target,
            prompt_version="v3",
            prompt_hash="abc123abc123",
        )
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["status"] == "error"
    assert "ValueError: boom" in row["error"]


def test_each_request_appends_one_line_with_distinct_ids(tmp_path):
    target = tmp_path / "traces.jsonl"
    _, id1 = traced_chat("A", 5, answer_fn=_fake_success, store_path=target,
                         prompt_version="v3", prompt_hash="h")
    _, id2 = traced_chat("B", 5, answer_fn=_fake_success, store_path=target,
                         prompt_version="v3", prompt_hash="h")
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert id1 != id2
