"""Tests for the trace store: JSONL append, fail-open, and CWD-independent path."""
import json
import os
from pathlib import Path

from src.tracing import store


def test_append_creates_dir_and_appends_lines(tmp_path):
    target = tmp_path / "sub" / "traces.jsonl"
    store.append_trace({"trace_id": "a", "status": "success"}, path=target)
    store.append_trace({"trace_id": "b", "status": "error"}, path=target)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["trace_id"] == "a"
    assert json.loads(lines[1])["trace_id"] == "b"


def test_append_is_failopen(tmp_path):
    # Point at a directory instead of a file → open('a') fails; must NOT raise.
    bad_path = tmp_path  # a directory
    result = store.append_trace({"trace_id": "x"}, path=bad_path)
    assert result is None  # swallowed, no exception


def test_append_unicode_preserved(tmp_path):
    target = tmp_path / "traces.jsonl"
    store.append_trace({"question": "臭氧氧化机理？"}, path=target)
    line = target.read_text(encoding="utf-8").strip()
    assert "臭氧氧化机理？" in line  # ensure_ascii=False


def test_traces_path_is_absolute_and_cwd_independent(tmp_path, monkeypatch):
    """The canonical path must resolve to the project root regardless of CWD."""
    before = store.default_traces_path()
    assert before.is_absolute()
    assert before.parts[-3:] == ("outputs", "traces", "traces.jsonl")

    # The resolved root must actually be the project root (contains this package).
    project_root = before.parents[2]
    assert (project_root / "src" / "tracing" / "store.py").exists()

    # Starting from a totally different working directory must not change it.
    monkeypatch.chdir(tmp_path)
    assert Path(os.getcwd()).resolve() == tmp_path.resolve()  # sanity: cwd moved
    after = store.default_traces_path()

    assert after == before
    assert after.is_absolute()
    # Module-level constant is likewise CWD-independent (computed from __file__).
    assert store.TRACES_PATH == before
