"""Deterministic tests for the GPU preflight and the build watchdog decision logic.

No real GPU / process / network is required: the preflight failure paths and the
pure :func:`watch_build.evaluate` are exercised with injected values.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import preflight_gpu  # noqa: E402
import watch_build  # noqa: E402


# ---------------------------------------------------------------- preflight

def test_preflight_wrong_python_fails(monkeypatch):
    monkeypatch.setattr(sys, "executable", "C:/Windows/py.exe")
    with pytest.raises(SystemExit):
        preflight_gpu._check_python()


def test_preflight_cuda_unavailable_fails(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        preflight_gpu._check_torch()


def test_preflight_wrong_gpu_name_fails(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Tesla T4")
    with pytest.raises(SystemExit):
        preflight_gpu._check_torch()


# ---------------------------------------------------------------- watchdog evaluate

def _hist(lats, stage="dense", t0=0.0):
    return [{"stage": stage, "batch_latency": l, "timestamp": t0 + i * 60} for i, l in enumerate(lats)]


def test_healthy_progress_no_alarm():
    history = _hist([1.0] * 10)
    current = {"stage": "dense", "batch_latency": 1.1, "timestamp": 600,
               "gpu_util": 100, "gpu_memory_used": 4000, "gpu_memory_total": 8000,
               "cpu_percent": 30, "progress_current": 5}
    status, reasons = watch_build.evaluate(history, current)
    assert status == "running"
    assert reasons == []


def test_latency_2x_is_warning():
    history = _hist([1.0] * 10)
    current = {"stage": "dense", "batch_latency": 2.5, "timestamp": 600}
    status, reasons = watch_build.evaluate(history, current)
    assert status == "warning"
    assert any("batch_latency" in r for r in reasons)


def test_latency_5x_three_consecutive_is_error():
    history = _hist([1.0] * 10 + [5.0, 5.0])
    current = {"stage": "dense", "batch_latency": 5.0, "timestamp": 2000}
    status, reasons = watch_build.evaluate(history, current)
    assert status == "error"
    assert any("5x" in r for r in reasons)


def test_latency_single_5x_spike_is_only_warning():
    history = _hist([1.0] * 10)
    current = {"stage": "dense", "batch_latency": 6.0, "timestamp": 2000}
    status, _ = watch_build.evaluate(history, current)
    assert status == "warning"  # not error (needs 3 consecutive)


def test_progress_stall_is_error():
    now = 10_000.0
    history = [{"timestamp": now - 60 * i, "progress_current": 10, "stage": "dense"}
               for i in range(1, 11)]
    current = {"timestamp": now, "progress_current": 10, "stage": "dense"}
    status, reasons = watch_build.evaluate(history, current)
    assert status == "error"
    assert any("stall" in r for r in reasons)


def test_stage_aware_baseline_isolated_per_stage():
    # dense baseline ~1s, sparse baseline ~10s; a 3s sparse batch is NOT a 3x alarm
    history = _hist([1.0] * 10, stage="dense") + _hist([10.0] * 10, stage="sparse", t0=1000)
    current = {"stage": "sparse", "batch_latency": 15.0, "timestamp": 2000}
    status, _ = watch_build.evaluate(history, current)
    assert status == "running"  # 1.5x sparse baseline, not vs dense baseline


def test_malformed_progress_file_no_crash(tmp_path):
    bad = tmp_path / "progress.json"
    bad.write_text("{not json", encoding="utf-8")
    assert watch_build.read_progress(str(bad)) is None


def test_completion_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(watch_build, "ROOT", tmp_path)
    (tmp_path / "sparse_index_x").mkdir(parents=True)
    (tmp_path / "sparse_index_x" / "manifest.json").write_text("{}", encoding="utf-8")
    assert watch_build.completion_ok("x") is True
    assert watch_build.completion_ok("y") is False


def test_infer_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(watch_build, "ROOT", tmp_path)
    assert watch_build.infer_stage("x") == "dense"
    (tmp_path / "chroma_db_x").mkdir()
    (tmp_path / "chroma_db_x" / "chroma.sqlite3").write_text("", encoding="utf-8")
    assert watch_build.infer_stage("x") == "sparse"
    (tmp_path / "sparse_index_x").mkdir()
    (tmp_path / "sparse_index_x" / "manifest.json").write_text("{}", encoding="utf-8")
    assert watch_build.infer_stage("x") == "completed"


def test_evaluate_no_latency_history_no_alarm():
    # no batch_latency anywhere -> no baseline -> no latency alarms
    history = [{"stage": "dense", "timestamp": float(i)} for i in range(5)]
    current = {"stage": "dense", "batch_latency": 99.0, "timestamp": 999.0}
    status, _ = watch_build.evaluate(history, current)
    assert status == "running"
