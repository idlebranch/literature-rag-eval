from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("launcher_safety", str(ROOT / "launcher.pyw"))
spec = importlib.util.spec_from_loader(loader.name, loader)
launcher = importlib.util.module_from_spec(spec)
loader.exec_module(launcher)


def current_health(**overrides):
    data = {
        "service": "literature-rag-api",
        "project_id": launcher.EXPECTED_BUILD["project_id"],
        "application_version": launcher.EXPECTED_BUILD["application_version"],
        "build_id": launcher.EXPECTED_BUILD["build_id"],
        "prompt_version": launcher.EXPECTED_BUILD["prompt_version"],
    }
    data.update(overrides)
    return data


def test_health_identity_rejects_legacy_old_and_foreign_services():
    assert launcher.health_identity(current_health()) == "current"
    assert launcher.health_identity({"service": "literature-rag-api"}) == "legacy_project"
    assert launcher.health_identity(current_health(build_id="old")) == "old_project"
    assert launcher.health_identity({"project_id": "another-project"}) == "foreign"


def test_verified_tree_requires_project_venv_ancestor(monkeypatch):
    project_python = str((launcher.ROOT / ".venv" / "Scripts" / "python.exe").resolve())
    snapshots = {
        300: {
            "ProcessId": 300,
            "ParentProcessId": 200,
            "ExecutablePath": "C:/base/python.exe",
            "CommandLine": "python -m uvicorn api_server:app --port 8010",
        },
        200: {
            "ProcessId": 200,
            "ParentProcessId": 1,
            "ExecutablePath": project_python,
            "CommandLine": f"{project_python} -m uvicorn api_server:app --port 8010",
        },
    }
    monkeypatch.setattr(launcher, "listener_pid", lambda port: 300)
    monkeypatch.setattr(launcher, "process_snapshot", lambda pid: snapshots.get(pid))
    result = launcher.verified_project_tree_root("api", 8010)
    assert result and result[0] == 200


def test_unrelated_8010_process_is_never_verified(monkeypatch):
    monkeypatch.setattr(launcher, "listener_pid", lambda port: 999)
    monkeypatch.setattr(
        launcher,
        "process_snapshot",
        lambda pid: {
            "ProcessId": 999,
            "ParentProcessId": 1,
            "ExecutablePath": "C:/other/python.exe",
            "CommandLine": "python -m http.server 8010",
        },
    )
    assert launcher.verified_project_tree_root("api", 8010) is None


def test_missing_pid_file_can_stop_only_verified_project_tree(tmp_path, monkeypatch):
    pid_file = tmp_path / "api.pid"
    monkeypatch.setitem(launcher.PID_FILES, "api", pid_file)
    monkeypatch.setattr(launcher, "owned_process", lambda kind: (False, None))
    monkeypatch.setattr(launcher, "port_open", lambda port: True)
    monkeypatch.setattr(
        launcher,
        "verified_project_tree_root",
        lambda kind, port: (200, {"CommandLine": "project uvicorn"}),
    )
    monkeypatch.setattr(launcher, "wait_port_released", lambda port: True)
    calls = []
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda args, **kwargs: calls.append(args) or SimpleNamespace(returncode=0),
    )
    assert launcher.stop_service("api") is True
    assert calls == [["taskkill", "/PID", "200", "/T", "/F"]]


def test_stop_refuses_unrelated_process_without_killing(monkeypatch):
    monkeypatch.setattr(launcher, "owned_process", lambda kind: (False, None))
    monkeypatch.setattr(launcher, "port_open", lambda port: True)
    monkeypatch.setattr(launcher, "verified_project_tree_root", lambda kind, port: None)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not kill")),
    )
    with pytest.raises(RuntimeError, match="拒绝停止"):
        launcher.stop_service("api")


def test_stop_fails_if_port_is_not_released(monkeypatch):
    monkeypatch.setattr(launcher, "owned_process", lambda kind: (True, 200))
    monkeypatch.setattr(launcher, "wait_port_released", lambda port: False)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    with pytest.raises(RuntimeError, match="未释放"):
        launcher.stop_service("api")
