"""Lightweight build watchdog (no LLM / no API calls).

Monitors a running build PID every ``--interval`` seconds and appends one JSON
sample per tick to data/processed/build_watchdog.jsonl, while keeping a rolling
summary in data/processed/build_status.json.

Usage:
    .venv/Scripts/python.exe scripts/watch_build.py --pid 12345 --index section_aware_270_gpu

The core decision logic is the pure function :func:`evaluate`, which is unit
tested independently of any real process / GPU.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
DEFAULT_LOG = DATA / "build_watchdog.jsonl"
DEFAULT_STATUS = DATA / "build_status.json"
DEFAULT_PROGRESS = DATA / "build_progress.json"


# ---------------------------------------------------------------- pure logic

def evaluate(history: list[dict], current: dict) -> tuple[str, list[str]]:
    """Decide (status, reasons) for one sample.

    status in {"running", "warning", "error"}. ``history`` is the list of prior
    samples (oldest first, excluding ``current``). Pure: no I/O.
    """
    errors: list[str] = []
    warnings: list[str] = []

    stage = current.get("stage")
    latency = current.get("batch_latency")
    now = current.get("timestamp")
    gpu_util = current.get("gpu_util")
    gpu_mem_used = current.get("gpu_memory_used")
    gpu_mem_total = current.get("gpu_memory_total")
    cpu = current.get("cpu_percent")
    progress = current.get("progress_current")

    # --- stage-aware latency baseline (median of first 10 stable batches) ---
    same_stage = [s for s in history if s.get("stage") == stage
                  and s.get("batch_latency") is not None]
    if latency is not None and same_stage:
        baseline_lats = [s["batch_latency"] for s in same_stage[:10]]
        base = statistics.median(baseline_lats)
        if base and base > 0:
            ratio = latency / base
            if ratio >= 2:
                warnings.append(
                    f"batch_latency {ratio:.1f}x baseline ({latency:.1f}s vs {base:.1f}s)")
            if ratio >= 5:
                recent = [s["batch_latency"] for s in same_stage[-2:]] + [latency]
                if len(recent) >= 3 and all(r / base >= 5 for r in recent):
                    errors.append("batch_latency >=5x baseline for 3 consecutive batches")

    # --- progress stall: no growth for >= 10 minutes ---
    if progress is not None and now is not None:
        cutoff = now - 600
        recent = [s for s in history
                  if s.get("timestamp", 0) >= cutoff and s.get("progress_current") is not None]
        if recent and all(s["progress_current"] == progress for s in recent):
            errors.append("progress stalled (no growth for >=10 min)")

    # --- gpu util < 20% for 3 consecutive samples ---
    if gpu_util is not None and gpu_util < 20:
        prior = [s for s in history[-2:] if s.get("gpu_util") is not None and s["gpu_util"] < 20]
        if len(prior) == 2:
            warnings.append("gpu_util <20% for 3 min")

    # --- gpu memory > 95% ---
    if gpu_mem_used is not None and gpu_mem_total and gpu_mem_used / gpu_mem_total > 0.95:
        warnings.append(f"gpu_memory >95% ({gpu_mem_used}/{gpu_mem_total} MiB)")

    # --- cpu > 95% for 5 consecutive samples ---
    if cpu is not None and cpu > 95:
        prior = [s for s in history[-4:] if s.get("cpu_percent") is not None and s["cpu_percent"] > 95]
        if len(prior) == 4:
            warnings.append("cpu >95% for 5 min")

    if errors:
        return "error", errors
    if warnings:
        return "warning", warnings
    return "running", []


# ---------------------------------------------------------------- collection

def _nvidia_smi() -> tuple[int | None, int | None, int | None]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except Exception:  # noqa: BLE001
        return None, None, None
    if out.returncode != 0 or not out.stdout.strip():
        return None, None, None
    parts = [p.strip() for p in out.stdout.strip().split(",")]
    if len(parts) < 3:
        return None, None, None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None, None, None


def _proc(pid: int):
    try:
        import psutil
        p = psutil.Process(pid)
        return p, p.create_time()
    except Exception:  # noqa: BLE001
        return None, None


def _proc_exe(pid: int) -> str | None:
    try:
        import psutil
        return psutil.Process(pid).exe()
    except Exception:  # noqa: BLE001
        return None


def _proc_interpreter(pid: int) -> str | None:
    """Best-effort interpreter path: the invoked argv[0] (may be a .venv shim)."""
    try:
        import psutil
        cmd = psutil.Process(pid).cmdline()
        if cmd:
            return cmd[0]
    except Exception:  # noqa: BLE001
        pass
    return _proc_exe(pid)


def read_progress(progress_file: str) -> dict | None:
    try:
        return json.loads(Path(progress_file).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def infer_stage(index: str | None) -> str:
    if not index:
        return "unknown"
    if (ROOT / f"sparse_index_{index}" / "manifest.json").exists():
        return "completed"
    if (ROOT / f"chroma_db_{index}" / "chroma.sqlite3").exists():
        return "sparse"
    return "dense"


def completion_ok(index: str | None) -> bool:
    return bool(index) and (ROOT / f"sparse_index_{index}" / "manifest.json").exists()


# ---------------------------------------------------------------- main loop

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--index", default=None, help="index name (for stage/completion inference)")
    ap.add_argument("--progress-file", default=str(DEFAULT_PROGRESS))
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--status", default=str(DEFAULT_STATUS))
    ap.add_argument("--kill-on-error", action="store_true")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    status_path = Path(args.status)

    # wrong interpreter check (hard fail before starting): allow the .venv shim
    # or the uv base interpreter it resolves to, but reject the system Python311.
    interp = _proc_interpreter(args.pid)
    if interp:
        p = str(interp).replace("\\", "/").lower()
        if ".venv" not in p and "python311" in p:
            msg = f"build PID {args.pid} runs non-.venv python: {interp}"
            print(f"ERROR: {msg}")
            _write_diagnostic(status_path, "error", [msg])
            sys.exit(2)

    history: list[dict] = []
    started = time.time()
    print(f"watchdog attached to PID {args.pid} (interval={args.interval}s)")

    while True:
        p, create_time = _proc(args.pid)
        alive = p is not None and p.is_running()
        now = time.time()

        if not alive:
            if completion_ok(args.index):
                sample = {"timestamp": now, "pid": args.pid, "status": "completed",
                          "stage": "completed"}
            else:
                sample = {"timestamp": now, "pid": args.pid, "status": "error",
                          "stage": infer_stage(args.index)}
            _append(log_path, sample)
            _write_status(status_path, sample["status"], sample.get("stage"), [])
            print(f"build PID {args.pid} gone -> {sample['status']}")
            sys.exit(0 if sample["status"] == "completed" else 1)

        gpu_util, gpu_mem_used, gpu_mem_total = _nvidia_smi()
        cpu_percent = p.cpu_percent(interval=None) if p else None
        elapsed = now - (create_time or now)

        progress = read_progress(args.progress_file) or {}
        stage = progress.get("stage") or infer_stage(args.index)

        current = {
            "timestamp": now,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "pid": args.pid,
            "stage": stage,
            "elapsed_s": round(elapsed, 1),
            "progress_current": progress.get("progress_current"),
            "progress_total": progress.get("progress_total"),
            "batch_latency": progress.get("batch_latency"),
            "cpu_percent": cpu_percent,
            "gpu_util": gpu_util,
            "gpu_memory_used": gpu_mem_used,
            "gpu_memory_total": gpu_mem_total,
        }
        status, reasons = evaluate(history, current)
        current["status"] = status
        if reasons:
            current["reasons"] = reasons

        _append(log_path, current)
        _write_status(status_path, status, stage, reasons)

        if status == "error":
            print(f"ERROR: {reasons}")
            if args.kill_on_error and alive:
                try:
                    p.kill()
                    print(f"terminated build PID {args.pid} (hard fail)")
                except Exception as e:  # noqa: BLE001
                    print(f"failed to terminate PID {args.pid}: {e}")
            sys.exit(1)
        if status == "warning":
            print(f"WARNING: {reasons}")

        history.append(current)
        time.sleep(args.interval)


def _append(log_path: Path, sample: dict) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _write_status(status_path: Path, status: str, stage, reasons: list[str]) -> None:
    status_path.write_text(json.dumps(
        {"status": status, "stage": stage, "reasons": reasons,
         "updated_at": datetime.now(timezone.utc).isoformat()},
        ensure_ascii=False, indent=2), encoding="utf-8")


def _write_diagnostic(status_path: Path, status: str, reasons: list[str]) -> None:
    diag = status_path.with_name("build_diagnostic.json")
    diag.write_text(json.dumps(
        {"status": status, "reasons": reasons,
         "at": datetime.now(timezone.utc).isoformat()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    _write_status(status_path, status, "unknown", reasons)


if __name__ == "__main__":
    main()
