"""Append-only JSONL sink for request-level traces.

The traces file is located relative to THIS source file, not the current
working directory, so ``uvicorn`` / ``api_server`` launched from any directory
always writes to the same ``<project_root>/outputs/traces/traces.jsonl``.

Writes are fail-open: a storage error must never break the /chat request, so
every exception is swallowed (logged to stderr) instead of propagating.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

# store.py -> src/tracing -> src -> <project root>
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Serialize appends so concurrent FastAPI threadpool requests don't interleave
# partial lines in the JSONL file.
_write_lock = threading.Lock()


def default_traces_path() -> Path:
    """Absolute path to the canonical traces file, independent of CWD.

    Recomputed from ``__file__`` on every call so it is provably stable no
    matter where the process was started.
    """
    return _PROJECT_ROOT / "outputs" / "traces" / "traces.jsonl"


TRACES_PATH = default_traces_path()


def append_trace(record: Dict[str, Any], path: Optional[Path] = None) -> None:
    """Append one trace record as a single JSON line.

    Fail-open: never raises. Returns None regardless of success/failure.
    """
    target = Path(path) if path is not None else TRACES_PATH
    try:
        with _write_lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001 — tracing must not break the request
        logger.warning("append_trace failed (%s): %s", type(e).__name__, e)
