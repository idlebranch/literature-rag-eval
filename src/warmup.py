"""One-time local resource warmup; never sends a paid LLM request."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import threading
from typing import Any

from src.config import settings


_lock = threading.RLock()
_thread: threading.Thread | None = None
_state: dict[str, Any] = {
    "started_at": None,
    "completed_at": None,
    "prewarmed": False,
    "pdf_dir": settings.pdf_dir,
    "chroma_dir": settings.chroma_dir,
    "document_count": None,
    "chunk_count": None,
    "embedding": {"status": "not_started", "error_type": None},
    "chroma": {"status": "not_started", "error_type": None},
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def get_warmup_state() -> dict[str, Any]:
    with _lock:
        return deepcopy(_state)


def warmup_sync() -> dict[str, Any]:
    """Load BGE/Chroma and execute minimal local embedding/vector queries."""
    with _lock:
        if _state["prewarmed"]:
            return deepcopy(_state)
        _state.update(
            {
                "started_at": _now(),
                "completed_at": None,
                "prewarmed": False,
                "pdf_dir": settings.pdf_dir,
                "chroma_dir": settings.chroma_dir,
                "document_count": None,
                "chunk_count": None,
            }
        )
        _state["embedding"] = {"status": "loading", "error_type": None}
        _state["chroma"] = {"status": "loading", "error_type": None}

    pdf_dir = Path(settings.pdf_dir)
    document_count = len(list(pdf_dir.rglob("*.pdf"))) if pdf_dir.is_dir() else 0
    with _lock:
        _state["document_count"] = document_count

    vector: list[float] | None = None
    try:
        from src.embedder import embed_query

        vector = embed_query("literature retrieval warmup")
    except Exception as exc:
        with _lock:
            _state["embedding"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
            }
    else:
        with _lock:
            _state["embedding"] = {"status": "ready", "error_type": None}

    try:
        from src.vectorstore import get_collection

        collection = get_collection()
        chunk_count = collection.count()
        if vector is not None and chunk_count > 0:
            collection.query(
                query_embeddings=[vector],
                n_results=1,
                include=["distances"],
            )
    except Exception as exc:
        with _lock:
            _state["chroma"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
            }
    else:
        with _lock:
            _state["chunk_count"] = chunk_count
            _state["chroma"] = {
                "status": "ready" if chunk_count > 0 else "failed",
                "error_type": None if chunk_count > 0 else "EmptyCollection",
            }

    # Build the pooled HTTP client without making any network/model request.
    if settings.openai_api_key:
        try:
            from src.llm_client import get_llm_client

            get_llm_client()
        except Exception:
            pass

    with _lock:
        _state["prewarmed"] = (
            _state["embedding"]["status"] == "ready"
            and _state["chroma"]["status"] == "ready"
        )
        _state["completed_at"] = _now()
        return deepcopy(_state)


def start_warmup() -> threading.Thread:
    global _thread
    with _lock:
        if _thread is None or not _thread.is_alive() and not _state["prewarmed"]:
            _thread = threading.Thread(
                target=warmup_sync,
                name="literature-rag-warmup",
                daemon=True,
            )
            _thread.start()
        return _thread


def reset_warmup_state_for_tests() -> None:
    global _thread
    with _lock:
        _thread = None
        _state.update(
            {
                "started_at": None,
                "completed_at": None,
                "prewarmed": False,
                "pdf_dir": settings.pdf_dir,
                "chroma_dir": settings.chroma_dir,
                "document_count": None,
                "chunk_count": None,
                "embedding": {"status": "not_started", "error_type": None},
                "chroma": {"status": "not_started", "error_type": None},
            }
        )
