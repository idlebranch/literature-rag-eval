"""Cheap, redacted runtime status for the API and Windows launcher."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import importlib.util
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

from src.config import settings
from src.build_info import BUILD_INFO


# Warmup changes loading → ready quickly; counts still come from one-time warmup,
# so a short status cache does not rescan PDFs or recount Chroma per request.
_CACHE_SECONDS = 1.0
_cache_lock = threading.Lock()
_cached_at = 0.0
_cached_status: dict[str, Any] | None = None
_STARTED_AT = datetime.now().astimezone().isoformat(timespec="seconds")


def _iso_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def _build_status() -> dict[str, Any]:
    pdf_dir = Path(settings.pdf_dir)
    chroma_dir = Path(settings.chroma_dir)
    sqlite_file = chroma_dir / "chroma.sqlite3"
    from src.warmup import get_warmup_state

    warmup = get_warmup_state()
    warmup_matches = (
        warmup.get("pdf_dir") == settings.pdf_dir
        and warmup.get("chroma_dir") == settings.chroma_dir
    )
    warmup_started = warmup_matches and warmup.get("started_at") is not None

    if warmup_started:
        # These counts are computed once by startup warmup, not on every health poll.
        pdf_count = int(warmup.get("document_count") or 0)
        chunk_count = warmup.get("chunk_count")
        chroma_warmup = warmup.get("chroma") or {}
        index_status = str(chroma_warmup.get("status", "loading"))
        index_error = chroma_warmup.get("error_type")
    else:
        # Offline/test fallback. The public API starts warmup before normal polling.
        pdf_count = len(list(pdf_dir.rglob("*.pdf"))) if pdf_dir.is_dir() else 0
        index_status = "missing"
        chunk_count: int | None = None
        index_error: str | None = None
        if sqlite_file.is_file() and sqlite_file.stat().st_size > 0:
            try:
                from src.vectorstore import get_collection

                chunk_count = get_collection().count()
                index_status = "ready" if chunk_count > 0 else "empty"
            except Exception as exc:  # status remains available if Chroma is damaged
                index_status = "failed"
                index_error = type(exc).__name__

    llm_configured = bool(settings.openai_api_key and settings.llm_model)
    embedding_package = importlib.util.find_spec("sentence_transformers") is not None
    evaluation_available = importlib.util.find_spec("src.eval.runner") is not None
    knowledge_ready = pdf_count > 0
    vector_ready = index_status == "ready"
    prewarmed = bool(warmup.get("prewarmed")) if warmup_matches else False
    rag_ready = (
        knowledge_ready
        and vector_ready
        and llm_configured
        and embedding_package
        and prewarmed
    )

    knowledge_base = {
        "status": "ready" if knowledge_ready else "missing",
        "document_count": pdf_count,
        "path": "data/pdfs",
    }
    vector_index = {
        "status": index_status,
        "type": "ChromaDB",
        "collection": settings.collection_name,
        "chunk_count": chunk_count,
        "path": "chroma_db",
        "last_updated": _iso_mtime(sqlite_file),
        "error_type": index_error,
    }
    llm_runtime: dict[str, Any] = {
        "network_checked": False,
        "last_success_at": None,
        "last_error_type": None,
    }
    llm_module = sys.modules.get("src.llm_client")
    if llm_module is not None and hasattr(llm_module, "get_llm_runtime_state"):
        llm_runtime.update(llm_module.get_llm_runtime_state())
    llm_ready = bool(llm_runtime.get("last_success_at"))
    llm = {
        "status": "ready" if llm_ready else ("configured" if llm_configured else "not_configured"),
        "configured": llm_configured,
        "model": settings.llm_model,
        **llm_runtime,
    }
    embedding_loaded = False
    embedder_module = sys.modules.get("src.embedder")
    if embedder_module is not None and hasattr(embedder_module, "get_embedding_model"):
        embedding_loaded = embedder_module.get_embedding_model.cache_info().currsize > 0
    embedding_warmup = warmup.get("embedding") or {}
    if warmup_started:
        embedding_status = str(embedding_warmup.get("status", "loading"))
        embedding_error = embedding_warmup.get("error_type")
    else:
        embedding_status = "ready" if embedding_loaded else (
            "configured" if embedding_package else "package_missing"
        )
        embedding_error = None
    embedding = {
        "status": embedding_status,
        "configured": bool(settings.embedding_model),
        "package_available": embedding_package,
        "model": settings.embedding_model,
        "loaded": embedding_loaded,
        "error_type": embedding_error,
    }

    return {
        "status": "ok" if rag_ready else "degraded",
        "service": "literature-rag-api",
        "version": BUILD_INFO["application_version"],
        "project_id": BUILD_INFO["project_id"],
        "application_version": BUILD_INFO["application_version"],
        "build_id": BUILD_INFO["build_id"],
        "prompt_version": BUILD_INFO["prompt_version"],
        "started_at": _STARTED_AT,
        "process_id": os.getpid(),
        "top_k_default": settings.top_k,
        "api_ready": True,
        "rag_ready": rag_ready,
        "prewarmed": prewarmed,
        "warmup": {
            "started_at": warmup.get("started_at") if warmup_matches else None,
            "completed_at": warmup.get("completed_at") if warmup_matches else None,
            "prewarmed": prewarmed,
        },
        "knowledge_base": knowledge_base,
        "vector_index": vector_index,
        "llm": llm,
        "embedding": embedding,
        "tracing": {
            "enabled": settings.tracing_enabled,
            "status": "enabled" if settings.tracing_enabled else "disabled",
        },
        "evaluation": {
            "available": evaluation_available,
            "status": "available" if evaluation_available else "unavailable",
        },
    }


def get_runtime_status(*, force: bool = False) -> dict[str, Any]:
    """Return a cached status snapshot without loading the embedding or LLM model."""
    global _cached_at, _cached_status
    now = time.monotonic()
    with _cache_lock:
        if force or _cached_status is None or now - _cached_at >= _CACHE_SECONDS:
            _cached_status = _build_status()
            _cached_at = now
        return deepcopy(_cached_status)
