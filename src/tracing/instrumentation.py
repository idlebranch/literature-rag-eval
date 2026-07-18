"""Trace lifecycle orchestration for POST /chat (milestone 1).

``traced_chat`` is the single instrumentation seam: it creates one trace_id,
times the call, runs the RAG function, and emits exactly one trace record
(success or error) via the store. It is the ONLY place that calls the store in
the request path — rag_chain / llm_client stay trace-agnostic.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from src.config import settings
from src.tracing.schema import TraceRecord, TracedSource
from src.tracing.store import append_trace


class TraceRecorder:
    """Builds and emits one TraceRecord for a single request.

    Emission is idempotent (``_emitted`` guard) so a record is written at most
    once; the caller's ``finally`` block guarantees it is written at least once.
    """

    def __init__(
        self,
        question: str,
        top_k: int,
        *,
        prompt_version: str = "",
        prompt_hash: str = "",
    ) -> None:
        self.trace_id = uuid.uuid4().hex
        self._t0 = time.perf_counter()
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.question = question
        self.top_k = top_k
        self.prompt_version = prompt_version
        self.prompt_hash = prompt_hash
        self.record: Optional[TraceRecord] = None
        self._emitted = False

    def _latency_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 2)

    def record_success(self, result: Dict[str, Any]) -> None:
        """Build a success record from answer_question's result dict.

        Defensive (`.get` everywhere, no strict casts that can raise) so a
        successful answer is never turned into a 500 by trace assembly.
        """
        retrieved = []
        for hit in result.get("contexts") or []:
            if not isinstance(hit, dict):
                continue
            meta = hit.get("metadata") or {}
            try:
                page = int(meta.get("page", 0) or 0)
            except (TypeError, ValueError):
                page = 0
            try:
                distance = float(hit.get("distance", 0.0) or 0.0)
            except (TypeError, ValueError):
                distance = 0.0
            retrieved.append(
                TracedSource(
                    source=str(meta.get("source", "")),
                    page=page,
                    distance=distance,
                    text=str(hit.get("text", "") or ""),
                )
            )
        self.record = TraceRecord(
            trace_id=self.trace_id,
            timestamp=self.timestamp,
            question=self.question,
            top_k=self.top_k,
            retrieved=retrieved,
            model_answer=str(result.get("answer", "") or ""),
            model=str(result.get("model", "") or settings.llm_model),
            embedding_model=settings.embedding_model,
            prompt_version=self.prompt_version,
            prompt_hash=self.prompt_hash,
            latency_ms=self._latency_ms(),
            token_usage=result.get("usage"),
            status="success",
            error=None,
        )

    def record_error(self, exc: BaseException) -> None:
        self.record = TraceRecord(
            trace_id=self.trace_id,
            timestamp=self.timestamp,
            question=self.question,
            top_k=self.top_k,
            retrieved=[],
            model_answer="",
            model=settings.llm_model,
            embedding_model=settings.embedding_model,
            prompt_version=self.prompt_version,
            prompt_hash=self.prompt_hash,
            latency_ms=self._latency_ms(),
            token_usage=None,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )

    def emit(self, path: Optional[Path] = None) -> None:
        if self._emitted:
            return
        self._emitted = True
        if self.record is None:
            # Guarantee a row even if neither record_* was called.
            self.record_error(RuntimeError("trace recorder produced no record"))
        append_trace(self.record.to_dict(), path=path)


def traced_chat(
    question: str,
    top_k: int,
    *,
    answer_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    store_path: Optional[Path] = None,
    prompt_version: Optional[str] = None,
    prompt_hash: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """Run one RAG call under tracing and return ``(result, trace_id)``.

    On error, an ``status=error`` trace is still emitted and the exception is
    re-raised (preserving the existing HTTP 500 behavior of /chat).

    ``answer_fn`` / ``prompt_version`` / ``prompt_hash`` are injectable so unit
    tests can exercise this without importing the heavy RAG stack. The real
    values are imported lazily from ``src.rag_chain`` only when not provided.
    """
    if answer_fn is None or prompt_version is None or prompt_hash is None:
        from src import rag_chain

        if answer_fn is None:
            answer_fn = rag_chain.answer_question
        if prompt_version is None:
            prompt_version = rag_chain.PROMPT_VERSION
        if prompt_hash is None:
            prompt_hash = rag_chain.PROMPT_HASH

    recorder = TraceRecorder(
        question,
        top_k,
        prompt_version=prompt_version,
        prompt_hash=prompt_hash,
    )
    try:
        result = answer_fn(question, top_k=top_k)
        recorder.record_success(result)
        return result, recorder.trace_id
    except Exception as e:
        recorder.record_error(e)
        raise
    finally:
        recorder.emit(store_path)
