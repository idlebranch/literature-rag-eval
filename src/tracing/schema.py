"""Dataclass schema for request-level traces (JSON/JSONL canonical form).

Intentionally standalone — does NOT import ``src.eval`` — so the online trace
spine and the offline eval subsystem stay decoupled for milestone 1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TracedSource:
    """One retrieved chunk as captured in a trace."""

    source: str
    page: int
    distance: float
    text: str


@dataclass
class TraceRecord:
    """A single request-level trace for one POST /chat call."""

    trace_id: str
    timestamp: str
    question: str
    top_k: int
    retrieved: List[TracedSource] = field(default_factory=list)
    model_answer: str = ""
    model: str = ""
    embedding_model: str = ""
    prompt_version: str = ""
    prompt_hash: str = ""
    answer_mode: str = "quick"
    latency_ms: float = 0.0
    token_usage: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    citation_validation: Optional[Dict[str, Any]] = None
    status: str = "success"  # "success" | "error"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
