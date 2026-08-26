"""Dataclass schema for V3 evaluation runs (JSON canonical form)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


ERROR_TYPES = (
    "none",
    "retrieval_error",
    "generation_error",
    "citation_error",
    "incomplete_answer",
    "hallucination",
    "insufficient_context",
    "script_error",
)


@dataclass
class RetrievedSource:
    sid: str
    source: str
    page: int
    distance: float
    chunk_text: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetrievedSource":
        return cls(
            sid=d.get("sid", ""),
            source=d.get("source", ""),
            page=int(d.get("page", 0) or 0),
            distance=float(d.get("distance", 0.0) or 0.0),
            chunk_text=d.get("chunk_text", "") or "",
        )


@dataclass
class JudgeScore:
    faithfulness: int
    completeness: int
    citation: int
    overall: int
    error_type: str
    reason: str
    correctness: int = 0
    evidence_relevance: int = 0
    judge_model: str = ""
    judged_at: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JudgeScore":
        return cls(
            correctness=int(d.get("correctness", 0) or 0),
            evidence_relevance=int(d.get("evidence_relevance", 0) or 0),
            faithfulness=int(d.get("faithfulness", 0) or 0),
            completeness=int(d.get("completeness", 0) or 0),
            citation=int(d.get("citation", 0) or 0),
            overall=int(d.get("overall", 0) or 0),
            error_type=d.get("error_type", "none") or "none",
            reason=d.get("reason", "") or "",
            judge_model=d.get("judge_model", "") or "",
            judged_at=d.get("judged_at", "") or "",
        )


@dataclass
class HumanReview:
    score: Optional[int] = None
    notes: str = ""
    reviewed_by: str = ""
    reviewed_at: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HumanReview":
        s = d.get("score")
        return cls(
            score=int(s) if s not in (None, "") else None,
            notes=d.get("notes", "") or "",
            reviewed_by=d.get("reviewed_by", "") or "",
            reviewed_at=d.get("reviewed_at", "") or "",
        )


@dataclass
class QuestionResult:
    qid: str
    question: str
    ideal_answer: str
    model_answer: str
    answer_type: str = ""
    retrieved: List[RetrievedSource] = field(default_factory=list)
    judge: Optional[JudgeScore] = None
    human_review: Optional[HumanReview] = None
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QuestionResult":
        return cls(
            qid=d.get("qid", ""),
            question=d.get("question", "") or "",
            ideal_answer=d.get("ideal_answer", "") or "",
            model_answer=d.get("model_answer", "") or "",
            answer_type=d.get("answer_type", "") or "",
            retrieved=[RetrievedSource.from_dict(r) for r in (d.get("retrieved") or [])],
            judge=JudgeScore.from_dict(d["judge"]) if d.get("judge") else None,
            human_review=(
                HumanReview.from_dict(d["human_review"]) if d.get("human_review") else None
            ),
            error=d.get("error"),
        )


@dataclass
class RunConfig:
    rag_model: str = ""
    rag_prompt_version: str = ""
    rag_prompt_hash: str = ""
    answer_mode: str = "quick"
    embedding_model: str = ""
    top_k: int = 5
    judge_model: str = ""
    judge_prompt_version: str = ""
    groundtruth_file: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        return cls(
            rag_model=d.get("rag_model", "") or "",
            rag_prompt_version=d.get("rag_prompt_version", "") or "",
            rag_prompt_hash=d.get("rag_prompt_hash", "") or "",
            answer_mode=d.get("answer_mode", "quick") or "quick",
            embedding_model=d.get("embedding_model", "") or "",
            top_k=int(d.get("top_k", 5) or 5),
            judge_model=d.get("judge_model", "") or "",
            judge_prompt_version=d.get("judge_prompt_version", "") or "",
            groundtruth_file=d.get("groundtruth_file", "") or "",
        )


@dataclass
class RunSummary:
    n_questions: int = 0
    avg_faithfulness: float = 0.0
    avg_correctness: float = 0.0
    avg_evidence_relevance: float = 0.0
    avg_completeness: float = 0.0
    avg_citation: float = 0.0
    avg_overall: float = 0.0
    by_error_type: Dict[str, int] = field(default_factory=dict)
    badcase_count: int = 0
    n_judged: int = 0
    n_errored: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunSummary":
        return cls(
            n_questions=int(d.get("n_questions", 0) or 0),
            avg_faithfulness=float(d.get("avg_faithfulness", 0.0) or 0.0),
            avg_correctness=float(d.get("avg_correctness", 0.0) or 0.0),
            avg_evidence_relevance=float(d.get("avg_evidence_relevance", 0.0) or 0.0),
            avg_completeness=float(d.get("avg_completeness", 0.0) or 0.0),
            avg_citation=float(d.get("avg_citation", 0.0) or 0.0),
            avg_overall=float(d.get("avg_overall", 0.0) or 0.0),
            by_error_type=dict(d.get("by_error_type") or {}),
            badcase_count=int(d.get("badcase_count", 0) or 0),
            n_judged=int(d.get("n_judged", 0) or 0),
            n_errored=int(d.get("n_errored", 0) or 0),
        )


@dataclass
class EvalRun:
    run_id: str
    timestamp: str
    config: RunConfig = field(default_factory=RunConfig)
    results: List[QuestionResult] = field(default_factory=list)
    summary: Optional[RunSummary] = None
    notes: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalRun":
        return cls(
            run_id=d.get("run_id", ""),
            timestamp=d.get("timestamp", "") or "",
            config=RunConfig.from_dict(d.get("config") or {}),
            results=[QuestionResult.from_dict(r) for r in (d.get("results") or [])],
            summary=RunSummary.from_dict(d["summary"]) if d.get("summary") else None,
            notes=d.get("notes", "") or "",
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
