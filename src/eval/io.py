"""JSON load/save + legacy CSV migration for V3 evaluation runs."""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional, Tuple

from src.eval.schema import (
    EvalRun,
    JudgeScore,
    QuestionResult,
    RetrievedSource,
    RunConfig,
    RunSummary,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def save_run(run: EvalRun, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = run.to_dict()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_run(path: Path) -> EvalRun:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return EvalRun.from_dict(raw)


_SOURCE_RE = re.compile(
    r"\[S(\d+)\]\s*(.+?)\s*\|\s*page\s*(\d+)\s*\|\s*distance=([0-9.]+)"
)


def parse_retrieved_sources_field(field_value: str) -> List[RetrievedSource]:
    """Parse the legacy `[S1] file.pdf | page 2 | distance=0.4153 || [S2] ...` string."""
    if not field_value:
        return []
    out: List[RetrievedSource] = []
    for segment in field_value.split("||"):
        segment = segment.strip()
        m = _SOURCE_RE.search(segment)
        if not m:
            continue
        out.append(
            RetrievedSource(
                sid=f"S{m.group(1)}",
                source=m.group(2).strip(),
                page=int(m.group(3)),
                distance=float(m.group(4)),
            )
        )
    return out


def compute_summary(run: EvalRun, badcase_threshold: int = 4) -> RunSummary:
    judged = [r.judge for r in run.results if r.judge is not None]
    f = [j.faithfulness for j in judged]
    co = [j.correctness for j in judged if j.correctness > 0]
    er = [j.evidence_relevance for j in judged if j.evidence_relevance > 0]
    c = [j.completeness for j in judged]
    ci = [j.citation for j in judged]
    o = [j.overall for j in judged]
    by_type: dict[str, int] = {}
    for j in judged:
        by_type[j.error_type] = by_type.get(j.error_type, 0) + 1
    badcase_count = sum(1 for j in judged if j.overall < badcase_threshold)

    return RunSummary(
        n_questions=len(run.results),
        avg_faithfulness=round(mean(f), 2) if f else 0.0,
        avg_correctness=round(mean(co), 2) if co else 0.0,
        avg_evidence_relevance=round(mean(er), 2) if er else 0.0,
        avg_completeness=round(mean(c), 2) if c else 0.0,
        avg_citation=round(mean(ci), 2) if ci else 0.0,
        avg_overall=round(mean(o), 2) if o else 0.0,
        by_error_type=dict(sorted(by_type.items())),
        badcase_count=badcase_count,
        n_judged=len(judged),
        n_errored=sum(1 for r in run.results if r.error),
    )


def migrate_from_csv(
    eval_csv: Path,
    judge_csv: Optional[Path],
    run_id: str,
    config: Optional[RunConfig] = None,
    judge_model: str = "",
    timestamp: Optional[str] = None,
) -> EvalRun:
    """Build an EvalRun by joining the legacy answer_eval CSV with a judge summary CSV.

    eval_csv has columns: id, answer_type, question, ideal_answer, model_answer,
                          retrieved_sources, ...
    judge_csv has columns: id, faithfulness_score, completeness_score, citation_score,
                           overall_score, error_type, judge_reason
    """
    if config is None:
        config = RunConfig()
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    judge_index: dict[str, JudgeScore] = {}
    if judge_csv and judge_csv.exists():
        with judge_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                qid = row.get("id", "").strip()
                if not qid:
                    continue
                judge_index[qid] = JudgeScore(
                    correctness=int(row.get("correctness_score") or 0),
                    evidence_relevance=int(row.get("evidence_relevance_score") or 0),
                    faithfulness=int(row.get("faithfulness_score") or 0),
                    completeness=int(row.get("completeness_score") or 0),
                    citation=int(row.get("citation_score") or 0),
                    overall=int(row.get("overall_score") or 0),
                    error_type=row.get("error_type") or "none",
                    reason=row.get("judge_reason") or "",
                    judge_model=judge_model,
                    judged_at=timestamp,
                )

    results: List[QuestionResult] = []
    with eval_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qid = row.get("id", "").strip()
            model_answer = (row.get("model_answer") or "").strip()
            err = None
            if model_answer.startswith("[ERROR]"):
                err = model_answer
            results.append(
                QuestionResult(
                    qid=qid,
                    question=(row.get("question") or "").strip(),
                    ideal_answer=(row.get("ideal_answer") or "").strip(),
                    model_answer=model_answer,
                    answer_type=(row.get("answer_type") or "").strip(),
                    retrieved=parse_retrieved_sources_field(
                        row.get("retrieved_sources") or ""
                    ),
                    judge=judge_index.get(qid),
                    error=err,
                )
            )

    run = EvalRun(run_id=run_id, timestamp=timestamp, config=config, results=results)
    run.summary = compute_summary(run)
    return run


def iter_runs(runs_dir: Path) -> Iterable[Tuple[Path, EvalRun]]:
    if not runs_dir.exists():
        return
    for p in sorted(runs_dir.glob("*.json")):
        try:
            yield p, load_run(p)
        except Exception as e:
            logger.warning("Failed to load %s: %s", p, e)
