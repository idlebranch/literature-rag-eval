"""Export an EvalRun to human-readable Markdown and a flat summary CSV."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional

from src.eval.io import compute_summary
from src.eval.schema import EvalRun, JudgeScore


def _score_bar(score: int, total: int = 5) -> str:
    if score < 1:
        return "—"
    filled = "█" * score
    empty = "░" * (total - score)
    return f"{filled}{empty} {score}"


def _slug(qid: str) -> str:
    return qid.lower().replace(" ", "-")


def to_markdown(run: EvalRun, path: Path, badcase_threshold: int = 4) -> Path:
    """Write a TOC-equipped, navigable Markdown report."""
    if run.summary is None:
        run.summary = compute_summary(run, badcase_threshold=badcase_threshold)

    summary = run.summary
    lines: List[str] = []
    lines.append(f"# RAG Evaluation Report · {run.run_id}")
    lines.append("")
    lines.append(f"- **Generated**: {run.timestamp}")
    lines.append(f"- **RAG model**: `{run.config.rag_model or '-'}`")
    lines.append(f"- **Prompt version**: `{run.config.rag_prompt_version or '-'}`")
    lines.append(f"- **Answer mode**: `{run.config.answer_mode or '-'}`")
    lines.append(f"- **Embedding**: `{run.config.embedding_model or '-'}`")
    lines.append(f"- **Top-K**: {run.config.top_k}")
    lines.append(f"- **Judge model**: `{run.config.judge_model or '-'}`")
    lines.append(f"- **Judge prompt**: `{run.config.judge_prompt_version or '-'}`")
    lines.append(f"- **Groundtruth**: `{run.config.groundtruth_file or '-'}`")
    if run.notes:
        lines.append(f"- **Notes**: {run.notes}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| 维度 | 平均分 (1-5) |")
    lines.append("| --- | --- |")
    lines.append(f"| correctness | {summary.avg_correctness} |")
    lines.append(f"| evidence relevance | {summary.avg_evidence_relevance} |")
    lines.append(f"| faithfulness | {summary.avg_faithfulness} |")
    lines.append(f"| completeness | {summary.avg_completeness} |")
    lines.append(f"| citation | {summary.avg_citation} |")
    lines.append(f"| **overall** | **{summary.avg_overall}** |")
    lines.append("")
    lines.append(
        f"- 样本数：{summary.n_questions}，已评测：{summary.n_judged}，运行错误：{summary.n_errored}"
    )
    lines.append(f"- Badcase（overall < {badcase_threshold}）：{summary.badcase_count}")
    lines.append("")

    lines.append("### error_type 分布")
    lines.append("")
    lines.append("| error_type | 计数 |")
    lines.append("| --- | --- |")
    for k, v in summary.by_error_type.items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")

    lines.append("## 目录")
    lines.append("")
    for r in run.results:
        overall = r.judge.overall if r.judge else None
        flag = ""
        if overall is not None and overall < badcase_threshold:
            flag = " 🔴"
        elif overall == 5:
            flag = " ⭐"
        score_str = f"overall={overall}" if overall is not None else "未评测"
        lines.append(f"- [{r.qid} · {r.answer_type} · {score_str}{flag}](#{_slug(r.qid)}) — {r.question}")
    lines.append("")

    lines.append("---")
    lines.append("")

    for r in run.results:
        lines.append(f"## {r.qid}")
        lines.append("")
        meta_bits = [f"_{r.answer_type}_" if r.answer_type else ""]
        if r.judge:
            badge = "🔴" if r.judge.overall < badcase_threshold else ("⭐" if r.judge.overall == 5 else "🟢")
            meta_bits.append(f"{badge} overall={r.judge.overall}")
        lines.append(" · ".join(b for b in meta_bits if b))
        lines.append("")

        lines.append("### Question")
        lines.append(r.question or "_(空)_")
        lines.append("")

        lines.append("### Ideal Answer")
        lines.append(r.ideal_answer or "_(空)_")
        lines.append("")

        lines.append("### Model Answer")
        if r.error:
            lines.append(f"> ⚠️ 运行错误：`{r.error}`")
            lines.append("")
        lines.append(r.model_answer or "_(空)_")
        lines.append("")

        lines.append("### Retrieved Sources")
        if r.retrieved:
            for s in r.retrieved:
                lines.append(
                    f"- **[{s.sid}]** {s.source} | page {s.page} | distance={s.distance:.4f}"
                )
        else:
            lines.append("_(无)_")
        lines.append("")

        if r.judge:
            lines.append("### Judge Evaluation")
            lines.append("")
            lines.append(f"_Judge model: {r.judge.judge_model or '-'}_")
            lines.append("")
            lines.append("| 维度 | 评分 |")
            lines.append("| --- | --- |")
            lines.append(f"| correctness | {_score_bar(r.judge.correctness)} |")
            lines.append(f"| evidence relevance | {_score_bar(r.judge.evidence_relevance)} |")
            lines.append(f"| faithfulness | {_score_bar(r.judge.faithfulness)} |")
            lines.append(f"| completeness | {_score_bar(r.judge.completeness)} |")
            lines.append(f"| citation | {_score_bar(r.judge.citation)} |")
            lines.append(f"| **overall** | **{_score_bar(r.judge.overall)}** |")
            lines.append("")
            lines.append(f"- **error_type**: `{r.judge.error_type}`")
            lines.append(f"- **reason**: {r.judge.reason}")
            lines.append("")
        else:
            lines.append("### Judge Evaluation")
            lines.append("")
            lines.append("_未评测_")
            lines.append("")

        if r.human_review and (r.human_review.score is not None or r.human_review.notes):
            lines.append("### Human Review")
            lines.append("")
            lines.append(f"- score: {r.human_review.score}")
            lines.append(f"- notes: {r.human_review.notes}")
            lines.append(f"- reviewed_by: {r.human_review.reviewed_by}")
            lines.append("")

        lines.append("---")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def to_csv(run: EvalRun, path: Path) -> Path:
    """Flat CSV (no long-text fields) suitable for spreadsheets / pandas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "qid",
                "answer_type",
                "question",
                "correctness",
                "evidence_relevance",
                "faithfulness",
                "completeness",
                "citation",
                "overall",
                "error_type",
                "reason",
                "n_retrieved",
                "top1_source",
                "top1_distance",
                "has_error",
            ]
        )
        for r in run.results:
            j = r.judge
            top1 = r.retrieved[0] if r.retrieved else None
            writer.writerow(
                [
                    r.qid,
                    r.answer_type,
                    r.question,
                    j.correctness if j else "",
                    j.evidence_relevance if j else "",
                    j.faithfulness if j else "",
                    j.completeness if j else "",
                    j.citation if j else "",
                    j.overall if j else "",
                    j.error_type if j else "",
                    j.reason if j else "",
                    len(r.retrieved),
                    top1.source if top1 else "",
                    f"{top1.distance:.4f}" if top1 else "",
                    int(bool(r.error)),
                ]
            )
    return path


def to_index_md(runs: Iterable[EvalRun], path: Path) -> Path:
    """Write outputs/INDEX.md summarizing all runs."""
    lines = ["# Evaluation Runs Index", ""]
    lines.append("| run_id | timestamp | rag_model | prompt | n | avg_overall | badcase | judge |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for run in sorted(runs, key=lambda r: r.timestamp, reverse=True):
        s = run.summary
        lines.append(
            f"| {run.run_id} | {run.timestamp} | `{run.config.rag_model or '-'}` "
            f"| `{run.config.rag_prompt_version or '-'}` "
            f"| {s.n_questions if s else 0} "
            f"| {s.avg_overall if s else 0} "
            f"| {s.badcase_count if s else 0} "
            f"| `{run.config.judge_model or '-'}` |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
