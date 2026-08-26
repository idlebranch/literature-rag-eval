"""Retrieval ablation: Dense vs Hybrid vs Hybrid+Reranker.

Runs the project's existing groundtruth set through ``retrieve_with_metrics``
under each configured retrieval mode and reports retrieval-quality metrics plus
latency. This script does not invent a new evaluation framework; it reuses the
same expected-source matching semantics as ``src/eval_retrieval.py``.

Usage:
    python -m scripts.ablation_retrieval [--modes dense_only,hybrid_dense_sparse,hybrid_reranker]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.config import settings
from src.retriever import clear_retrieval_caches, retrieve_with_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)

GROUNDTRUTH_PATH = Path(settings.project_root) / "groundtruth" / "groundtruth.example.jsonl"
OUTPUT_PATH = Path(settings.output_dir) / "retrieval_ablation.json"

# Keep enough context so top-K evaluation is never truncated by the budget.
_EVAL_CONTEXT_BUDGET = 100_000


def load_groundtruth(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _matches(expected: str, source: str, paper_id: str) -> bool:
    return expected in source or expected in paper_id


def score_hits(expected_sources, hits):
    """Return per-K hit/recall bookkeeping for one question."""
    retrieved = []
    for h in hits:
        meta = h.get("metadata") or {}
        retrieved.append((str(meta.get("source", "")), str(meta.get("paper_id", ""))))

    result = {}
    first_match_rank = None
    for k in (5, 10):
        topk = retrieved[:k]
        matched_sources = {
            exp
            for exp in expected_sources
            if any(_matches(exp, src, pid) for src, pid in topk)
        }
        result[f"hit@{k}"] = 1 if matched_sources else 0
        result[f"recall@{k}"] = (
            len(matched_sources) / len(expected_sources) if expected_sources else 0.0
        )

    for rank, (src, pid) in enumerate(retrieved[:10], start=1):
        if any(_matches(exp, src, pid) for exp in expected_sources):
            first_match_rank = rank
            break
    result["mrr@10"] = 1.0 / first_match_rank if first_match_rank else 0.0
    result["first_match_rank"] = first_match_rank
    return result


def run_mode(mode: str, cases, top_k: int):
    settings.retrieval_mode = mode
    settings.context_token_budget = _EVAL_CONTEXT_BUDGET
    clear_retrieval_caches()

    per_case = []
    totals = {f"hit@{k}": 0 for k in (5, 10)}
    totals.update({f"recall@{k}": 0.0 for k in (5, 10)})
    totals["mrr@10"] = 0.0
    latency_ms_total = 0.0
    stage_totals = {
        "query_rewrite_ms": 0.0,
        "query_embedding_ms": 0.0,
        "chroma_search_ms": 0.0,
        "sparse_search_ms": 0.0,
        "fusion_ms": 0.0,
        "rerank_ms": 0.0,
        "filter_diversify_ms": 0.0,
    }
    errors = 0

    for case in cases:
        question = case["question"]
        expected_sources = case.get("expected_sources", [])
        started = time.perf_counter()
        try:
            result = retrieve_with_metrics(question, top_k=top_k)
        except Exception as e:
            errors += 1
            per_case.append(
                {
                    "id": case.get("id"),
                    "question": question,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            logger.warning("[%s] %s failed: %s", mode, case.get("id"), e)
            continue
        latency_ms = (time.perf_counter() - started) * 1000
        latency_ms_total += latency_ms

        metrics = score_hits(expected_sources, result.hits)
        for key in totals:
            totals[key] += metrics[key]
        for key in stage_totals:
            stage_totals[key] += getattr(result, key, 0.0) or 0.0

        per_case.append(
            {
                "id": case.get("id"),
                "question": question,
                "expected_sources": expected_sources,
                "retrieved": [
                    {
                        "source": (h.get("metadata") or {}).get("source", ""),
                        "page": (h.get("metadata") or {}).get("page", ""),
                        "chunk_id": h.get("id"),
                    }
                    for h in result.hits
                ],
                "metrics": metrics,
                "latency_ms": round(latency_ms, 1),
            }
        )

    n = len(cases)
    summary = {
        "mode": mode,
        "cases": n,
        "errors": errors,
        "hit@5": totals["hit@5"] / n,
        "hit@10": totals["hit@10"] / n,
        "recall@5": totals["recall@5"] / n,
        "recall@10": totals["recall@10"] / n,
        "mrr@10": totals["mrr@10"] / n,
        "avg_latency_ms": latency_ms_total / n,
        "avg_stage_ms": {k: v / n for k, v in stage_totals.items()},
    }
    return summary, per_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        default="dense_only,hybrid_dense_sparse,hybrid_reranker",
        help="Comma-separated retrieval modes to evaluate.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    cases = load_groundtruth(GROUNDTRUTH_PATH)
    logger.info("Loaded %d groundtruth questions", len(cases))

    import torch

    report = {
        "groundtruth": str(GROUNDTRUTH_PATH),
        "chunk_count": None,
        "reranker_device": "cuda" if torch.cuda.is_available() else "cpu",
        "modes": {},
    }

    from src.vectorstore import get_collection_count

    report["chunk_count"] = get_collection_count()

    for mode in modes:
        logger.info("=== Running mode: %s ===", mode)
        try:
            summary, per_case = run_mode(mode, cases, args.top_k)
        except Exception as e:
            logger.error("Mode %s failed to run: %s", mode, e)
            report["modes"][mode] = {"status": "NOT RUN", "reason": str(e)}
            continue
        report["modes"][mode] = {"status": "ok", "summary": summary, "cases": per_case}
        logger.info(
            "[%s] hit@5=%.3f hit@10=%.3f recall@5=%.3f recall@10=%.3f mrr@10=%.3f latency=%.0fms",
            mode,
            summary["hit@5"],
            summary["hit@10"],
            summary["recall@5"],
            summary["recall@10"],
            summary["mrr@10"],
            summary["avg_latency_ms"],
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved ablation report to %s", OUTPUT_PATH)

    # Console comparison table
    print("\n=== Retrieval Ablation ===")
    print(f"chunks={report['chunk_count']} reranker_device={report['reranker_device']}")
    header = f"{'mode':<22}{'hit@5':>8}{'hit@10':>8}{'recall@5':>9}{'recall@10':>10}{'mrr@10':>8}{'latency':>9}"
    print(header)
    for mode, payload in report["modes"].items():
        if payload.get("status") != "ok":
            print(f"{mode:<22} NOT RUN: {payload.get('reason','')[:60]}")
            continue
        s = payload["summary"]
        print(
            f"{mode:<22}{s['hit@5']:>8.3f}{s['hit@10']:>8.3f}"
            f"{s['recall@5']:>9.3f}{s['recall@10']:>10.3f}"
            f"{s['mrr@10']:>8.3f}{s['avg_latency_ms']:>8.0f}ms"
        )


if __name__ == "__main__":
    main()
