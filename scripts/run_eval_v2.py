"""Run Eval V2: retrieval benchmark + behavior baseline (dev only).

Retrieval: 4 configs = {fixed_270_gpu, section_aware_270_gpu} x {Dense, Hybrid}.
    Metrics per config: Recall@5/10, MRR@10, Hit@5/10, Evidence Page Hit@5/10.

Behavior: current answer_question() baseline on dev. Metrics:
    Action Accuracy, Clarification P/R, Refusal P/R, False-premise Correction
    Rate, Unsupported Claim Rate, Citation Support Rate.

Writes data/eval_v2/dev_retrieval_results.csv, dev_behavior_results.csv,
baseline_report.md.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402
from src.embedder import embed_query  # noqa: E402
from src.retriever import clear_retrieval_caches  # noqa: E402
from src.vectorstore import clear_vectorstore_caches, search  # noqa: E402

EVAL = ROOT / "data" / "eval_v2"
DEV = EVAL / "dev.jsonl"

CONFIGS = [
    ("fixed_dense", "fixed_270_gpu", "dense_only"),
    ("section_dense", "section_aware_270_gpu", "dense_only"),
    ("fixed_hybrid", "fixed_270_gpu", "hybrid_dense_sparse"),
    ("section_hybrid", "section_aware_270_gpu", "hybrid_dense_sparse"),
]


def load_dev():
    return [json.loads(l) for l in DEV.read_text(encoding="utf-8").splitlines() if l.strip()]


def set_index(index_name, mode):
    settings.chroma_dir = str(ROOT / f"chroma_db_{index_name}")
    settings.collection_name = index_name
    settings.sparse_index_dir = str(ROOT / f"sparse_index_{index_name}")
    settings.retrieval_mode = mode
    clear_vectorstore_caches()
    clear_retrieval_caches()
    from src.sparse_index import clear_index_cache
    clear_index_cache()


def raw_retrieve(query, top_k, mode):
    emb = embed_query(query)
    if mode == "dense_only":
        return search(emb, top_k=top_k)
    from src.sparse_encoder import encode_query_sparse
    from src.sparse_index import load_index
    from src.fusion import resolve_sparse_hits, rrf_fuse
    dense = search(emb, top_k=settings.hybrid_dense_k)
    qw = encode_query_sparse(query)
    scored = load_index(strict=True).search(qw, top_k=settings.hybrid_sparse_k)
    sparse = resolve_sparse_hits(scored)
    return rrf_fuse(dense, sparse, fusion_k=top_k)[:top_k]


def paper_hit(hits, gold_pid):
    return [h.get("metadata", {}).get("paper_id") for h in hits]


def page_hit(hits, gold_pid, gold_page):
    for h in hits:
        m = h.get("metadata", {})
        if m.get("paper_id") != gold_pid:
            continue
        ps, pe = m.get("page_start"), m.get("page_end")
        if ps is not None and ps <= gold_page <= pe:
            return True
    return False


def retrieval_benchmark():
    cases = [c for c in load_dev() if c.get("gold_paper_id")]
    rows = []
    summary = {}
    for cfg_name, index, mode in CONFIGS:
        set_index(index, mode)
        per = {k: 0 for k in ("recall5", "recall10", "mrr10", "page5", "page10")}
        n = 0
        for c in cases:
            hits = raw_retrieve(c["query"], 10, mode)
            pids = paper_hit(hits, c["gold_paper_id"])
            rank = None
            for i, pid in enumerate(pids, 1):
                if pid == c["gold_paper_id"]:
                    rank = i
                    break
            r5 = 1 if rank is not None and rank <= 5 else 0
            r10 = 1 if rank is not None else 0
            mrr = 1.0 / rank if rank else 0.0
            p5 = 1 if page_hit(hits[:5], c["gold_paper_id"], c["gold_page_start"]) else 0
            p10 = 1 if page_hit(hits[:10], c["gold_paper_id"], c["gold_page_start"]) else 0
            per["recall5"] += r5
            per["recall10"] += r10
            per["mrr10"] += mrr
            per["page5"] += p5
            per["page10"] += p10
            n += 1
            rows.append({"config": cfg_name, "case_id": c["case_id"],
                         "recall5": r5, "recall10": r10, "mrr10": round(mrr, 4),
                         "page_hit5": p5, "page_hit10": p10})
        summary[cfg_name] = {k: round(v / n, 4) for k, v in per.items()}
        summary[cfg_name]["n"] = n
        print(f"{cfg_name}: " + " ".join(f"{k}={v}" for k, v in summary[cfg_name].items()))

    with (EVAL / "dev_retrieval_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return summary


def classify_action(result):
    if result.get("fallback"):
        r = result.get("fallback_reason")
        return "clarify" if r == "needs_clarification" else "refuse"
    answer = result.get("answer", "") or ""
    if "没有足够证据" in answer:
        return "refuse"
    if "缺少关键限定" in answer:
        return "clarify"
    return "answer"


def behavior_baseline():
    from src.rag_chain import answer_question

    set_index("section_aware_270_gpu", "dense_only")
    cases = load_dev()
    rows = []
    tp = {"clarify": 0, "refuse": 0, "answer": 0}
    fp = {"clarify": 0, "refuse": 0, "answer": 0}
    n_class = {}
    for c in cases:
        cls = c["answerability_class"]
        n_class[cls] = n_class.get(cls, 0) + 1
        try:
            res = answer_question(c["query"], answer_mode="quick")
        except Exception as e:  # noqa: BLE001
            res = {"fallback": True, "fallback_reason": "error", "answer": f"[ERROR] {e}",
                   "citation_validation": {"status": "failed"}}
        pred = classify_action(res)
        exp = c["expected_action"]
        # map 6-class expected to 3-class for exact action accuracy
        exp3 = {"answer": "answer", "clarify": "clarify", "refuse": "refuse",
                "partial_answer": "answer", "correct_premise": "answer",
                "present_conflict": "answer"}[exp]
        correct = pred == exp3
        if correct:
            tp[pred] += 1
        fp[pred] += 1

        # false-premise correction: answer contains gold answer_key (correct value)
        corrects = (cls == "FALSE_PREMISE" and c.get("answer_key")
                    and c["answer_key"] in res.get("answer", ""))
        # unsupported claim / citation support: only meaningful when the system
        # actually produced a substantive answer (not a refusal / clarification).
        cv = res.get("citation_validation", {})
        answered = (pred == "answer")
        unsupported = (answered and
                       (cv.get("status") in ("failed",) or not cv.get("used_source_ids")))
        citation_ok = (answered and cv.get("status") == "passed")

        rows.append({
            "case_id": c["case_id"], "class": cls, "expected": exp,
            "predicted": pred, "action_correct": int(correct),
            "false_premise_corrected": int(corrects),
            "unsupported_claim": int(unsupported),
            "citation_supported": int(citation_ok),
            "answer": (res.get("answer", "") or "")[:200].replace("\n", " "),
        })

    n = len(cases)
    n_answered = sum(1 for r in rows if r["predicted"] == "answer")
    metrics = {
        "n": n,
        "predicted_dist": {"answer": sum(1 for r in rows if r["predicted"] == "answer"),
                           "clarify": sum(1 for r in rows if r["predicted"] == "clarify"),
                           "refuse": sum(1 for r in rows if r["predicted"] == "refuse")},
        "action_accuracy": round(sum(r["action_correct"] for r in rows) / n, 4),
        "clarify_precision": round(tp["clarify"] / max(1, fp["clarify"]), 4),
        "clarify_recall": round(tp["clarify"] / max(1, n_class.get("AMBIGUOUS", 0)), 4),
        "refuse_precision": round(tp["refuse"] / max(1, fp["refuse"]), 4),
        "refuse_recall": round(tp["refuse"] / max(1, n_class.get("NO_EVIDENCE", 0)), 4),
        "false_premise_correction": round(
            sum(r["false_premise_corrected"] for r in rows) / max(1, n_class.get("FALSE_PREMISE", 0)), 4),
        "unsupported_claim_rate": round(
            sum(r["unsupported_claim"] for r in rows) / max(1, n_answered), 4),
        "citation_support_rate": round(
            sum(r["citation_supported"] for r in rows) / max(1, n_answered), 4),
        "per_class": n_class,
    }
    with (EVAL / "dev_behavior_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("behavior metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def write_report(ret, beh):
    lines = ["# Eval V2 Baseline Report", "",
             f"- cases (dev): **{beh['n']}**", ""]
    lines += ["## Dev retrieval metrics", "",
              "| config | Recall@5 | Recall@10 | MRR@10 | PageHit@5 | PageHit@10 |", "|---|---|---|---|---|---|"]
    for cfg, m in ret.items():
        lines.append(f"| {cfg} | {m['recall5']} | {m['recall10']} | {m['mrr10']} | {m['page5']} | {m['page10']} |")
    lines += ["", "## Behavior baseline (section_aware dense)", "",
              f"- Predicted action distribution: **{beh['predicted_dist']}**",
              f"- Action Accuracy: **{beh['action_accuracy']}**",
              f"- Clarification Precision/Recall: **{beh['clarify_precision']} / {beh['clarify_recall']}**",
              f"- Refusal Precision/Recall: **{beh['refuse_precision']} / {beh['refuse_recall']}**",
              f"- False-premise Correction Rate: **{beh['false_premise_correction']}**",
              f"- Unsupported Claim Rate: **{beh['unsupported_claim_rate']}**",
              f"- Citation Support Rate: **{beh['citation_support_rate']}**",
              "", "## Class distribution (dev)", "",
              f"`{beh['per_class']}`", ""]
    (EVAL / "baseline_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {EVAL / 'baseline_report.md'}")


def main():
    ret = retrieval_benchmark()
    beh = behavior_baseline()
    write_report(ret, beh)


if __name__ == "__main__":
    main()
