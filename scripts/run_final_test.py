"""FINAL held-out TEST evaluation (one-shot, 35 cases).

Frozen: section_hybrid + Phase E answerability + existing generation/citation.
Computes retrieval, answerability, trustworthiness, error attribution, and
DEV->TEST generalization. Never modifies the system.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_eval_v2 import load_dev, set_index  # noqa: E402
from run_eval_v2 import raw_retrieve  # noqa: E402
from validate_eval_v2 import norm  # noqa: E402

EVAL = ROOT / "data" / "eval_v2"
TEST = EVAL / "test.jsonl"
DEV = EVAL / "dev.jsonl"

_CITE_RE = re.compile(r"\[S(\d+)\]", re.IGNORECASE)


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def retrieval_row(c, set_index_fn):
    set_index_fn("section_aware_270_gpu", "hybrid_dense_sparse")
    hits = raw_retrieve(c["query"], 10, "hybrid_dense_sparse")
    gold = norm(c["gold_evidence_text"])
    rank = None
    for i, h in enumerate(hits, 1):
        if h["metadata"].get("paper_id") == c["gold_paper_id"]:
            rank = rank or i
    page_hit = any(
        h["metadata"].get("paper_id") == c["gold_paper_id"]
        and (h["metadata"].get("page_start") or 0) <= c["gold_page_start"] <= (h["metadata"].get("page_end") or c["gold_page_start"])
        for h in hits)
    span_hit = any(gold and gold in norm(h.get("text", "")) for h in hits)
    return {
        "case_id": c["case_id"], "rank": rank or 0,
        "recall5": int(rank is not None and rank <= 5),
        "recall10": int(rank is not None),
        "mrr10": round(1.0 / rank, 4) if rank else 0.0,
        "page_hit5": int(any(
            h["metadata"].get("paper_id") == c["gold_paper_id"]
            and (h["metadata"].get("page_start") or 0) <= c["gold_page_start"] <= (h["metadata"].get("page_end") or c["gold_page_start"])
            for h in hits[:5])),
        "page_hit10": int(page_hit),
        "span_hit5": int(any(gold and gold in norm(h.get("text", "")) for h in hits[:5])),
        "span_hit10": int(span_hit),
    }


def main():
    from src.rag_chain import answer_question

    cases = load(TEST)
    ev_cases = [c for c in cases if c.get("gold_paper_id")]
    set_index("section_aware_270_gpu", "hybrid_dense_sparse")

    # ---- retrieval ----
    ret_rows = [retrieval_row(c, set_index) for c in ev_cases]
    n_ret = len(ret_rows)
    ret_metrics = {
        "n": n_ret,
        "recall5": round(sum(r["recall5"] for r in ret_rows) / n_ret, 4),
        "recall10": round(sum(r["recall10"] for r in ret_rows) / n_ret, 4),
        "mrr10": round(sum(r["mrr10"] for r in ret_rows) / n_ret, 4),
        "page_hit5": round(sum(r["page_hit5"] for r in ret_rows) / n_ret, 4),
        "page_hit10": round(sum(r["page_hit10"] for r in ret_rows) / n_ret, 4),
        "span_hit5": round(sum(r["span_hit5"] for r in ret_rows) / n_ret, 4),
        "span_hit10": round(sum(r["span_hit10"] for r in ret_rows) / n_ret, 4),
    }

    # ---- behavior + trustworthiness ----
    beh_rows, claim_rows = [], []
    class_acc = Counter(); class_n = Counter()
    tp = {"clarify": 0, "refuse": 0}; fp = {"clarify": 0, "refuse": 0}
    attr = Counter()
    n_class = Counter(c["answerability_class"] for c in cases)

    for c in cases:
        cls = c["answerability_class"]
        try:
            res = answer_question(c["query"], answer_mode="quick")
        except Exception as e:  # noqa: BLE001
            res = {"action": "refuse", "answer": f"[ERROR] {e}",
                   "citation_validation": {"status": "failed"}, "contexts": []}
        pred = res.get("action", "answer")
        if pred == "answer" and "没有足够证据" in (res.get("answer", "") or ""):
            pred = "refuse"
        correct = pred == c["expected_action"]
        class_n[cls] += 1
        if correct:
            class_acc[cls] += 1
        if pred == "clarify":
            fp["clarify"] += 1
            if cls == "AMBIGUOUS":
                tp["clarify"] += 1
        if pred == "refuse":
            fp["refuse"] += 1
            if cls == "NO_EVIDENCE":
                tp["refuse"] += 1

        span_hit = any(norm(c.get("gold_evidence_text","")) and norm(c.get("gold_evidence_text","")) in norm(h.get("text",""))
                       for h in res.get("contexts", []))

        # error attribution
        cause = "NONE" if correct else _attribute(c, pred, span_hit, res)
        if cause != "NONE":
            attr[cause] += 1

        # claim classification (trustworthiness) for answered evidence cases
        claim = _claim_classify(c, res)

        beh_rows.append({"case_id": c["case_id"], "class": cls,
                         "expected": c["expected_action"], "predicted": pred,
                         "correct": int(correct), "primary_cause": cause,
                         "answer": (res.get("answer", "") or "")[:140].replace("\n", " ")})
        claim_rows.append({"case_id": c["case_id"], "class": cls, "predicted": pred,
                           "claim_support": claim,
                           "answer": (res.get("answer", "") or "")[:140].replace("\n", " ")})

    n = len(cases)
    beh_metrics = {
        "n": n,
        "action_accuracy": round(sum(r["correct"] for r in beh_rows) / n, 4),
        "per_class_accuracy": {k: round(class_acc[k] / class_n[k], 4) for k in sorted(class_n)},
        "clarify_precision": round(tp["clarify"] / max(1, fp["clarify"]), 4),
        "clarify_recall": round(tp["clarify"] / max(1, n_class["AMBIGUOUS"]), 4),
        "refuse_precision": round(tp["refuse"] / max(1, fp["refuse"]), 4),
        "refuse_recall": round(tp["refuse"] / max(1, n_class["NO_EVIDENCE"]), 4),
        "false_premise_correction": round(
            sum(1 for r in beh_rows if r["class"] == "FALSE_PREMISE" and r["predicted"] == "correct_premise")
            / max(1, n_class["FALSE_PREMISE"]), 4),
        "error_attribution": dict(attr),
    }
    claim_counts = Counter(r["claim_support"] for r in claim_rows)
    trust = {
        "answered_but_should_refuse": sum(1 for r in beh_rows if r["class"] in ("AMBIGUOUS", "NO_EVIDENCE") and r["predicted"] == "answer"),
        "refused_but_should_answer": sum(1 for r in beh_rows if r["class"] == "ANSWERABLE" and r["predicted"] == "refuse"),
        "near_evidence_masquerade": sum(1 for r in claim_rows if r["class"] == "ANSWERABLE" and r["predicted"] == "answer" and r["claim_support"] == "UNSUPPORTED"),
        "unsupported_claim_rate": round(claim_counts["UNSUPPORTED"] / max(1, claim_counts["SUPPORTED"] + claim_counts["UNSUPPORTED"] + claim_counts["PARTIALLY_SUPPORTED"]), 4),
        "citation_support_rate": round(claim_counts["SUPPORTED"] / max(1, sum(1 for r in claim_rows if r["predicted"] == "answer")), 4),
        "claim_distribution": dict(claim_counts),
    }

    # ---- write outputs ----
    with (EVAL / "test_retrieval_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(ret_rows[0].keys())); w.writeheader(); w.writerows(ret_rows)
    with (EVAL / "test_behavior_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(beh_rows[0].keys())); w.writeheader(); w.writerows(beh_rows)
    with (EVAL / "test_claim_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(claim_rows[0].keys())); w.writeheader(); w.writerows(claim_rows)

    print(json.dumps({"retrieval": ret_metrics, "behavior": beh_metrics, "trust": trust},
                     ensure_ascii=False, indent=2))


def _attribute(c, pred, span_hit, res):
    cls = c["answerability_class"]
    if cls == "ANSWERABLE" and pred == "refuse":
        return "RETRIEVAL_MISS" if not span_hit else "ANSWERABILITY_MISCLASSIFICATION"
    if cls == "ANSWERABLE" and pred == "answer" and not span_hit:
        return "WRONG_EVIDENCE"
    return "ANSWERABILITY_MISCLASSIFICATION"


def _claim_classify(c, res):
    if res.get("predicted") == "refuse" and res.get("predicted") == "refuse":
        pass
    pred = res.get("action", "answer")
    if pred == "answer" and "没有足够证据" in (res.get("answer", "") or ""):
        pred = "refuse"
    if pred != "answer":
        return "NO_CLAIM"
    answer = res.get("answer", "")
    cited = _CITE_RE.findall(answer)
    if not cited:
        return "NO_CITATION"
    gold = norm(c.get("gold_evidence_text", ""))
    contexts = res.get("contexts", [])
    for sid in cited:
        idx = int(sid) - 1
        if 0 <= idx < len(contexts) and gold and gold in norm(contexts[idx].get("text", "")):
            return "SUPPORTED"
    return "UNSUPPORTED"


if __name__ == "__main__":
    main()
