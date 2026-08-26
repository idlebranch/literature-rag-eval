"""DEV behavior eval after Phase D answerability routing (section_hybrid).

Reads the 6-class ``action`` now emitted by rag_chain.answer_question and
compares it directly against expected_action; also re-derives error attribution
and the ANSWERABLE retrieval-vs-gate split.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_eval_v2 import load_dev, set_index  # noqa: E402
from validate_eval_v2 import norm  # noqa: E402

EVAL = ROOT / "data" / "eval_v2"


def span_in_contexts(contexts, gold_ev):
    ge = norm(gold_ev)
    return any(ge and ge in norm(h.get("text", "")) for h in contexts)


def attribute(case, pred, span_hit, res):
    cls = case["answerability_class"]
    if pred == case["expected_action"]:
        return "NONE"
    if cls == "ANSWERABLE" and pred in ("refuse", "answer"):
        if pred == "refuse":
            return "RETRIEVAL_MISS" if not span_hit else "ANSWERABILITY_MISCLASSIFICATION"
        return "WRONG_EVIDENCE"
    return "ANSWERABILITY_MISCLASSIFICATION"


def main():
    from src.rag_chain import answer_question

    set_index("section_aware_270_gpu", "hybrid_dense_sparse")
    cases = load_dev()
    rows = []
    split = {"retrieval_miss_refused": 0, "gate_rejected_despite_hit": 0, "answered_after_hit": 0}
    attr = Counter()
    n_class = Counter(c["answerability_class"] for c in cases)

    for c in cases:
        cls = c["answerability_class"]
        try:
            res = answer_question(c["query"], answer_mode="quick")
        except Exception as e:  # noqa: BLE001
            res = {"fallback": True, "action": "refuse", "answer": f"[ERROR] {e}",
                   "citation_validation": {"status": "failed"}, "contexts": []}
        pred = res.get("action", "answer")
        # the LLM may refuse even when the deterministic route defaulted to
        # "answer"; reflect the actual behaviour for the action measurement.
        if pred == "answer" and "没有足够证据" in (res.get("answer", "") or ""):
            pred = "refuse"
        span_hit = span_in_contexts(res.get("contexts", []), c.get("gold_evidence_text", ""))
        correct = pred == c["expected_action"]
        cause = attribute(c, pred, span_hit, res)
        if cause != "NONE":
            attr[cause] += 1
        if cls == "ANSWERABLE":
            if pred == "refuse":
                if not span_hit:
                    split["retrieval_miss_refused"] += 1
                else:
                    split["gate_rejected_despite_hit"] += 1
            elif pred == "answer" and span_hit:
                split["answered_after_hit"] += 1
        rows.append({"case_id": c["case_id"], "class": cls, "expected": c["expected_action"],
                     "predicted": pred, "action_correct": int(correct),
                     "span_hit": int(span_hit), "primary_cause": cause,
                     "answer": (res.get("answer", "") or "")[:120].replace("\n", " ")})

    n = len(cases)
    metrics = {
        "n": n,
        "predicted_dist": dict(Counter(r["predicted"] for r in rows)),
        "action_accuracy": round(sum(r["action_correct"] for r in rows) / n, 4),
        "clarify_precision": _p_r(rows, "clarify", "AMBIGUOUS")[0],
        "clarify_recall": _p_r(rows, "clarify", "AMBIGUOUS")[1],
        "refuse_precision": _p_r(rows, "refuse", "NO_EVIDENCE")[0],
        "refuse_recall": _p_r(rows, "refuse", "NO_EVIDENCE")[1],
        "false_premise_correction": round(
            sum(1 for r in rows if r["class"] == "FALSE_PREMISE" and r["predicted"] == "correct_premise")
            / max(1, n_class["FALSE_PREMISE"]), 4),
        "partial_rate": round(
            sum(1 for r in rows if r["class"] == "PARTIAL_EVIDENCE" and r["predicted"] == "partial_answer")
            / max(1, n_class["PARTIAL_EVIDENCE"]), 4),
        "conflict_rate": round(
            sum(1 for r in rows if r["class"] == "CONFLICTING_EVIDENCE" and r["predicted"] == "present_conflict")
            / max(1, n_class["CONFLICTING_EVIDENCE"]), 4),
        "answerable_split": split,
        "error_attribution": dict(attr),
    }

    with (EVAL / "dev_behavior_results_phase_d.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def _p_r(rows, action, cls):
    preds = [r for r in rows if r["predicted"] == action]
    golds = [r for r in rows if r["class"] == cls]
    tp = sum(1 for r in rows if r["predicted"] == action and r["class"] == cls)
    prec = round(tp / max(1, len(preds)), 4)
    rec = round(tp / max(1, len(golds)), 4)
    return prec, rec


if __name__ == "__main__":
    main()
