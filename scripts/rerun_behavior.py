"""Re-run the DEV behavior baseline under the FROZEN retrieval config (section_hybrid).

Keeps rag_chain / evidence gate / refusal / prompts / generation UNCHANGED — the
only change vs the earlier run is the retrieval config. Also splits ANSWERABLE
false refusals into retrieval-miss vs gate-rejected-despite-hit, and attributes
every failed case a primary cause (RETRIEVAL_MISS / WRONG_EVIDENCE /
ANSWERABILITY_MISCLASSIFICATION / GENERATION_ERROR / CITATION_ERROR).
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

from run_eval_v2 import classify_action, load_dev, set_index  # noqa: E402
from validate_eval_v2 import norm  # noqa: E402

EVAL = ROOT / "data" / "eval_v2"


def span_in_contexts(contexts, gold_ev):
    ge = norm(gold_ev)
    return any(ge and ge in norm(h.get("text", "")) for h in contexts)


def attribute(case, pred, span_hit, res):
    cls = case["answerability_class"]
    cv = res.get("citation_validation", {})
    if pred == "answer":
        if cls == "ANSWERABLE":
            if span_hit:
                return "NONE"  # answered with the gold evidence
            if cv.get("status") == "failed" or not cv.get("used_source_ids"):
                return "CITATION_ERROR"
            return "WRONG_EVIDENCE"
        if cls in ("PARTIAL_EVIDENCE", "CONFLICTING_EVIDENCE"):
            return "ANSWERABILITY_MISCLASSIFICATION"
        if cls == "FALSE_PREMISE":
            # did it correct the premise (state the correct value)?
            nums = re.findall(r"\d+(?:\.\d+)?", case.get("answer_key", ""))
            if any(n in (res.get("answer") or "") for n in nums if len(n) >= 2):
                return "NONE"
            return "ANSWERABILITY_MISCLASSIFICATION"
        return "ANSWERABILITY_MISCLASSIFICATION"  # AMBIGUOUS / NO_EVIDENCE answered
    if pred == "refuse":
        if cls == "NO_EVIDENCE":
            return "NONE"  # correct refusal
        if cls == "ANSWERABLE":
            return "RETRIEVAL_MISS" if not span_hit else "ANSWERABILITY_MISCLASSIFICATION"
        if cls == "FALSE_PREMISE":
            return "ANSWERABILITY_MISCLASSIFICATION"  # should correct, not refuse
        return "ANSWERABILITY_MISCLASSIFICATION"
    if pred == "clarify":
        return "NONE" if cls == "AMBIGUOUS" else "ANSWERABILITY_MISCLASSIFICATION"
    return "GENERATION_ERROR"


def main():
    from src.rag_chain import answer_question

    set_index("section_aware_270_gpu", "hybrid_dense_sparse")
    cases = load_dev()
    rows = []
    attr = Counter()
    split = {"retrieval_miss_refused": 0, "gate_rejected_despite_hit": 0, "answered_after_hit": 0}
    tp = {"answer": 0, "clarify": 0, "refuse": 0}
    fp = {"answer": 0, "clarify": 0, "refuse": 0}
    n_class = Counter(c["answerability_class"] for c in cases)

    for c in cases:
        cls = c["answerability_class"]
        try:
            res = answer_question(c["query"], answer_mode="quick")
        except Exception as e:  # noqa: BLE001
            res = {"fallback": True, "fallback_reason": "error",
                   "answer": f"[ERROR] {e}", "citation_validation": {"status": "failed"},
                   "contexts": []}
        pred = classify_action(res)
        span_hit = span_in_contexts(res.get("contexts", []), c.get("gold_evidence_text", ""))
        exp3 = {"answer": "answer", "clarify": "clarify", "refuse": "refuse",
                "partial_answer": "answer", "correct_premise": "answer",
                "present_conflict": "answer"}[c["expected_action"]]
        correct = pred == exp3
        if correct:
            tp[pred] += 1
        fp[pred] += 1

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

        cv = res.get("citation_validation", {})
        rows.append({
            "case_id": c["case_id"], "class": cls, "expected": c["expected_action"],
            "predicted": pred, "action_correct": int(correct),
            "span_hit": int(span_hit), "primary_cause": cause,
            "answer": (res.get("answer", "") or "")[:120].replace("\n", " "),
        })

    n = len(cases)
    n_answered = sum(1 for r in rows if r["predicted"] == "answer")
    metrics = {
        "config": "section_hybrid",
        "n": n,
        "predicted_dist": dict(Counter(r["predicted"] for r in rows)),
        "action_accuracy": round(sum(r["action_correct"] for r in rows) / n, 4),
        "clarify_precision": round(tp["clarify"] / max(1, fp["clarify"]), 4),
        "clarify_recall": round(tp["clarify"] / max(1, n_class["AMBIGUOUS"]), 4),
        "refuse_precision": round(tp["refuse"] / max(1, fp["refuse"]), 4),
        "refuse_recall": round(tp["refuse"] / max(1, n_class["NO_EVIDENCE"]), 4),
        "false_premise_correction": round(
            sum(1 for r in rows if r["class"] == "FALSE_PREMISE" and r["primary_cause"] == "NONE")
            / max(1, n_class["FALSE_PREMISE"]), 4),
        "unsupported_claim_rate": round(
            sum(1 for r in rows if r["predicted"] == "answer" and r["primary_cause"] in ("CITATION_ERROR", "WRONG_EVIDENCE"))
            / max(1, n_answered), 4),
        "citation_support_rate": round(
            sum(1 for r in rows if r["predicted"] == "answer" and r["primary_cause"] == "NONE")
            / max(1, n_answered), 4),
        "answerable_split": split,
        "error_attribution": dict(attr),
    }

    with (EVAL / "dev_behavior_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    main()
