"""FINAL ACCEPTANCE one-shot run (32 fresh cases, never-used gold paper_ids).

Frozen pipeline: section_hybrid retrieval + Phase D answerability (no auto
conflict) + Phase E citation validation. Computes the pre-fixed Release Gate,
writes data/acceptance/*.csv + acceptance_report.md, and prints a verdict.
Never modifies the RAG kernel.
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

from run_eval_v2 import raw_retrieve, set_index  # noqa: E402
from validate_acceptance import norm  # noqa: E402

ACC = ROOT / "data" / "acceptance"
CASES = [json.loads(l) for l in (ACC / "acceptance.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

_CITE_RE = re.compile(r"\[S(\d+)\]", re.IGNORECASE)
SUBSTANTIVE = {"answer", "partial_answer", "correct_premise"}

GATE = [
    ("Recall@10", "recall10", 0.70, ">="),
    ("PageHit@10", "page_hit10", 0.55, ">="),
    ("EvidenceSpanHit@10", "span_hit10", 0.50, ">="),
    ("Action Accuracy", "action_accuracy", 0.60, ">="),
    ("ANSWERABLE accuracy", "answerable_accuracy", 0.55, ">="),
    ("NO_EVIDENCE Recall", "no_evidence_recall", 0.80, ">="),
    ("Citation Support", "citation_support", 0.65, ">="),
    ("Unsupported Claim", "unsupported_claim", 0.25, "<="),
]

# Per-case keyword checks for CONDITIONALLY_DIVERGENT (eval harness only).
CONDITIONAL_CHECKS = {
    "acc_f001": ["煤质", "椰壳"],
    "acc_f002": ["46.64", "34.64"],
}


def load():
    return CASES


def retrieval_metrics():
    set_index("section_aware_270_gpu", "hybrid_dense_sparse")
    rows = []
    for c in CASES:
        if not c.get("gold_paper_id"):
            continue
        hits = raw_retrieve(c["query"], 10, "hybrid_dense_sparse")
        gold = norm(c["gold_evidence_text"])
        rank = next((i for i, h in enumerate(hits, 1)
                     if h["metadata"].get("paper_id") == c["gold_paper_id"]), None)
        page_hit = any(
            h["metadata"].get("paper_id") == c["gold_paper_id"]
            and (h["metadata"].get("page_start") or 0) <= c["gold_page_start"] <= (h["metadata"].get("page_end") or c["gold_page_start"])
            for h in hits)
        span_hit = any(gold and gold in norm(h.get("text", "")) for h in hits)
        rows.append({
            "case_id": c["case_id"], "rank": rank or 0,
            "recall5": int(rank is not None and rank <= 5),
            "recall10": int(rank is not None),
            "page_hit10": int(page_hit),
            "span_hit10": int(span_hit),
        })
    n = len(rows)
    return {
        "n": n,
        "recall10": round(sum(r["recall10"] for r in rows) / n, 4),
        "page_hit10": round(sum(r["page_hit10"] for r in rows) / n, 4),
        "span_hit10": round(sum(r["span_hit10"] for r in rows) / n, 4),
    }, rows


def predicted_action(res):
    pred = res.get("action", "answer")
    if pred == "answer" and "没有足够证据" in (res.get("answer", "") or ""):
        return "refuse"
    return pred


def behavior_metrics():
    from src.rag_chain import answer_question
    set_index("section_aware_270_gpu", "hybrid_dense_sparse")

    rows = []
    class_acc = Counter(); class_n = Counter()
    tp = {"clarify": 0, "refuse": 0}; fp = {"clarify": 0, "refuse": 0}
    for c in CASES:
        cls = c["answerability_class"]
        try:
            res = answer_question(c["query"], answer_mode="quick")
        except Exception as e:  # noqa: BLE001
            res = {"action": "refuse", "answer": f"[ERROR] {e}",
                   "citation_validation": {"status": "failed", "used_source_ids": []}, "contexts": []}
        pred = predicted_action(res)
        exp = c["expected_action"]
        correct = pred == exp
        class_n[cls] += 1
        class_acc[cls] += int(correct)
        if pred == "clarify":
            fp["clarify"] += 1
            tp["clarify"] += int(cls == "AMBIGUOUS")
        if pred == "refuse":
            fp["refuse"] += 1
            tp["refuse"] += int(cls == "NO_EVIDENCE")

        answered = pred in SUBSTANTIVE
        cv = res.get("citation_validation", {})
        citation_ok = answered and cv.get("status") == "passed"
        unsupported = answered and (cv.get("status") == "failed" or not cv.get("used_source_ids"))

        # claim-level evidence support (cited source actually contains gold)
        claim = _claim_classify(c, res, pred)

        # conditional divergence qualitative check
        cond_ok = None
        if cls == "CONDITIONALLY_DIVERGENT":
            cond_ok = _conditional_ok(c, res, pred)

        rows.append({
            "case_id": c["case_id"], "class": cls, "expected": exp,
            "predicted": pred, "correct": int(correct),
            "citation_supported": int(citation_ok),
            "unsupported_claim": int(unsupported),
            "claim_support": claim,
            "conditional_ok": "" if cond_ok is None else ("yes" if cond_ok else "no"),
            "answer": (res.get("answer", "") or "")[:160].replace("\n", " "),
        })

    n = len(CASES)
    n_answered = sum(1 for r in rows if r["predicted"] in SUBSTANTIVE)
    claim_counts = Counter(r["claim_support"] for r in rows)
    cond_rows = [r for r in rows if r["class"] == "CONDITIONALLY_DIVERGENT"]
    cond_pass = sum(1 for r in cond_rows if r["conditional_ok"] == "yes")

    metrics = {
        "n": n,
        "predicted_dist": dict(Counter(r["predicted"] for r in rows)),
        "action_accuracy": round(sum(r["correct"] for r in rows) / n, 4),
        "per_class_accuracy": {k: round(class_acc[k] / class_n[k], 4) for k in sorted(class_n)},
        "answerable_accuracy": round(class_acc["ANSWERABLE"] / class_n["ANSWERABLE"], 4),
        "no_evidence_recall": round(class_acc["NO_EVIDENCE"] / class_n["NO_EVIDENCE"], 4),
        "false_premise_correction": round(class_acc["FALSE_PREMISE"] / class_n["FALSE_PREMISE"], 4),
        "clarify_precision": round(tp["clarify"] / max(1, fp["clarify"]), 4),
        "clarify_recall": round(tp["clarify"] / max(1, class_n["AMBIGUOUS"]), 4),
        "refuse_precision": round(tp["refuse"] / max(1, fp["refuse"]), 4),
        "citation_support": round(sum(r["citation_supported"] for r in rows) / max(1, n_answered), 4),
        "unsupported_claim": round(sum(r["unsupported_claim"] for r in rows) / max(1, n_answered), 4),
        "claim_distribution": dict(claim_counts),
        "conditional_divergence_pass": f"{cond_pass}/{len(cond_rows)}",
    }
    return metrics, rows


def _claim_classify(c, res, pred):
    if pred != "answer":
        return "NO_CLAIM"
    cited = _CITE_RE.findall(res.get("answer", ""))
    if not cited:
        return "NO_CITATION"
    gold = norm(c.get("gold_evidence_text", ""))
    if not gold:
        return "NO_GOLD"
    contexts = res.get("contexts", [])
    for sid in cited:
        idx = int(sid) - 1
        if 0 <= idx < len(contexts) and gold in norm(contexts[idx].get("text", "")):
            return "SUPPORTED"
    return "UNSUPPORTED"


def _conditional_ok(c, res, pred):
    if pred not in SUBSTANTIVE:
        return False
    answer = res.get("answer", "")
    cited = _CITE_RE.findall(answer)
    if not cited:
        return False
    marks = CONDITIONAL_CHECKS.get(c["case_id"], [])
    return all(m in answer for m in marks)


def evaluate_gate(m):
    results = []
    for label, key, thresh, op in GATE:
        val = m[key]
        ok = (val >= thresh) if op == ">=" else (val <= thresh)
        results.append((label, key, val, thresh, op, ok))
    return results


def verdict(results):
    core_keys = {"recall10", "page_hit10", "span_hit10", "action_accuracy",
                 "answerable_accuracy", "no_evidence_recall"}
    trust_keys = {"citation_support", "unsupported_claim"}
    fails = [r for r in results if not r[5]]
    if not fails:
        return "FINAL_ACCEPT"
    if all(r[1] in trust_keys for r in fails) and all(r[5] for r in results if r[1] in core_keys):
        return "FINAL_ACCEPT_WITH_LIMITATIONS"
    return "FINAL_NOT_ACCEPTED"


def main():
    ret_m, ret_rows = retrieval_metrics()
    beh_m, beh_rows = behavior_metrics()
    gate_rows = evaluate_gate(beh_m | ret_m)
    v = verdict(gate_rows)

    # write artifacts
    with (ACC / "acceptance_retrieval_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(ret_rows[0].keys())); w.writeheader(); w.writerows(ret_rows)
    with (ACC / "acceptance_behavior_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(beh_rows[0].keys())); w.writeheader(); w.writerows(beh_rows)

    lines = ["# Final Acceptance Report", "",
             f"- verdict: **{v}**", "",
             "## Release Gate", "",
             "| metric | value | threshold | op | pass |", "|---|---|---|---|---|"]
    for label, key, val, thresh, op, ok in gate_rows:
        lines.append(f"| {label} | {val} | {thresh} | {op} | {'✅' if ok else '❌'} |")
    lines += ["", "## Retrieval", "", f"`{json.dumps(ret_m, ensure_ascii=False)}`", "",
              "## Behavior", "", f"`{json.dumps(beh_m, ensure_ascii=False, indent=2)}`", ""]
    (ACC / "acceptance_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("VERDICT:", v)
    print(json.dumps({"retrieval": ret_m, "behavior": beh_m, "gate": [
        {"metric": label, "value": val, "threshold": thresh, "op": op, "pass": ok}
        for label, key, val, thresh, op, ok in gate_rows]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
