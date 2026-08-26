"""Conflict auto-routing ablation over the full V2 set (56 cases, postmortem).

A. CURRENT     — existing conflict auto-routing.
B. CONFLICT_OFF — runtime monkeypatch: _evidence_status never returns "conflicting",
                 so PRESENT_CONFLICT is never auto-routed. Everything else unchanged.

This is a regression/postmortem comparison, NOT a held-out TEST result.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_eval_v2 import set_index  # noqa: E402
from validate_eval_v2 import norm  # noqa: E402

EVAL = ROOT / "data" / "eval_v2"
ALL = EVAL / "eval_v2.jsonl"


def load():
    return [json.loads(l) for l in ALL.read_text(encoding="utf-8").splitlines() if l.strip()]


def evaluate(cases, conflict_off):
    import src.rag_chain as rc
    from src.rag_chain import answer_question

    orig = rc._evidence_status
    if conflict_off:
        def no_conflict(q, hits):
            s = orig(q, hits)
            return "available" if s == "conflicting" else s
        rc._evidence_status = no_conflict
    else:
        rc._evidence_status = orig

    rows = []
    class_n = Counter(); class_acc = Counter()
    tp = {"clarify": 0, "refuse": 0}; fp = {"clarify": 0, "refuse": 0}
    n_answered = 0; n_cited_ok = 0; n_unsupported = 0

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

        cv = res.get("citation_validation", {})
        if pred == "answer":
            n_answered += 1
            if cv.get("status") == "passed":
                n_cited_ok += 1
            elif cv.get("status") == "failed" or not cv.get("used_source_ids"):
                n_unsupported += 1

        rows.append({"case_id": c["case_id"], "class": cls,
                     "expected": c["expected_action"], "predicted": pred,
                     "correct": int(correct)})

    rc._evidence_status = orig  # restore
    n = len(cases)
    return {
        "n": n,
        "action_accuracy": round(sum(r["correct"] for r in rows) / n, 4),
        "per_class": {k: round(class_acc[k] / class_n[k], 4) for k in sorted(class_n)},
        "clarify_precision": round(tp["clarify"] / max(1, fp["clarify"]), 4),
        "clarify_recall": round(tp["clarify"] / max(1, class_n["AMBIGUOUS"]), 4),
        "refuse_precision": round(tp["refuse"] / max(1, fp["refuse"]), 4),
        "refuse_recall": round(tp["refuse"] / max(1, class_n["NO_EVIDENCE"]), 4),
        "false_premise_correction": round(
            sum(1 for r in rows if r["class"] == "FALSE_PREMISE" and r["predicted"] == "correct_premise")
            / max(1, class_n["FALSE_PREMISE"]), 4),
        "unsupported_claim_rate": round(n_unsupported / max(1, n_answered), 4),
        "citation_support_rate": round(n_cited_ok / max(1, n_answered), 4),
        "rows": rows,
    }


def main():
    cases = load()
    set_index("section_aware_270_gpu", "hybrid_dense_sparse")
    cur = evaluate(cases, conflict_off=False)
    off = evaluate(cases, conflict_off=True)

    # case-level win/loss
    cur_map = {r["case_id"]: r["correct"] for r in cur["rows"]}
    off_map = {r["case_id"]: r["correct"] for r in off["rows"]}
    wins = sum(1 for cid in cur_map if off_map[cid] > cur_map[cid])      # CONFLICT_OFF gains
    losses = sum(1 for cid in cur_map if off_map[cid] < cur_map[cid])    # CONFLICT_OFF loses
    unchanged = sum(1 for cid in cur_map if off_map[cid] == cur_map[cid])

    # rescued ANSWERABLE / lost CONFLICTING
    rescued = [r["case_id"] for r in cur["rows"]
               if r["class"] == "ANSWERABLE" and r["correct"] == 0
               and off_map[r["case_id"]] == 1]
    lost_conflict = [r["case_id"] for r in cur["rows"]
                     if r["class"] == "CONFLICTING_EVIDENCE" and r["correct"] == 1
                     and off_map[r["case_id"]] == 0]

    print("=== CURRENT ===")
    print(json.dumps({k: v for k, v in cur.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print("=== CONFLICT_OFF ===")
    print(json.dumps({k: v for k, v in off.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print(f"wins={wins} losses={losses} unchanged={unchanged}")
    print(f"rescued ANSWERABLE={len(rescued)} {rescued}")
    print(f"lost CONFLICTING={len(lost_conflict)} {lost_conflict}")


if __name__ == "__main__":
    main()
