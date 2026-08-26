"""Route-only regression (POSTMORTEM/REGRESSION ONLY, no LLM, no held-out claim).

Computes the deterministic action for all 56 V2 cases with the frozen retrieval
(section_hybrid) and confirms PRESENT_CONFLICT no longer appears in the route.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_eval_v2 import raw_retrieve, set_index  # noqa: E402
from src.answerability import Action, classify_action  # noqa: E402

EVAL = ROOT / "data" / "eval_v2"


def main():
    cases = [json.loads(l) for l in (EVAL / "eval_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    set_index("section_aware_270_gpu", "hybrid_dense_sparse")

    dist = Counter()
    correct = 0
    for c in cases:
        hits = raw_retrieve(c["query"], 10, "hybrid_dense_sparse")
        from src.rag_chain import _evidence_status
        ev = _evidence_status(c["query"], hits)
        best = min((float(h.get("distance", 999.0)) for h in hits), default=999.0)
        action, _ = classify_action(c["query"], hits, ev, best)
        dist[action.value] += 1
        if action.value == c["expected_action"]:
            correct += 1

    print("action distribution:", dict(dist))
    print("PRESENT_CONFLICT count:", dist.get("present_conflict", 0))
    print("route-level action accuracy (deterministic, no LLM):", round(correct / len(cases), 4))


if __name__ == "__main__":
    main()
