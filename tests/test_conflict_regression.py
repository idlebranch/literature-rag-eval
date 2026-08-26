"""Regression tests: automatic conflict detection is removed (Final Code Freeze).

No case-specific logic; these are generic scenarios that the removed keyword
heuristic used to mis-classify as "conflicting".
"""
import pytest

from src import rag_chain
from src.answerability import Action, classify_action


def _hits(texts):
    return [{"text": t, "metadata": {"paper_id": f"p{i}", "page_start": 1, "page_end": 1}}
            for i, t in enumerate(texts)]


@pytest.mark.parametrize("texts", [
    # same paper: rise then fall (non-monotonic / threshold effect)
    ["removal rate increases with dose", "removal rate decreases at very high dose"],
    # different metrics: one up, one down (trade-off, not conflict)
    ["removal rate increases", "energy cost decreases"],
    # same metric, different conditions (pH / dosage / time)
    ["at pH 3 the removal rate is 80%", "at pH 7 the removal rate is 40%"],
    # different papers, incomparable conditions
    ["removal increases at 20 C", "removal decreases at 60 C"],
])
def test_evidence_status_not_conflicting(texts):
    assert rag_chain._evidence_status("q", _hits(texts)) != "conflicting"


def test_opposite_phrases_do_not_auto_route_conflict():
    a, _ = classify_action(
        "does X increase removal?",
        _hits(["removal rate increases with dose", "removal rate decreases at high dose"]),
        evidence_status="available", best_distance=0.3)
    assert a != Action.PRESENT_CONFLICT


def test_refuse_clarify_correct_premise_still_work():
    # no evidence -> refuse
    a, _ = classify_action("q", [], "available", 0.3)
    assert a == Action.REFUSE
    # ambiguous -> clarify
    a, _ = classify_action("哪种工艺最好？", _hits(["some text"]), "available", 0.3)
    assert a == Action.CLARIFY
    # false premise -> correct premise
    a, _ = classify_action("最佳 pH 是不是 10？", _hits(["best pH is 3.0"]), "available", 0.3)
    assert a == Action.CORRECT_PREMISE


def test_no_evidence_not_regressed():
    a, _ = classify_action("q", _hits(["topic related but no answer"]), "insufficient", 0.3)
    assert a == Action.REFUSE
