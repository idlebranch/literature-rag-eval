"""Tests for the deterministic answerability action classifier (Phase D P0-B)."""
import pytest

from src.answerability import (
    Action,
    classify_action,
    detect_false_premise,
    is_ambiguous,
    is_partial,
)

# synthetic evidence hits (generic, no Eval V2 case_ids / gold labels)
def _hits(texts, distance=0.3):
    return [{"text": t, "metadata": {"paper_id": f"p{i}", "page_start": 1, "page_end": 1}}
            for i, t in enumerate(texts)]


def _classify(question, texts, status="available", distance=0.3):
    return classify_action(question, _hits(texts), status, distance)


# ---------------------------------------------------------------- action mapping

def test_answer_when_evidence_supports():
    a, _ = _classify(
        "What is the adsorption capacity of activated carbon for copper?",
        ["the activated carbon showed an adsorption capacity of 46.3 mg/g for copper"])
    assert a == Action.ANSWER


def test_refuse_when_no_evidence():
    a, _ = _classify("what is x?", [])
    assert a == Action.REFUSE


def test_refuse_when_distance_above_threshold(monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "max_retrieval_distance", 0.5)
    a, _ = classify_action("q", _hits(["some text"]), "available", best_distance=0.9)
    assert a == Action.REFUSE


def test_clarify_when_ambiguous_superlative():
    a, _ = _classify("哪种高级氧化工艺最好？", ["various AOP text"])
    assert a == Action.CLARIFY


def test_clarify_when_underspecified():
    a, _ = _classify("这个方法效果如何？", ["some method text"])
    assert a == Action.CLARIFY


def test_partial_when_exhaustive_request():
    a, _ = _classify(
        "请给出所有膜工艺的完整成本对比。",
        ["one membrane method cost data"])
    assert a == Action.PARTIAL_ANSWER


def test_correct_premise_when_numeric_claim_unsupported():
    a, _ = _classify(
        "电絮凝去除 Acid Green 50 的最佳 pH 是不是 10？",
        ["an optimum pH of 6.9 was found for color removal by EC"])
    assert a == Action.CORRECT_PREMISE


def test_present_conflict_when_conflicting_evidence():
    a, _ = _classify("does X increase Y?", ["X increases Y", "X decreases Y"],
                     status="conflicting")
    assert a == Action.PRESENT_CONFLICT


# ---------------------------------------------------------------- behavioral rules

def test_ambiguous_not_guessed():
    # a comparative question without target must not be routed to ANSWER
    assert not _classify("哪种膜最好？", ["membrane A flux", "membrane B flux"])[0] == Action.ANSWER


def test_partial_not_fully_refused():
    # exhaustive request with some evidence should not be REFUSE
    a, _ = _classify("给出所有膜工艺的完整对比", ["partial membrane data"])
    assert a == Action.PARTIAL_ANSWER


def test_false_premise_is_corrected_not_refused():
    a, _ = _classify("是不是最佳 pH 为 10？", ["best pH is 3.0"])
    assert a == Action.CORRECT_PREMISE


def test_conflict_not_single_conclusion():
    a, _ = _classify("q", ["A says higher", "B says lower"], status="conflicting")
    assert a == Action.PRESENT_CONFLICT


# ---------------------------------------------------------------- helpers

def test_detect_false_premise_positive_and_negative():
    assert detect_false_premise("是不是 95%？", _hits(["removal rate 64.4%"])) is True
    assert detect_false_premise("是不是 64.4%？", _hits(["removal rate 64.4%"])) is False
    assert detect_false_premise("去除率是多少？", _hits(["64.4%"])) is False  # not verification


def test_detect_false_premise_superlative():
    assert detect_false_premise("PMS 在 pH=9 时是不是最稳定？",
                                _hits(["Minimum stability of PMS is observed at pH 9"])) is True
    assert detect_false_premise("PMS 在 pH=9 时是不是最稳定？",
                                _hits(["PMS is stable at pH 9"])) is False


def test_is_ambiguous_and_partial():
    assert is_ambiguous("哪种工艺最好？") is True
    assert is_ambiguous("这个效果如何？") is True
    assert is_ambiguous("PFAS 去除率是多少？") is False
    assert is_partial("所有膜工艺的完整对比") is True
    assert is_partial("single membrane flux?") is False
