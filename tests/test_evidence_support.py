"""Tests for claim-level evidence support (Phase E)."""
import pytest

from src import evidence_support
from src.answerability import Action, classify_action


def _hits(texts):
    return [{"text": t, "metadata": {"paper_id": f"p{i}", "page_start": 1, "page_end": 1}}
            for i, t in enumerate(texts)]


def _cls(question, texts, status="available", distance=0.3):
    return classify_action(question, _hits(texts), status, distance)


# ---------------------------------------------------------------- claim support

def test_specific_numeric_value_supported():
    s, _ = evidence_support.evaluate_support(
        "活性炭对铜的吸附容量是多少？", _hits(["adsorption capacity of 46.3 mg/g for copper"]))
    assert s == evidence_support.SupportStatus.SUPPORTED


def test_topic_related_but_no_target_value_unsupported():
    # topic-related text with no number near the requested metric
    s, _ = evidence_support.evaluate_support(
        "最佳 pH 是多少？", _hits(["the process depends on water matrix but no quantitative pH is reported"]))
    assert s == evidence_support.SupportStatus.UNSUPPORTED


def test_near_evidence_not_mistaken_for_gold_claim():
    # a chunk that is topic-adjacent but lacks the target value must not ANSWER
    a, _ = _cls("吸附容量是多少？", ["吸附受到水温和基质影响，但未报告具体容量数值"])
    assert a != Action.ANSWER


def test_best_without_comparison_is_partial():
    s, _ = evidence_support.evaluate_support(
        "which method is best?", _hits(["one method is described here"]))
    assert s == evidence_support.SupportStatus.PARTIAL


def test_false_premise_with_direct_contradiction():
    a, _ = _cls("最佳 pH 是不是 10？", ["maximum removal was obtained at pH 6.9"])
    assert a == Action.CORRECT_PREMISE


def test_false_premise_without_contradiction_not_guessed():
    # evidence says nothing about the claimed number -> do NOT assert a correction
    a, _ = _cls("最佳 pH 是不是 10？", ["the process is described qualitatively"])
    assert a != Action.CORRECT_PREMISE


def test_partial_evidence_routes_partial_answer():
    a, _ = _cls("请给出所有膜工艺的完整成本对比。", ["one membrane method cost data"])
    assert a == Action.PARTIAL_ANSWER


def test_conflicting_evidence_routes_present_conflict():
    a, _ = _cls("does X increase Y?", ["X increases Y", "X decreases Y"], status="conflicting")
    assert a == Action.PRESENT_CONFLICT


def test_citation_chunk_mismatch_unsupported():
    # numeric value request whose evidence has the metric but no number nearby
    s, _ = evidence_support.evaluate_support(
        "removal efficiency是多少？", _hits(["removal efficiency was investigated", "another topic"]))
    assert s == evidence_support.SupportStatus.UNSUPPORTED


def test_best_ph_value_request_is_answerable_not_ambiguous():
    # "最佳 pH 是多少" asks for a value, not "which is best"
    a, _ = _cls("电絮凝的最佳 pH 是多少？", ["an optimum pH of 6.9 was found for color removal"])
    assert a == Action.ANSWER


def test_which_is_best_is_ambiguous():
    a, _ = _cls("哪种高级氧化工艺最好？", ["various AOP text"])
    assert a == Action.CLARIFY
