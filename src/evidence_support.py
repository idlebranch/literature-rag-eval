"""Deterministic claim-level evidence support (Phase E).

Decides whether retrieved chunks actually support the query's *specific* claim
rather than merely being topic-related. No gold, no case_id, no LLM judge, no
Eval V2 leakage.

Support status:
    SUPPORTED    — evidence has the requested number/relation.
    PARTIAL      — evidence covers only part of a comparison/exhaustive request.
    UNSUPPORTED  — topic-related but missing the target value/relation.
    CONFLICTING  — opposite conclusions across sources.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONFLICTING = "conflicting"


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# canonical metric -> regex for the metric mention in a chunk
_METRICS = [
    ("pH", re.compile(r"\bph\b", re.IGNORECASE)),
    ("removal", re.compile(r"(removal|degradation|mineralization)\s*(rate|efficiency)?|去除率|降解率|矿化率", re.IGNORECASE)),
    ("dosage", re.compile(r"(dosage|dose|投加量|剂量)", re.IGNORECASE)),
    ("capacity", re.compile(r"(adsorption\s*capacity|uptake\s*capacity|吸附容量)", re.IGNORECASE)),
    ("concentration", re.compile(r"(concentration|浓度)", re.IGNORECASE)),
    ("time", re.compile(r"(reaction\s*time|contact\s*time|反应时间|接触时间)", re.IGNORECASE)),
    ("flux", re.compile(r"\bflux\b|通量", re.IGNORECASE)),
    ("rate", re.compile(r"\b(rate|constant)\b|速率|常数", re.IGNORECASE)),
]

# question patterns that ask for a specific numeric value
_VALUE_REQUEST_RE = re.compile(
    r"(是多少|是什么|多少|what is|what's|how much|how many|最佳|最优|optimum|optimal|maximum|minimum|最高|最低)",
    re.IGNORECASE,
)
_VERIFY_RE = re.compile(r"(是不是|是否|是不是证明|是否证明|does .{0,50}(?:prove|show|report|reach)|is it (?:true|correct))", re.IGNORECASE)
_BEST_WHICH_RE = re.compile(r"(哪种|哪个|哪一种|哪些|which .{0,30}(?:is|are) (?:the )?(?:best|better|optimal)|what is the best)", re.IGNORECASE)


def _metric_re(metric: str):
    for name, rx in _METRICS:
        if name == metric:
            return rx
    return None


def metric_of(question: str) -> str | None:
    for metric, rx in _METRICS:
        if rx.search(question):
            return metric
    return None


def metric_number_in(metric: str, text: str) -> bool:
    """True if ``text`` has a number within a short window of the metric term."""
    rx = _metric_re(metric)
    if rx is None:
        return False
    for m in rx.finditer(text):
        window = text[max(0, m.start() - 70): m.end() + 70]
        if _NUM_RE.search(window):
            return True
    return False


def evaluate_support(question: str, hits: List[dict]) -> tuple[SupportStatus, str]:
    """Claim-level support check over retrieved chunks."""
    if not hits:
        return SupportStatus.UNSUPPORTED, "no_retrieved_evidence"

    metric = metric_of(question)

    # numeric value request: need a number near the metric in at least one chunk
    if metric and _VALUE_REQUEST_RE.search(question):
        supported = any(metric_number_in(metric, str(h.get("text", ""))) for h in hits)
        if supported:
            return SupportStatus.SUPPORTED, f"{metric}_value_present"
        return SupportStatus.UNSUPPORTED, f"no_{metric}_value_in_evidence"

    # "which is best" without a comparison target -> cannot claim global best
    if _BEST_WHICH_RE.search(question):
        return SupportStatus.PARTIAL, "no_comparison_scope"

    return SupportStatus.SUPPORTED, ""


def contradicting_number(question: str, hits: List[dict]) -> bool:
    """True when a verification question claims a number that the evidence
    contradicts. Metric-aware when the query names a metric; otherwise falls back
    to "claimed number is absent from the evidence"."""
    if not _VERIFY_RE.search(question):
        return False
    qnums = _NUM_RE.findall(question)
    if not qnums:
        return False
    metric = metric_of(question)
    if metric:
        rx = _metric_re(metric)
        if rx is None:
            return False
        for h in hits:
            text = str(h.get("text", ""))
            for m in rx.finditer(text):
                window = text[max(0, m.start() - 70): m.end() + 70]
                for n in _NUM_RE.findall(window):
                    if n not in qnums:
                        return True
        return False
    # no metric named: loose fallback — the claimed number never appears anywhere
    evnums: set[str] = set()
    for h in hits:
        evnums.update(_NUM_RE.findall(str(h.get("text", ""))))
    return bool(evnums) and any(n not in evnums for n in qnums)
