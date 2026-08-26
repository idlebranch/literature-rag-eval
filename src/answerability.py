"""Deterministic answerability classification (Phase D P0-B).

Maps (question, retrieved evidence) -> a single action without any gold label,
LLM call, or case_id awareness. The action is then used by the RAG chain to route
generation. Purely rule-based and unit-testable.

Action set (aligned with Eval V2):
    ANSWER / CLARIFY / REFUSE / PARTIAL_ANSWER / CORRECT_PREMISE / PRESENT_CONFLICT
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List

from src import evidence_support
from src.config import settings


class Action(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    REFUSE = "refuse"
    PARTIAL_ANSWER = "partial_answer"
    CORRECT_PREMISE = "correct_premise"
    PRESENT_CONFLICT = "present_conflict"


# --- signals -----------------------------------------------------------------

# verification-style question ("是不是 X", "是否证明 X", "does ... show", ...)
_VERIFY_RE = re.compile(
    r"(是不是|是否|是不是证明|是否证明|是否达到|是不是达到|does .{0,50}(?:prove|show|report|reach)|"
    r"is it (?:true|correct)|did .{0,40}(?:prove|show|report|reach))",
    re.IGNORECASE,
)
# superlative / comparative question asking WHICH is best (not "what is the best X")
_SUPERLATIVE_RE = re.compile(
    r"(哪种|哪个|哪一种|哪些|which .{0,30}(?:is|are) (?:the )?(?:best|better|optimal)|what is the best)",
    re.IGNORECASE,
)
# under-specified question asking for an effect/result without a metric
_UNDERSPECIFIED_RE = re.compile(
    r"(效果如何|效果怎么样|好不好|行不行|是否有效|管用吗|does it work|"
    r"is .{0,20} (?:effective|good|better|useful))",
    re.IGNORECASE,
)
# request for exhaustive / complete coverage
_EXHAUSTIVE_RE = re.compile(
    r"(所有|全部|完整|全面|完整对比|所有.{0,6}(?:工艺|方法|技术|材料)|"
    r"complete|comprehensive|all (?:of )?(?:the )?(?:aop|method|technology|material|process)|full comparison)",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# broad water-treatment domain vocabulary; a query with none of these is
# almost certainly out-of-scope for this corpus.
_DOMAIN_ANCHOR_RE = re.compile(
    r"(pfas|pfoa|pfos|aop|pms|pds|ozone|臭氧|photocatal|光催化|fenton|芬顿|membrane|膜|"
    r"adsorption|吸附|activated carbon|活性炭|catalyst|催化|pollutant|contaminant|污染物|"
    r"wastewater|废水|drinking water|饮用水|dom|nom|antibiotic|抗生素|bisphenol|双酚|"
    r"sludge|污泥|biological|生物|coagul|絮凝|混凝|disinfect|消毒|electro|电化学|电絮凝|"
    r"nitrogen|氮|phosphorus|磷|microplastic|微塑料|water treatment|水处理|oxidation|氧化|"
    r"高级氧化|工艺|再生|降解|去除|水质|污水)",
    re.IGNORECASE,
)


def is_out_of_scope(question: str) -> bool:
    return not _DOMAIN_ANCHOR_RE.search(question)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").casefold()


def _evidence_numbers(hits: List[dict]) -> set[str]:
    out: set[str] = set()
    for h in hits:
        out.update(_NUM_RE.findall(str(h.get("text", ""))))
    return out


_SUPERLATIVE_CLAIM_RE = re.compile(
    r"(最(?:稳定|好|高|强|有效|大|快|优)|most (?:stable|effective|efficient|optimal))",
    re.IGNORECASE,
)
_SUPERLATIVE_NEG_RE = re.compile(
    r"(minimum|least|lowest|poorest|最不|最低|最小|最差|稳定性最低)", re.IGNORECASE,
)


def detect_false_premise(question: str, hits: List[dict]) -> bool:
    """True when a verification question carries a numeric/superlative claim
    that the retrieved evidence contradicts (i.e. the premise is unsupported)."""
    if not _VERIFY_RE.search(question):
        return False
    # metric-aware numeric contradiction (avoids coincidental numbers elsewhere)
    if evidence_support.contradicting_number(question, hits):
        return True
    # superlative claim ("最稳定") contradicted by a negative marker in evidence
    if _SUPERLATIVE_CLAIM_RE.search(question):
        evtext = " ".join(str(h.get("text", "")) for h in hits)
        if _SUPERLATIVE_NEG_RE.search(evtext):
            return True
    return False


def is_ambiguous(question: str) -> bool:
    q = _norm(question)
    if _SUPERLATIVE_RE.search(q):
        return True
    if len(q) <= 30 and _UNDERSPECIFIED_RE.search(q):
        return True
    return False


def is_partial(question: str) -> bool:
    return bool(_EXHAUSTIVE_RE.search(_norm(question)))


def classify_action(
    question: str,
    hits: List[dict],
    evidence_status: str,
    best_distance: float,
) -> tuple[Action, str]:
    """Return (action, reason). Deterministic; no gold label, no case_id."""
    # evidence_status already folds no-hits / distance / exact-value-without-number
    # into "insufficient" (see rag_chain._prepare_retrieval).
    if evidence_status == "insufficient":
        return Action.REFUSE, "insufficient_evidence"
    # defensive re-checks for callers that do not pre-fold:
    if not hits:
        return Action.REFUSE, "no_retrieved_evidence"
    if best_distance > settings.max_retrieval_distance:
        return Action.REFUSE, "distance_above_threshold"
    # false premise takes precedence (a verification question whose claim the
    # evidence contradicts is a premise error).
    if detect_false_premise(question, hits):
        return Action.CORRECT_PREMISE, "verification_claim_unsupported_by_evidence"
    if is_ambiguous(question):
        return Action.CLARIFY, "ambiguous_or_underspecified_query"
    # NOTE: automatic PRESENT_CONFLICT routing has been removed. Reliable conflict
    # detection needs claim/condition alignment, which a keyword heuristic cannot
    # provide; genuine source/condition disagreements are left to the generation
    # prompt. Action.PRESENT_CONFLICT stays defined for API compatibility only.
    if is_partial(question):
        return Action.PARTIAL_ANSWER, "exhaustive_request_but_limited_evidence"
    # claim-level support: a numeric value request whose metric has no number in
    # the evidence is unsupported, not answerable.
    support, support_reason = evidence_support.evaluate_support(question, hits)
    if support == evidence_support.SupportStatus.UNSUPPORTED:
        return Action.REFUSE, f"claim_unsupported:{support_reason}"
    return Action.ANSWER, ""


# Chinese action instructions appended to the generation prompt (NOT the gold
# label — the action is derived from query+evidence above).
ACTION_INSTRUCTION = {
    # ANSWER keeps the general evidence rules (which already say "refuse if
    # evidence does not support"); no overriding "answer directly".
    Action.ANSWER: "请先核对检索证据是否真正支持该问题；能支持才回答并引用，不能支持则明确说明证据不足。",
    Action.CLARIFY: "该问题缺少关键限定，请向用户提出 1-2 个澄清问题，不要猜测作答。",
    Action.REFUSE: "请明确说明当前知识库证据不足，不得编造。",
    Action.PARTIAL_ANSWER: "只回答证据能支持的部分，并明确指出证据缺失的部分。",
    Action.CORRECT_PREMISE: "先指出问题中的错误前提并给出正确信息，再回答证据可支持的部分。",
    Action.PRESENT_CONFLICT: "分别呈现冲突来源的不同结论与适用条件，不得强行给唯一结论。",
}
