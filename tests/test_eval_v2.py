"""Deterministic tests for the Eval V2 dataset, validation rules, and metrics."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_eval_v2  # noqa: E402
import validate_eval_v2  # noqa: E402

EVAL = ROOT / "data" / "eval_v2" / "eval_v2.jsonl"
CHUNKS = ROOT / "data" / "processed" / "section_chunks.jsonl"


def _cases():
    return [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]


def _rows():
    return [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_case_ids_unique_and_splits_valid():
    cases = _cases()
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        assert c["split"] in ("dev", "test")


def test_dev_test_no_overlap():
    cases = _cases()
    dev = {c["case_id"] for c in cases if c["split"] == "dev"}
    test = {c["case_id"] for c in cases if c["split"] == "test"}
    assert not (dev & test)


def test_class_action_mapping():
    for c in _cases():
        assert validate_eval_v2.CLASS_ACTION[c["answerability_class"]] == c["expected_action"]


def test_no_gold_for_non_answerable():
    for c in _cases():
        if c["answerability_class"] in ("AMBIGUOUS", "NO_EVIDENCE"):
            assert not c["gold_evidence_text"]
            assert not c["gold_paper_id"]


def test_gold_evidence_in_chunks():
    by_pid = {}
    for r in _rows():
        by_pid.setdefault(r["paper_id"], []).append(r)
    for c in _cases():
        if not c["gold_evidence_text"]:
            continue
        found = any(
            r["page_start"] <= c["gold_page_start"] <= r["page_end"]
            and c["gold_evidence_text"] in r["text"]
            for r in by_pid.get(c["gold_paper_id"], [])
        )
        assert found, f"{c['case_id']}: evidence not in chunk for paper/page"


def test_paired_relations_complete():
    cases = _cases()
    ids = {c["case_id"] for c in cases}
    for c in cases:
        if c.get("paired_with"):
            assert c["paired_with"] in ids, f"{c['case_id']} paired_with missing"
            pair = next(x for x in cases if x["case_id"] == c["paired_with"])
            assert pair["answerability_class"] == "ANSWERABLE"


def test_invalid_paper_anchor_rejected():
    rows = _rows()
    span, rec = build_eval_v2.locate_evidence(
        rows, {"gold_anchor": "zzz this anchor does not exist anywhere", "gold_page_start": 1})
    assert span is None and rec is None


def test_retrieval_paper_and_page_hit():
    from run_eval_v2 import page_hit, paper_hit
    hits = [{"metadata": {"paper_id": "p1", "page_start": 3, "page_end": 3}},
            {"metadata": {"paper_id": "p2", "page_start": 5, "page_end": 6}}]
    assert paper_hit(hits, "p1") == ["p1", "p2"]
    assert page_hit(hits, "p1", 3) is True
    assert page_hit(hits, "p1", 4) is False
    assert page_hit(hits, "p9", 3) is False


def test_unsupported_citation_identified():
    from src.citation_validation import validate_citations
    ans, val = validate_citations(
        "This result is supported by evidence. [S9]",
        [{"text": "only one context", "metadata": {"page": 1}}])
    assert val["status"] in ("corrected", "failed")
    assert "S9" in val["invalid_source_ids"]


def test_answerability_class_enum():
    for c in _cases():
        assert c["answerability_class"] in validate_eval_v2.CLASS_ACTION
