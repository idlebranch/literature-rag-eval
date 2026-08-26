"""Validate the Eval V2 dataset against the frozen corpus.

Hard rules:
  - Every case carrying gold_evidence_text must re-locate in the actual PDF page
    (re-extracted via PyMuPDF, normalized comparison — never semantic).
  - AMBIGUOUS / NO_EVIDENCE cases must NOT carry fake gold evidence.
  - answerability_class / expected_action must be a valid pairing.
  - splits must be dev/test only and never overlap.
  - paired_with (when present) must reference an existing case.

Writes data/eval_v2/eval_v2_validation_report.md and exits non-zero on any
violation.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval_v2" / "eval_v2.jsonl"
CORPUS = ROOT / "data" / "papers" / "final_corpus"

CLASS_ACTION = {
    "ANSWERABLE": "answer",
    "AMBIGUOUS": "clarify",
    "NO_EVIDENCE": "refuse",
    "PARTIAL_EVIDENCE": "partial_answer",
    "FALSE_PREMISE": "correct_premise",
    "CONFLICTING_EVIDENCE": "present_conflict",
}
NO_GOLD = {"AMBIGUOUS", "NO_EVIDENCE"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    for lig, ascii_ in (("\ufb00", "ff"), ("\ufb01", "fi"), ("\ufb02", "fl"),
                        ("\ufb03", "ffi"), ("\ufb04", "ffl")):
        s = s.replace(lig, ascii_)
    for a, b in (("\u2212", "-"), ("\u2013", "-"), ("\u2014", "-"), ("\u03bc", "u")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).casefold()


def pdf_pages(paper_id: str, page_start: int, page_end: int) -> str:
    import fitz
    path = CORPUS / f"{paper_id}.pdf"
    if not path.exists():
        # try manifest fallback for any collision-suffixed filename
        raise FileNotFoundError(f"PDF not found for {paper_id!r}")
    doc = fitz.open(str(path))
    parts = []
    try:
        for p in range(page_start, page_end + 1):
            if 1 <= p <= doc.page_count:
                parts.append(doc.load_page(p - 1).get_text("text") or "")
    finally:
        doc.close()
    return "\n".join(parts)


def main() -> int:
    cases = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    errors, warnings = [], []
    ids = [c["case_id"] for c in cases]
    id_set = set(ids)

    # duplicate ids / split overlap
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        errors.append(f"duplicate case_id: {dup}")

    for c in cases:
        cid = c["case_id"]
        cls = c.get("answerability_class")
        act = c.get("expected_action")
        split = c.get("split")
        if cls not in CLASS_ACTION:
            errors.append(f"{cid}: bad answerability_class {cls!r}")
            continue
        if CLASS_ACTION[cls] != act:
            errors.append(f"{cid}: expected_action {act!r} != {CLASS_ACTION[cls]!r} for {cls}")
        if split not in ("dev", "test"):
            errors.append(f"{cid}: bad split {split!r}")
        if c.get("paired_with") and c["paired_with"] not in id_set:
            errors.append(f"{cid}: paired_with {c['paired_with']!r} not found")

        ev = c.get("gold_evidence_text") or ""
        pid = c.get("gold_paper_id") or ""
        if cls in NO_GOLD:
            if ev or pid:
                errors.append(f"{cid}: {cls} must not carry gold evidence/paper_id")
            continue
        # evidence classes: must have paper + page + evidence and re-locate
        if not ev or not pid or c.get("gold_page_start") is None:
            errors.append(f"{cid}: missing gold fields")
            continue
        try:
            raw = pdf_pages(pid, c["gold_page_start"], c["gold_page_end"])
        except FileNotFoundError as e:
            errors.append(f"{cid}: {e}")
            continue
        if norm(ev) not in norm(raw):
            errors.append(f"{cid}: gold_evidence_text not found in PDF pages "
                          f"{c['gold_page_start']}-{c['gold_page_end']}")

    report = ["# Eval V2 Validation Report", "",
              f"- total cases: **{len(cases)}**",
              f"- dev: **{sum(1 for c in cases if c['split']=='dev')}**",
              f"- test: **{sum(1 for c in cases if c['split']=='test')}**",
              f"- class distribution: {dict(Counter(c['answerability_class'] for c in cases))}",
              f"- errors: **{len(errors)}**",
              f"- warnings: **{len(warnings)}**", ""]
    if errors:
        report += ["## Errors"] + [f"- {e}" for e in errors]
    if warnings:
        report += ["", "## Warnings"] + [f"- {w}" for w in warnings]
    if not errors:
        report += ["", "## Result", "VALIDATION PASSED"]
    (ROOT / "data" / "eval_v2" / "eval_v2_validation_report.md").write_text(
        "\n".join(report), encoding="utf-8")

    print(f"validated {len(cases)} cases: {len(errors)} errors, {len(warnings)} warnings")
    for e in errors:
        print("  ERROR:", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
