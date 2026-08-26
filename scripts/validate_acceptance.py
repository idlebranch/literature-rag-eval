"""Validate the Final Acceptance Set against the frozen corpus.

Hard rules:
  - gold_evidence_text must re-locate in the actual PDF page (normalized,
    never semantic).
  - AMBIGUOUS / NO_EVIDENCE cases must NOT carry fake gold evidence.
  - answerability_class / expected_action pairing must be valid.
  - every gold paper_id must be FRESH (never used as Eval V2 gold).
Writes data/acceptance/acceptance_validation_report.md; exit non-zero on error.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACC = ROOT / "data" / "acceptance" / "acceptance.jsonl"
EVAL_V2_DIR = ROOT / "data" / "eval_v2"
CORPUS = ROOT / "data" / "papers" / "final_corpus"

CLASS_ACTION = {
    "ANSWERABLE": "answer",
    "AMBIGUOUS": "clarify",
    "NO_EVIDENCE": "refuse",
    "PARTIAL_EVIDENCE": "partial_answer",
    "FALSE_PREMISE": "correct_premise",
    "CONDITIONALLY_DIVERGENT": "answer",
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


def used_eval_v2_paper_ids() -> set[str]:
    ids = set()
    for f in EVAL_V2_DIR.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            pid = json.loads(line).get("gold_paper_id")
            if pid:
                ids.add(pid)
    return ids


def pdf_pages(paper_id: str, page_start: int, page_end: int) -> str:
    import fitz
    path = CORPUS / f"{paper_id}.pdf"
    if not path.exists():
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
    cases = [json.loads(l) for l in ACC.read_text(encoding="utf-8").splitlines() if l.strip()]
    used = used_eval_v2_paper_ids()
    errors = []
    ids = [c["case_id"] for c in cases]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        errors.append(f"duplicate case_id: {dup}")

    for c in cases:
        cid = c["case_id"]
        cls = c.get("answerability_class")
        act = c.get("expected_action")
        if cls not in CLASS_ACTION:
            errors.append(f"{cid}: bad class {cls!r}"); continue
        if CLASS_ACTION[cls] != act:
            errors.append(f"{cid}: action {act!r} != {CLASS_ACTION[cls]!r}")
        ev = c.get("gold_evidence_text") or ""
        pid = c.get("gold_paper_id") or ""
        if cls in NO_GOLD:
            if ev or pid:
                errors.append(f"{cid}: {cls} must not carry gold evidence")
            continue
        if not ev or not pid or c.get("gold_page_start") is None:
            errors.append(f"{cid}: missing gold fields"); continue
        if pid in used:
            errors.append(f"{cid}: gold paper_id {pid!r} is NOT fresh (used in Eval V2)")
        try:
            raw = pdf_pages(pid, c["gold_page_start"], c["gold_page_end"])
        except FileNotFoundError as e:
            errors.append(f"{cid}: {e}"); continue
        if norm(ev) not in norm(raw):
            errors.append(f"{cid}: gold not found in PDF p{c['gold_page_start']}-{c['gold_page_end']}")

    report = ["# Acceptance Set Validation Report", "",
              f"- total cases: **{len(cases)}**",
              f"- class distribution: {dict(Counter(c['answerability_class'] for c in cases))}",
              f"- fresh paper_ids: **{len({c['gold_paper_id'] for c in cases if c.get('gold_paper_id')})}**",
              f"- errors: **{len(errors)}**", ""]
    if errors:
        report += ["## Errors"] + [f"- {e}" for e in errors]
    else:
        report += ["## Result", "VALIDATION PASSED"]
    (ROOT / "data" / "acceptance" / "acceptance_validation_report.md").write_text(
        "\n".join(report), encoding="utf-8")

    print(f"validated {len(cases)} cases: {len(errors)} errors")
    for e in errors:
        print("  ERROR:", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
