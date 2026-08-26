"""Build an inventory of the existing RAG corpus (data/pdfs/**/*.pdf).

For each PDF extract: filename, title (from metadata + first page), DOI,
year, language heuristic, sha256, page count, extracted-char count. This is
the dedup baseline for Phase 1. No download, no index modification.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF_DIRS = [ROOT / "data" / "pdfs"]

DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>;,)\]，。；]+", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_inventory(path: Path) -> dict:
    rec = {
        "file": str(path.relative_to(ROOT)),
        "filename": path.name,
        "ok": False,
        "title_meta": "",
        "doi": "",
        "year": "",
        "pages": 0,
        "chars": 0,
        "first_page_preview": "",
        "sha256": "",
        "size_kb": round(path.stat().st_size / 1024, 1),
    }
    try:
        rec["sha256"] = sha256_of(path)
        with fitz.open(path) as doc:
            rec["pages"] = doc.page_count
            meta = doc.metadata or {}
            rec["title_meta"] = norm_text(str(meta.get("title", "") or ""))
            first = doc[0].get_text() if doc.page_count else ""
            rec["chars"] = sum(len(p.get_text()) for p in doc)
            rec["first_page_preview"] = norm_text(first)[:400]
            # DOI: prefer metadata, then full first two pages text
            text_scan = first
            if doc.page_count > 1:
                text_scan += "\n" + doc[1].get_text()
            m = DOI_RE.search(str(meta.get("subject", "")) + " " + str(meta.get("keywords", "")))
            if not m:
                m = DOI_RE.search(text_scan)
            if m:
                rec["doi"] = m.group(0).rstrip(".,;")
            # Year: metadata creationDate, else from text
            cd = str(meta.get("creationDate", ""))
            my = YEAR_RE.search(cd)
            if my:
                rec["year"] = my.group(1)
            else:
                ym = YEAR_RE.findall(text_scan[:3000])
                if ym:
                    rec["year"] = max(set(ym), key=ym.count)
            rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main() -> None:
    files = []
    for d in PDF_DIRS:
        files.extend(sorted(d.rglob("*.pdf")))
    recs = [extract_inventory(p) for p in files]
    out = ROOT / "data" / "_inventory_existing.json"
    out.write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in recs if r["ok"])
    print(f"total={len(recs)} ok={ok} written={out}")
    for r in recs:
        if not r["ok"]:
            print("BROKEN:", r["filename"], r.get("error"))


if __name__ == "__main__":
    main()
