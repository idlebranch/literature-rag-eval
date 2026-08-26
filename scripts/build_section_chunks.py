"""Section-aware chunking dry run over final_corpus (no embedding, no Chroma).

Reads final_paper_manifest.csv for title/doi, extracts line-preserving page text
from every PDF in data/papers/final_corpus, chunks with the section-aware chunker
(falling back to page-aware fixed chunking when no section is detected), and
writes:

    data/processed/section_chunks.jsonl
    data/processed/section_chunking_report.md
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunking import build_chunks_section_aware, _clean_pages
from src.config import settings
from src.pdf_loader import load_pdf_pages
FINAL_CORPUS = ROOT / "data" / "papers" / "final_corpus"
MANIFEST = ROOT / "data" / "papers" / "final_paper_manifest.csv"
OUT_DIR = ROOT / "data" / "processed"
JSONL = OUT_DIR / "section_chunks.jsonl"
REPORT = OUT_DIR / "section_chunking_report.md"


def load_meta() -> dict:
    meta: dict = {}
    if not MANIFEST.exists():
        return meta
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ff = row["final_file"]
            info = {"title": (row["title"] or ""), "doi": (row["doi"] or "")}
            meta[ff] = info
            meta[Path(ff).stem] = info
    return meta


def main() -> None:
    meta = load_meta()
    pdfs = sorted(FINAL_CORPUS.glob("*.pdf"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_total = len(pdfs)
    n_parsed = 0
    n_fallback = 0
    n_failed = 0
    failed_names: list[str] = []

    chunks_total = 0
    section_aware_chunks = 0
    fallback_chunks = 0
    empty_chunks = 0
    page_meta_missing = 0
    section_dist: Counter = Counter()

    total_raw_chars = 0
    total_clean_chars = 0
    total_chunk_chars = 0

    low_coverage: list[tuple[str, float]] = []

    with JSONL.open("w", encoding="utf-8") as fh:
        for pdf in pdfs:
            try:
                records = load_pdf_pages(str(pdf))
            except Exception as e:  # cannot open / parse the PDF at all
                n_failed += 1
                failed_names.append(f"{pdf.name}: {e}")
                continue

            if not records or all(not r["text"].strip() for r in records):
                n_failed += 1
                failed_names.append(f"{pdf.name}: no extractable text")
                continue

            chunks = build_chunks_section_aware(records, paper_meta=meta)
            if not chunks:
                n_failed += 1
                failed_names.append(f"{pdf.name}: no chunks produced")
                continue

            n_parsed += 1
            is_fallback = all(c["metadata"]["chunking_mode"] == "fallback_fixed" for c in chunks)
            if is_fallback:
                n_fallback += 1

            raw_chars = sum(len(r["text"]) for r in records)
            clean_chars = sum(len(t) for _p, t in _clean_pages([(r["page"], r["text"]) for r in records]))
            chunk_chars = sum(len(c["text"]) for c in chunks)
            total_raw_chars += raw_chars
            total_clean_chars += clean_chars
            total_chunk_chars += chunk_chars

            cov = chunk_chars / clean_chars if clean_chars else 0.0
            if cov < 0.5:
                low_coverage.append((pdf.name, round(cov, 3)))

            for c in chunks:
                m = c["metadata"]
                chunks_total += 1
                if m["chunking_mode"] == "fallback_fixed":
                    fallback_chunks += 1
                else:
                    section_aware_chunks += 1
                if not c["text"].strip():
                    empty_chunks += 1
                if not m.get("page_start") or not m.get("page_end"):
                    page_meta_missing += 1
                section_dist[m["section"]] += 1

                fh.write(json.dumps({
                    "chunk_id": c["id"],
                    "paper_id": m["paper_id"],
                    "title": m["title"],
                    "doi": m["doi"],
                    "section": m["section"],
                    "page_start": m["page_start"],
                    "page_end": m["page_end"],
                    "chunk_index": m["chunk_index"],
                    "source": m["source"],
                    "chunking_mode": m["chunking_mode"],
                    "text": c["text"],
                }, ensure_ascii=False) + "\n")

    avg = round(chunks_total / n_parsed, 2) if n_parsed else 0.0
    cov_clean = round(total_chunk_chars / total_clean_chars, 3) if total_clean_chars else 0.0

    section_lines = "\n".join(
        f"- `{name}`: {count}" for name, count in section_dist.most_common()
    )

    report = f"""# Section-aware Chunking Dry Run Report

Source: `{FINAL_CORPUS}` (frozen final corpus)
Mode: `section_aware` (fallback = page-aware fixed chunking)
chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}, min_chunk={settings.section_min_chunk}

## Totals

- PDFs total: **{n_total}**
- PDFs successfully parsed: **{n_parsed}**
- fallback PDFs (no section detected): **{n_fallback}**
- failed PDFs: **{n_failed}**
- chunks total: **{chunks_total}**
- average chunks / paper: **{avg}**
- section-aware chunks: **{section_aware_chunks}**
- fallback chunks: **{fallback_chunks}**
- empty chunks: **{empty_chunks}**
- chunks missing page metadata: **{page_meta_missing}**

## Text coverage

- raw extracted chars: **{total_raw_chars}**
- cleaned body chars (header/footer + whitespace removed): **{total_clean_chars}**
- chunk chars (includes overlap): **{total_chunk_chars}**
- coverage (chunk chars / cleaned chars): **{cov_clean}**

Coverage is measured against the *cleaned* body text (after running-header/footer
removal and whitespace normalisation). Values in the ~0.85–1.15 range are normal:
overlap pushes the ratio slightly up, while heading-line exclusion and header/footer
cleaning push it slightly down. A value far below 0.8 would indicate silent body
loss — see the low-coverage list below.

- papers with coverage < 0.5: **{len(low_coverage)}**
{chr(10).join(f"- {name}: {cov}" for name, cov in low_coverage) or "- (none)"}

## Section distribution

{section_lines}

## Failed PDFs

{chr(10).join("- " + n for n in failed_names) or "- (none)"}
"""
    REPORT.write_text(report, encoding="utf-8")

    print(f"PDFs total={n_total} parsed={n_parsed} fallback={n_fallback} failed={n_failed}")
    print(f"chunks total={chunks_total} avg={avg} section_aware={section_aware_chunks} fallback={fallback_chunks}")
    print(f"empty={empty_chunks} page_meta_missing={page_meta_missing}")
    print(f"coverage(clean)={cov_clean} low_coverage={len(low_coverage)}")
    print(f"wrote {JSONL}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
