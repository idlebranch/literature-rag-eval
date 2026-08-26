"""Tests for section-aware + page-traceable chunking.

Covers: section detection, fallback, page traceability, chunk boundaries,
determinism, metadata completeness, and a real-PDF regression over final_corpus.
"""
import json
from pathlib import Path

import pytest

from src.chunking import (
    build_chunks,
    build_chunks_fixed,
    build_chunks_section_aware,
    clean_page_text,
    match_section_heading,
)
from src.config import settings

ROOT = Path(__file__).resolve().parent.parent
FINAL_CORPUS = ROOT / "data" / "papers" / "final_corpus"


def rec(paper_id, page, text, source=None):
    return {"paper_id": paper_id, "source": source or f"{paper_id}.pdf", "page": page, "text": text}


# ---------------------------------------------------------------- A. detection

@pytest.mark.parametrize("line,expected", [
    ("Abstract", "abstract"),
    ("ABSTRACT", "abstract"),
    ("Abstract:", "abstract"),
    ("Introduction", "introduction"),
    ("Methods", "methods"),
    ("Materials and Methods", "methods"),
    ("Materials and methods", "methods"),
    ("Experimental", "methods"),
    ("Results", "results"),
    ("Results and Discussion", "results"),
    ("Discussion", "discussion"),
    ("Conclusion", "conclusion"),
    ("Conclusions", "conclusion"),
    ("References", "references"),
    ("1 Introduction", "introduction"),
    ("1. Introduction", "introduction"),
    ("2. Materials and methods", "methods"),
    ("3.1 Adsorption experiments", "adsorption experiments"),
])
def test_section_heading_detection(line, expected):
    assert match_section_heading(line) == expected


@pytest.mark.parametrize("line", [
    "This is not a heading because it is a full sentence about water treatment.",
    "1. Smith, J. A. et al. (2020) A reference line should not be a heading.",
    "10.1016/j.watres.2020.123456",
    "3.14 is just a number",
])
def test_section_heading_negative(line):
    assert match_section_heading(line) is None


# ---------------------------------------------------------------- B. fallback

def test_fallback_when_no_sections(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 50)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    monkeypatch.setattr(settings, "section_min_chunk", 10)
    records = [
        rec("p", 1, "plain paragraph " + ("water treatment kinetics " * 10)),
        rec("p", 2, "another plain paragraph " + ("membrane fouling " * 10)),
    ]
    chunks = build_chunks_section_aware(records)
    assert chunks, "fallback must still produce chunks"
    assert all(c["metadata"]["chunking_mode"] == "fallback_fixed" for c in chunks)
    assert all(c["metadata"]["section"] == "body" for c in chunks)


# ---------------------------------------------------------------- C. page traceability

def test_single_page_chunk(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 1000)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    monkeypatch.setattr(settings, "section_min_chunk", 1)
    records = [rec("p", 3, "Abstract\n" + "x " * 20)]
    chunks = build_chunks_section_aware(records)
    assert len(chunks) == 1
    c = chunks[0]
    assert c["metadata"]["page_start"] == 3
    assert c["metadata"]["page_end"] == 3


def test_cross_page_chunk(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 200)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    monkeypatch.setattr(settings, "section_min_chunk", 1)
    records = [
        rec("p", 1, "Abstract\n" + "alpha beta gamma delta epsilon " * 3),
        rec("p", 2, "zeta eta theta iota kappa lambda " * 3),
    ]
    chunks = build_chunks_section_aware(records)
    # one section spanning two pages -> first chunk must span page 1..2
    assert chunks[0]["metadata"]["page_start"] == 1
    assert chunks[0]["metadata"]["page_end"] == 2


def test_section_spans_pages(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 30)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    monkeypatch.setattr(settings, "section_min_chunk", 1)
    records = [
        rec("p", 1, "Abstract\n" + "A " * 12),
        rec("p", 2, "B " * 12),
        rec("p", 3, "C " * 12),
    ]
    chunks = build_chunks_section_aware(records)
    # every chunk of the abstract section must report a monotonically valid range
    for c in chunks:
        assert 1 <= c["metadata"]["page_start"] <= c["metadata"]["page_end"] <= 3
        assert c["metadata"]["section"] == "abstract"


# ---------------------------------------------------------------- D. boundaries

def test_no_cross_paper(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 500)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    monkeypatch.setattr(settings, "section_min_chunk", 1)
    records = [
        rec("paperA", 1, "Abstract\n" + "TOKEN_A " * 20),
        rec("paperB", 1, "Abstract\n" + "TOKEN_B " * 20),
    ]
    chunks = build_chunks_section_aware(records)
    for c in chunks:
        pid = c["metadata"]["paper_id"]
        if pid == "paperA":
            assert "TOKEN_A" in c["text"] and "TOKEN_B" not in c["text"]
        else:
            assert "TOKEN_B" in c["text"] and "TOKEN_A" not in c["text"]


def test_no_cross_section_and_overlap_within_section(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 40)
    monkeypatch.setattr(settings, "chunk_overlap", 10)
    monkeypatch.setattr(settings, "section_min_chunk", 1)
    records = [
        rec("p", 1, "Introduction\n" + "INTRO " * 15),
        rec("p", 2, "Methods\n" + "METHOD " * 15),
    ]
    chunks = build_chunks_section_aware(records)
    intro = [c for c in chunks if c["metadata"]["section"] == "introduction"]
    meth = [c for c in chunks if c["metadata"]["section"] == "methods"]
    assert intro and meth
    for c in intro:
        assert "METHOD" not in c["text"]
    for c in meth:
        assert "INTRO" not in c["text"]
    # overlap exists within the intro section (tail of one == head of next)
    if len(intro) >= 2:
        assert intro[0]["text"][-10:] == intro[1]["text"][:10]


def test_references_separated_from_body(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 1000)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    monkeypatch.setattr(settings, "section_min_chunk", 1)
    records = [
        rec("p", 1, "Introduction\nbody text about membranes " * 5),
        rec("p", 2, "Conclusion\nfinal remarks " * 5),
        rec("p", 3, "References\n[1] Smith et al. 2020. A paper title. Journal."),
    ]
    chunks = build_chunks_section_aware(records)
    refs = [c for c in chunks if c["metadata"]["section"] == "references"]
    body = [c for c in chunks if c["metadata"]["section"] != "references"]
    assert refs, "references must be its own section"
    for c in refs:
        assert "membranes" not in c["text"] and "final remarks" not in c["text"]
    for c in body:
        assert "[1] Smith" not in c["text"]


def test_no_empty_or_header_only_chunks(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 200)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    monkeypatch.setattr(settings, "section_min_chunk", 10)
    header = "Running Header - Journal of Water Research"
    records = [
        rec("p", 1, f"{header}\nAbstract\n" + "abstract body sentence about membranes. " * 4),
        rec("p", 2, f"{header}\nIntroduction\n" + "intro body sentence about fouling. " * 4),
        rec("p", 3, f"{header}\nMethods\n" + "methods body sentence about flux. " * 4),
    ]
    chunks = build_chunks_section_aware(records)
    assert chunks
    for c in chunks:
        assert len(c["text"].strip()) >= 10
        assert "Running Header" not in c["text"]


def test_no_empty_chunks(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 100)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    monkeypatch.setattr(settings, "section_min_chunk", 10)
    records = [rec("p", 1, "Abstract\n" + "y " * 30)]
    chunks = build_chunks_section_aware(records)
    assert chunks
    assert all(c["text"].strip() for c in chunks)


# ---------------------------------------------------------------- E. determinism

def test_determinism(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 60)
    monkeypatch.setattr(settings, "chunk_overlap", 15)
    monkeypatch.setattr(settings, "section_min_chunk", 5)
    records = [
        rec("p", 1, "Abstract\n" + "ab cd ef " * 10),
        rec("p", 2, "Introduction\n" + "gh ij kl " * 10),
        rec("p", 3, "Methods\n" + "mn op qr " * 10),
        rec("p", 4, "References\n" + "ref1 ref2 ref3"),
    ]
    a = build_chunks_section_aware(records, paper_meta={"p": {"title": "T", "doi": "10.1/x"}})
    b = build_chunks_section_aware(records, paper_meta={"p": {"title": "T", "doi": "10.1/x"}})
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert [c["text"] for c in a] == [c["text"] for c in b]
    assert [c["metadata"] for c in a] == [c["metadata"] for c in b]


# ---------------------------------------------------------------- F. metadata

def test_metadata_complete_and_doi_may_be_empty(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 500)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    monkeypatch.setattr(settings, "section_min_chunk", 1)
    records = [rec("p", 1, "Abstract\n" + "text " * 10)]
    chunks = build_chunks_section_aware(records, paper_meta={"p": {"title": "A Paper"}})
    assert chunks
    m = chunks[0]["metadata"]
    for key in ("paper_id", "title", "section", "page_start", "page_end", "chunk_id"):
        assert key in m
    assert m["title"] == "A Paper"
    assert m["doi"] == ""  # missing DOI is allowed, not an error
    assert m["chunk_id"] == chunks[0]["id"]
    assert chunks[0]["text"].strip()


# ---------------------------------------------------------------- G. real PDFs

def _pages_of(pdf_path: Path):
    import fitz
    doc = fitz.open(str(pdf_path))
    recs = []
    for i in range(len(doc)):
        text = doc.load_page(i).get_text("text") or ""
        recs.append(rec(pdf_path.stem, i + 1, text, source=pdf_path.name))
    doc.close()
    return recs


def _prefer(pdfs, substrings):
    chosen = []
    rest = list(pdfs)
    for sub in substrings:
        for p in list(rest):
            if sub in p.name:
                chosen.append(p)
                rest.remove(p)
                break
    for p in rest:
        if len(chosen) >= 5:
            break
        chosen.append(p)
    return chosen[:5]


def test_real_pdf_regression(monkeypatch):
    pdfs = sorted(FINAL_CORPUS.glob("*.pdf"))
    if len(pdfs) < 5:
        pytest.skip("final_corpus has fewer than 5 PDFs")

    monkeypatch.setattr(settings, "chunk_size", 1000)
    monkeypatch.setattr(settings, "chunk_overlap", 150)
    monkeypatch.setattr(settings, "section_min_chunk", 30)

    picks = _prefer(pdfs, [
        "1-s2.0-0043135495001743",   # old Elsevier (Water Research 1996)
        "1-s2.0-S0043135408005642",  # Elsevier 2009
        "s44221-024-00212-x",        # Springer Nature (npj Clean Water 2024)
        "cs4c07556",                 # ACS style short name
        "UV-H_2O_2_UV_H_2O_2",       # Chinese thesis
    ])
    assert len(picks) == 5

    for pdf in picks:
        records = _pages_of(pdf)
        raw_chars = sum(len(r["text"]) for r in records)
        assert raw_chars > 0, f"{pdf.name}: no extractable text"
        chunks = build_chunks_section_aware(records)
        assert chunks, f"{pdf.name}: produced no chunks"
        for c in chunks:
            assert c["text"].strip()
            assert c["metadata"]["page_start"] >= 1
            assert c["metadata"]["page_end"] >= c["metadata"]["page_start"]
        # gross loss guard: chunked text must retain a meaningful fraction of raw text
        chunked = sum(len(c["text"]) for c in chunks)
        assert chunked >= 0.3 * raw_chars, f"{pdf.name}: text coverage too low ({chunked}/{raw_chars})"


# ---------------------------------------------------------------- legacy intact

def test_legacy_fixed_unchanged(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 50)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    records = [rec("p", 7, "x" * 120)]
    chunks = build_chunks_fixed(records)
    assert len(chunks) == 3
    assert chunks[0]["id"] == "p_p7_c0"
    assert chunks[0]["metadata"] == {"paper_id": "p", "source": "p.pdf", "page": 7, "chunk_index": 0}


def test_dispatcher_default_fixed(monkeypatch):
    monkeypatch.setattr(settings, "chunking_mode", "fixed")
    monkeypatch.setattr(settings, "chunk_size", 50)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    records = [rec("p", 1, "x" * 120)]
    chunks = build_chunks(records)
    assert chunks[0]["id"].startswith("p_p1_c")
