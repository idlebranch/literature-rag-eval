"""Tests for the real ingestion/index wiring (fixed vs section_aware), the
index-eligibility policy, metadata enrichment, and sparse/dense version binding.

No BGE-M3 / Chroma network or model loading: everything is exercised with
synthetic page records and injected fakes.
"""
import json

import numpy as np
import pytest

from src import ingest as ingest_mod
from src.chunking import build_chunks, build_chunks_section_aware, is_indexable
from src.config import settings
from src.ingest import (
    REQUIRED_METADATA,
    enrich_chunks,
    load_paper_meta,
    load_pages,
    partition_indexable,
)


def _rec(pid, page, text):
    return {"paper_id": pid, "source": f"{pid}.pdf", "page": page, "text": text}


# ---------------------------------------------------------------- A/B: pipeline wiring

def test_fixed_mode_uses_load_pdfs_and_fixed_chunker(monkeypatch, tmp_path):
    (tmp_path / "dummy.pdf").write_bytes(b"%PDF-1.4\n")
    fake = [{"paper_id": "dummy", "source": "dummy.pdf", "page": 1, "text": "x" * 120}]
    monkeypatch.setattr(ingest_mod, "load_pdfs", lambda d: fake)

    def boom(_p):
        raise AssertionError("load_pdf_pages must not be used for fixed")

    monkeypatch.setattr(ingest_mod, "load_pdf_pages", boom)
    assert load_pages(str(tmp_path), "fixed") == fake

    monkeypatch.setattr(settings, "chunk_size", 50)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    chunks = build_chunks(fake, chunking_mode="fixed")
    assert chunks[0]["id"].startswith("dummy_p1_c")
    assert chunks[0]["metadata"].get("chunking_mode") is None  # legacy metadata untouched


def test_section_aware_mode_uses_load_pdf_pages(monkeypatch, tmp_path):
    (tmp_path / "dummy.pdf").write_bytes(b"%PDF-1.4\n")
    fake = [{"paper_id": "dummy", "source": "dummy.pdf", "page": 1, "text": "Abstract\ntext"}]
    monkeypatch.setattr(ingest_mod, "load_pdf_pages", lambda p: fake)

    def boom(_d):
        raise AssertionError("load_pdfs must not be used for section_aware")

    monkeypatch.setattr(ingest_mod, "load_pdfs", boom)
    assert load_pages(str(tmp_path), "section_aware") == fake

    monkeypatch.setattr(settings, "chunk_size", 100)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    monkeypatch.setattr(settings, "section_min_chunk", 1)
    chunks = build_chunks(fake, chunking_mode="section_aware")
    assert chunks[0]["metadata"]["chunking_mode"] == "section_aware"
    assert "abstract" in chunks[0]["id"]  # section slug present in the id


# ---------------------------------------------------------------- C/D: index eligibility

def test_is_indexable_policy():
    assert not is_indexable("references")
    assert not is_indexable("References")
    assert not is_indexable("acknowledgments")
    assert is_indexable("results")
    assert is_indexable("introduction")
    assert is_indexable("")  # unknown -> keep
    assert is_indexable(None)


def test_partition_indexable_filters_but_keeps_raw(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 1000)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    monkeypatch.setattr(settings, "section_min_chunk", 1)
    records = [
        _rec("p", 1, "Introduction\n" + "body sentence " * 20),
        _rec("p", 2, "References\n[1] Smith et al. 2020. A title."),
        _rec("p", 3, "Acknowledgments\nWe thank the funders."),
    ]
    chunks = build_chunks_section_aware(records)
    sections = {c["metadata"]["section"] for c in chunks}
    assert "references" in sections  # raw chunks still contain references
    assert "acknowledgments" in sections

    indexable, excluded = partition_indexable(chunks)
    assert all(c["metadata"]["section"] != "references" for c in indexable)
    assert all(c["metadata"]["section"] != "acknowledgments" for c in indexable)
    assert {c["metadata"]["section"] for c in excluded} == {"references", "acknowledgments"}
    assert len(chunks) == len(indexable) + len(excluded)  # nothing deleted


# ---------------------------------------------------------------- E: coverage

def test_same_paper_coverage_across_modes(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 100)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    monkeypatch.setattr(settings, "section_min_chunk", 1)
    records = [
        _rec("p1", 1, "Abstract\n" + "x " * 30),
        _rec("p2", 1, "y " * 60),
    ]
    fixed = build_chunks(records, chunking_mode="fixed")
    aware = build_chunks(records, chunking_mode="section_aware")
    assert {c["metadata"]["paper_id"] for c in fixed} == {"p1", "p2"}
    assert {c["metadata"]["paper_id"] for c in aware} == {"p1", "p2"}


# ---------------------------------------------------------------- F: metadata

def test_enrich_metadata_complete_for_fixed():
    fixed = [{"id": "p_p1_c0", "text": "t",
              "metadata": {"paper_id": "p", "source": "p.pdf", "page": 1, "chunk_index": 0}}]
    meta = {"p": {"title": "Title One", "doi": "10.1/x"}}
    enriched = enrich_chunks(fixed, meta, "fixed")
    m = enriched[0]["metadata"]
    for key in REQUIRED_METADATA:
        assert key in m, key
    assert m["title"] == "Title One"
    assert m["doi"] == "10.1/x"
    assert m["section"] == ""
    assert m["page_start"] == 1 and m["page_end"] == 1
    assert m["chunk_id"] == "p_p1_c0"
    assert m["chunking_mode"] == "fixed"


def test_enrich_metadata_doi_may_be_empty(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 100)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    monkeypatch.setattr(settings, "section_min_chunk", 1)
    records = [_rec("p", 1, "Abstract\n" + "x " * 20)]
    chunks = build_chunks_section_aware(records, paper_meta={"p": {"title": "T"}})
    enriched = enrich_chunks(chunks, {"p": {"title": "T"}}, "section_aware")
    m = enriched[0]["metadata"]
    assert m["doi"] == ""  # missing DOI ok
    for key in REQUIRED_METADATA:
        assert key in m, key


def test_load_paper_meta(tmp_path):
    mf = tmp_path / "manifest.csv"
    mf.write_text("final_file,resolution,doi,title\nfoo.pdf,unique_doi,10.1/x,Title One\n",
                  encoding="utf-8")
    meta = load_paper_meta(str(mf))
    assert meta["foo.pdf"] == {"title": "Title One", "doi": "10.1/x"}
    assert meta["foo"] == {"title": "Title One", "doi": "10.1/x"}


# ---------------------------------------------------------------- G: sparse/dense version binding

def test_sparse_save_records_dense_collection_version(tmp_path, monkeypatch):
    from src import sparse_index as si

    index = si.SparseIndex(
        token_ids=np.array([10], dtype=np.int64),
        indptr=np.array([0, 1], dtype=np.int64),
        doc_indices=np.array([0], dtype=np.int32),
        weights=np.array([0.5], dtype=np.float32),
        doc_ids=["docA"],
    )
    monkeypatch.setattr(settings, "sparse_index_dir", str(tmp_path))
    monkeypatch.setattr(si, "collection_version", lambda: "fixed_270:12345")
    index.save(tmp_path, manifest_extra={"index_name": "fixed_270"})
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert payload["manifest"]["collection_version"] == "fixed_270:12345"
    assert payload["manifest"]["chunk_count"] == 1
    assert payload["doc_ids"] == ["docA"]


# ---------------------------------------------------------------- H: determinism

def test_build_deterministic_and_no_duplicate_ids(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 60)
    monkeypatch.setattr(settings, "chunk_overlap", 15)
    monkeypatch.setattr(settings, "section_min_chunk", 5)
    records = [
        _rec("p", 1, "Abstract\n" + "ab cd ef " * 10),
        _rec("p", 2, "Methods\n" + "gh ij kl " * 10),
        _rec("p", 3, "References\n" + "ref1 ref2 ref3"),
    ]
    meta = {"p": {"title": "T", "doi": "10.1/x"}}
    a = build_chunks(records, chunking_mode="section_aware", paper_meta=meta)
    b = build_chunks(records, chunking_mode="section_aware", paper_meta=meta)
    assert [c["id"] for c in a] == [c["id"] for c in b]
    ids = [c["id"] for c in a]
    assert len(ids) == len(set(ids)), "chunk ids must be unique (no duplicate writes)"


def test_fixed_build_deterministic(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 50)
    monkeypatch.setattr(settings, "chunk_overlap", 0)
    records = [_rec("p", 1, "x" * 120), _rec("p", 2, "y" * 120)]
    a = build_chunks(records, chunking_mode="fixed")
    b = build_chunks(records, chunking_mode="fixed")
    assert [c["id"] for c in a] == [c["id"] for c in b]
