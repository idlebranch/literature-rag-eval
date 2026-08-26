"""Unit tests for the hybrid retrieval pipeline: RRF fusion, sparse index,
and reranker module. These avoid loading the real BGE-M3 model by injecting
fakes, so they run offline and fast.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import fusion
from src.config import settings
from src.fusion import rrf_fuse
from src.sparse_index import SparseIndex


def _hit(chunk_id: str, source: str, page: int, text: str, extra=None) -> dict:
    hit = {
        "id": chunk_id,
        "text": text,
        "metadata": {"source": source, "page": page, "chunk_index": 0},
    }
    if extra:
        hit.update(extra)
    return hit


# ─────────────────────────── RRF fusion ───────────────────────────


def test_rrf_merges_both_paths_and_keeps_metadata():
    dense = [
        _hit("a_p1_c0", "paperA.pdf", 1, "dense top"),
        _hit("a_p2_c0", "paperA.pdf", 2, "dense second"),
    ]
    sparse = [
        _hit("b_p1_c0", "paperB.pdf", 1, "sparse top"),
        _hit("a_p1_c0", "paperA.pdf", 1, "dense top"),  # overlap with dense
    ]

    fused = rrf_fuse(dense, sparse, fusion_k=10, rrf_k=60)

    # Overlapping chunk appears exactly once.
    ids = [h["id"] for h in fused]
    assert ids.count("a_p1_c0") == 1
    assert set(ids) == {"a_p1_c0", "a_p2_c0", "b_p1_c0"}

    # The chunk present in BOTH lists scores highest (1/(60+1) + 1/(60+2)).
    top = fused[0]
    assert top["id"] == "a_p1_c0"
    expected = 1 / 61 + 1 / 62
    assert top["rrf_score"] == pytest.approx(expected)

    # Metadata preserved from the source hit.
    assert top["metadata"]["source"] == "paperA.pdf"
    assert top["metadata"]["page"] == 1
    assert top["text"] == "dense top"


def test_rrf_single_path_ranking_matches_rrf_formula():
    hits = [
        _hit("c0", "s.pdf", 1, "first"),
        _hit("c1", "s.pdf", 2, "second"),
        _hit("c2", "s.pdf", 3, "third"),
    ]
    fused = rrf_fuse(hits, [], fusion_k=10, rrf_k=60)
    scores = [h["rrf_score"] for h in fused]
    assert scores == [
        pytest.approx(1 / 61),
        pytest.approx(1 / 62),
        pytest.approx(1 / 63),
    ]
    # Order preserved.
    assert [h["id"] for h in fused] == ["c0", "c1", "c2"]


def test_rrf_respects_fusion_k_and_no_duplicates():
    dense = [_hit(f"d{i}", "d.pdf", i, f"dense {i}") for i in range(5)]
    sparse = [_hit(f"s{i}", "s.pdf", i, f"sparse {i}") for i in range(5)]
    fused = rrf_fuse(dense, sparse, fusion_k=6, rrf_k=60)
    assert len(fused) == 6
    assert len({h["id"] for h in fused}) == 6


def test_rrf_falls_back_to_metadata_identity_when_id_missing():
    dense = [{"text": "x", "metadata": {"paper_id": "p", "page": 1, "chunk_index": 0}}]
    sparse = [{"text": "x", "metadata": {"paper_id": "p", "page": 1, "chunk_index": 0}}]
    fused = rrf_fuse(dense, sparse, fusion_k=5, rrf_k=60)
    assert len(fused) == 1
    # Ranked first in BOTH lists: 1/(60+1) + 1/(60+1).
    assert fused[0]["rrf_score"] == pytest.approx(2 / 61)


# ─────────────────────────── Sparse index ───────────────────────────


def _build_index(postings_by_term):
    """postings_by_term: {token_id: [(doc_index, weight), ...]}"""
    token_ids = np.array(sorted(postings_by_term), dtype=np.int64)
    indptr = np.zeros(len(token_ids) + 1, dtype=np.int64)
    doc_parts, w_parts = [], []
    for i, term in enumerate(token_ids):
        pairs = postings_by_term[term]
        indptr[i + 1] = indptr[i] + len(pairs)
        doc_parts.append(np.array([p[0] for p in pairs], dtype=np.int32))
        w_parts.append(np.array([p[1] for p in pairs], dtype=np.float32))
    doc_indices = np.concatenate(doc_parts)
    weights = np.concatenate(w_parts)
    return SparseIndex(token_ids, indptr, doc_indices, weights, ["docA", "docB", "docC"])


def test_sparse_index_search_sums_shared_tokens():
    # term 10 shared by docA(0.5) and docB(0.2); term 20 only in docA(0.3)
    index = _build_index({10: [(0, 0.5), (1, 0.2)], 20: [(0, 0.3)]})
    hits = index.search({"10": 1.0, "20": 1.0}, top_k=3)
    by_id = dict(hits)
    assert by_id["docA"] == pytest.approx(0.5 + 0.3)
    assert by_id["docB"] == pytest.approx(0.2)
    assert "docC" not in by_id
    # Sorted by descending score.
    assert [h[0] for h in hits][0] == "docA"


def test_sparse_index_search_empty_query():
    index = _build_index({10: [(0, 0.5)]})
    assert index.search({}, top_k=5) == []


def test_sparse_index_save_and_load(tmp_path, monkeypatch):
    from src import sparse_index as si_mod

    index = _build_index({10: [(0, 0.5), (1, 0.2)]})

    monkeypatch.setattr(settings, "collection_name", "literature_chunks")
    # Freeze collection_version so manifest matches on load.
    monkeypatch.setattr(si_mod, "collection_version", lambda: "test-version")
    monkeypatch.setattr(settings, "sparse_index_dir", str(tmp_path))

    index.save(tmp_path, manifest_extra={"builder": "test"})
    si_mod.clear_index_cache()

    loaded = si_mod.load_index(strict=True)
    assert loaded.chunk_count == 3
    hits = loaded.search({"10": 1.0}, top_k=2)
    by_id = dict(hits)
    assert by_id["docA"] == pytest.approx(0.5)


def test_sparse_index_load_detects_stale_collection(tmp_path, monkeypatch):
    from src import sparse_index as si_mod

    index = _build_index({10: [(0, 0.5)]})
    monkeypatch.setattr(settings, "sparse_index_dir", str(tmp_path))

    # Build under one collection version…
    monkeypatch.setattr(si_mod, "collection_version", lambda: "version-old")
    index.save(tmp_path, manifest_extra={})
    si_mod.clear_index_cache()

    # …then the collection changes.
    monkeypatch.setattr(si_mod, "collection_version", lambda: "version-new")
    with pytest.raises(RuntimeError, match="different Chroma collection"):
        si_mod.load_index(strict=True)


# ─────────────────────────── Reranker module ───────────────────────────


def test_reranker_rerank_orders_by_score_and_preserves_metadata(monkeypatch):
    from src import reranker as reranker_mod

    hits = [
        _hit("a", "a.pdf", 1, "weak passage"),
        _hit("b", "b.pdf", 2, "strong passage"),
    ]

    class FakeModel:
        def compute_score(self, pairs):
            # Score proportional to whether the passage is "strong".
            return [0.1 if "weak" in p[1] else 0.9 for p in pairs]

    fake = object.__new__(reranker_mod.BgeReranker)
    fake._model_name = "fake-reranker"
    fake._model = FakeModel()

    reranked = fake.rerank("query", hits, final_k=2)
    assert [h["id"] for h in reranked] == ["b", "a"]
    assert reranked[0]["rerank_score"] == pytest.approx(0.9)
    # Metadata preserved through reranking.
    assert reranked[0]["metadata"]["source"] == "b.pdf"
    assert reranked[0]["metadata"]["page"] == 2


def test_reranker_final_k_truncates(monkeypatch):
    from src import reranker as reranker_mod

    hits = [_hit(f"h{i}", "s.pdf", i, f"passage {i}") for i in range(6)]

    class FakeModel:
        def compute_score(self, pairs):
            return list(range(len(pairs)))

    fake = object.__new__(reranker_mod.BgeReranker)
    fake._model_name = "fake-reranker"
    fake._model = FakeModel()

    reranked = fake.rerank("q", hits, final_k=3)
    assert len(reranked) == 3
    assert [h["id"] for h in reranked] == ["h5", "h4", "h3"]


def test_reranker_empty_input():
    from src import reranker as reranker_mod

    fake = object.__new__(reranker_mod.BgeReranker)
    assert fake.rerank("q", [], final_k=5) == []


def test_get_reranker_raises_when_model_missing(monkeypatch):
    from src import reranker as reranker_mod

    monkeypatch.setattr(reranker_mod, "_reranker", None)
    monkeypatch.setattr(
        reranker_mod, "resolve_reranker_model_path", lambda: (_ for _ in ()).throw(
            RuntimeError("no local snapshot")
        )
    )
    with pytest.raises(RuntimeError):
        reranker_mod.get_reranker()


# ─────────────────────────── Mode wiring ───────────────────────────


def test_unknown_retrieval_mode_raises(monkeypatch):
    from src import retriever

    monkeypatch.setattr(settings, "retrieval_mode", "bogus_mode")
    with pytest.raises(ValueError, match="Unknown RETRIEVAL_MODE"):
        retriever.retrieve_with_metrics("question")


def test_hybrid_mode_without_index_fails_loudly(monkeypatch):
    from src import retriever
    from src import sparse_index as si_mod

    monkeypatch.setattr(settings, "retrieval_mode", "hybrid_dense_sparse")
    monkeypatch.setattr(settings, "sparse_index_dir", "sparse_index_does_not_exist_xyz")
    si_mod.clear_index_cache()
    retriever.clear_retrieval_caches()

    with pytest.raises(RuntimeError, match="Sparse index not found"):
        retriever.retrieve_with_metrics("question")


def test_hybrid_mode_result_reports_mode(monkeypatch):
    """End-to-end hybrid retrieval with injected fake dense/sparse paths."""
    from src import retriever
    from src import fusion as fusion_mod

    monkeypatch.setattr(settings, "retrieval_mode", "hybrid_dense_sparse")
    monkeypatch.setattr(settings, "hybrid_dense_k", 5)
    monkeypatch.setattr(settings, "hybrid_sparse_k", 5)
    monkeypatch.setattr(settings, "hybrid_fusion_k", 8)
    retriever.clear_retrieval_caches()

    dense_hits = [_hit("a_p1_c0", "a.pdf", 1, "dense content", {"distance": 0.3})]
    sparse_hits = [_hit("b_p1_c0", "b.pdf", 1, "sparse content", {"sparse_score": 0.7})]

    monkeypatch.setattr(retriever, "_cached_embedding", lambda q: ([0.0, 1.0], False))
    monkeypatch.setattr(retriever, "search", lambda emb, top_k: dense_hits)
    monkeypatch.setattr(
        "src.sparse_index.load_index", lambda strict=True: object()
    )
    monkeypatch.setattr(
        "src.sparse_encoder.encode_query_sparse", lambda q: {"10": 1.0}
    )
    monkeypatch.setattr(
        "src.fusion.resolve_sparse_hits", lambda scored: sparse_hits
    )

    class FakeIndex:
        def search(self, weights, top_k):
            return [("b_p1_c0", 0.7)]

    monkeypatch.setattr("src.sparse_index.load_index", lambda strict=True: FakeIndex())

    result = retriever.retrieve_with_metrics("question", top_k=5)
    assert result.retrieval_mode == "hybrid_dense_sparse"
    ids = [h["id"] for h in result.hits]
    assert "a_p1_c0" in ids and "b_p1_c0" in ids
    # Sparse-only hit inherited a dense distance for the evidence gate.
    sparse_only = next(h for h in result.hits if h["id"] == "b_p1_c0")
    assert sparse_only["distance"] == pytest.approx(0.3)
