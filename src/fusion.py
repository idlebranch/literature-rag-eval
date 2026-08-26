"""Reciprocal Rank Fusion (RRF) of dense and sparse retrieval results.

RRF merges two ranked lists without requiring their scores to be comparable:
    rrf_score(doc) = sum over lists L where doc appears: 1 / (k + rank_L(doc))
``k`` (default 60) dampens the contribution of deep ranks.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _chunk_key(hit: Dict) -> str:
    """Stable identity for a hit: Chroma chunk id when available."""
    metadata = hit.get("metadata") or {}
    chunk_id = hit.get("id")
    if chunk_id:
        return str(chunk_id)
    # Fallback identity reconstructed from metadata fields.
    paper = metadata.get("paper_id", "")
    page = metadata.get("page", "")
    chunk_index = metadata.get("chunk_index", "")
    return f"{paper}_p{page}_c{chunk_index}"


def resolve_sparse_hits(sparse_hits: List[Tuple[str, float]]) -> List[Dict]:
    """Fetch chunk text + metadata from Chroma for sparse-scored chunk ids."""
    from src.vectorstore import get_collection

    if not sparse_hits:
        return []

    ids = [chunk_id for chunk_id, _ in sparse_hits]
    score_by_id = {chunk_id: score for chunk_id, score in sparse_hits}

    collection = get_collection()
    result = collection.get(ids=ids, include=["documents", "metadatas"])

    hits: List[Dict] = []
    for doc_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        hits.append(
            {
                "id": doc_id,
                "text": doc,
                "metadata": meta or {},
                "sparse_score": float(score_by_id.get(doc_id, 0.0)),
            }
        )
    return hits


def rrf_fuse(
    dense_hits: List[Dict],
    sparse_hits: List[Dict],
    fusion_k: int,
    rrf_k: int | None = None,
) -> List[Dict]:
    """Merge dense and sparse ranked lists via RRF.

    Args:
        dense_hits: hits ranked by dense similarity (best first).
        sparse_hits: hits ranked by sparse score (best first).
        fusion_k: number of fused candidates to return.
        rrf_k: RRF dampening constant; defaults to settings.rrf_k.

    Returns:
        Fused hits (best first). Each hit carries ``rrf_score`` and the
        dense/sparse provenance fields from its source lists. Duplicate chunk
        ids are merged so a chunk never appears twice.
    """
    if rrf_k is None:
        rrf_k = settings.rrf_k

    fused: Dict[str, Dict] = {}

    for rank, hit in enumerate(dense_hits):
        key = _chunk_key(hit)
        merged = fused.setdefault(key, dict(hit))
        merged["dense_rank"] = rank
        merged["rrf_score"] = merged.get("rrf_score", 0.0) + 1.0 / (rrf_k + rank + 1)

    for rank, hit in enumerate(sparse_hits):
        key = _chunk_key(hit)
        merged = fused.setdefault(key, dict(hit))
        merged["sparse_rank"] = rank
        merged["rrf_score"] = merged.get("rrf_score", 0.0) + 1.0 / (rrf_k + rank + 1)

    ordered = sorted(fused.values(), key=lambda h: h.get("rrf_score", 0.0), reverse=True)
    return ordered[:fusion_k]
