"""Lightweight persistent sparse index for BGE-M3 lexical weights.

Chroma stores dense vectors only; this module keeps an inverted index of
BGE-M3 sparse lexical weights built from the same Chroma collection.

Layout under ``sparse_index/``:
  - ``index.npz``   token_ids / indptr / doc_indices / weights
  - ``manifest.json`` build metadata, chunk ids, consistency marker

The index is keyed to the Chroma collection it was built from. If the
collection changes, :func:`load_index` raises instead of serving stale
postings against new documents.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import settings
from src.retriever import collection_version
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SparseIndex:
    """Inverted index: term -> list of (doc_index, weight)."""

    def __init__(
        self,
        token_ids: np.ndarray,
        indptr: np.ndarray,
        doc_indices: np.ndarray,
        weights: np.ndarray,
        doc_ids: List[str],
    ) -> None:
        self.token_ids = token_ids
        self.indptr = indptr
        self.doc_indices = doc_indices
        self.weights = weights
        self.doc_ids = doc_ids
        self._id_to_doc_index: Dict[str, int] = {
            doc_id: i for i, doc_id in enumerate(doc_ids)
        }

    @property
    def chunk_count(self) -> int:
        return len(self.doc_ids)

    @property
    def posting_count(self) -> int:
        return int(self.doc_indices.size)

    def search(self, query_weights: Dict[str, float], top_k: int) -> List[Tuple[str, float]]:
        """Return [(chunk_id, score)] sorted by descending sparse score.

        Score is the standard BGE-M3 lexical matching score: sum over shared
        tokens of query_weight * document_weight.
        """
        if not query_weights:
            return []

        accumulated: Optional[np.ndarray] = None
        for token_str, q_weight in query_weights.items():
            if not token_str.isdigit():
                continue
            pos = np.searchsorted(self.token_ids, int(token_str))
            if pos >= len(self.token_ids) or self.token_ids[pos] != int(token_str):
                continue
            start, end = self.indptr[pos], self.indptr[pos + 1]
            if start == end:
                continue
            docs = self.doc_indices[start:end]
            contrib = self.weights[start:end] * float(q_weight)
            if accumulated is None:
                accumulated = np.bincount(docs, weights=contrib, minlength=len(self.doc_ids))
            else:
                accumulated += np.bincount(
                    docs, weights=contrib, minlength=len(self.doc_ids)
                )

        if accumulated is None:
            return []

        if top_k >= len(self.doc_ids):
            top_idx = np.argsort(-accumulated)
        else:
            top_idx = np.argpartition(-accumulated, top_k)[:top_k]
            top_idx = top_idx[np.argsort(-accumulated[top_idx])]

        hits: List[Tuple[str, float]] = []
        for idx in top_idx:
            score = float(accumulated[idx])
            if score <= 0.0:
                break
            hits.append((self.doc_ids[idx], score))
            if len(hits) >= top_k:
                break
        return hits

    def save(self, directory: Path, manifest_extra: Dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "index.npz",
            token_ids=self.token_ids,
            indptr=self.indptr,
            doc_indices=self.doc_indices,
            weights=self.weights,
        )
        manifest = {
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "embedding_model": settings.embedding_model,
            "sparse_max_length": settings.sparse_max_length,
            "chunk_count": self.chunk_count,
            "posting_count": self.posting_count,
            "collection_version": collection_version(),
            **manifest_extra,
        }
        (directory / "manifest.json").write_text(
            json.dumps(
                {"manifest": manifest, "doc_ids": self.doc_ids},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def build_sparse_index(verbose: bool = True) -> SparseIndex:
    """Build the inverted index from the current Chroma collection.

    Fetches every document from Chroma, encodes it with the official
    BGE-M3 sparse encoder, and aggregates per-token postings.
    """
    from src.sparse_encoder import encode_texts_sparse
    from src.vectorstore import get_collection

    collection = get_collection()
    total = collection.count()
    if total == 0:
        raise RuntimeError("Chroma collection is empty; nothing to index.")

    logger.info("Fetching %d chunks from Chroma for sparse indexing", total)
    batch_size = 512
    all_ids: List[str] = []
    all_docs: List[str] = []
    for offset in range(0, total, batch_size):
        chunk = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents"],
        )
        all_ids.extend(chunk["ids"])
        all_docs.extend(chunk["documents"])

    if len(all_ids) != total:
        raise RuntimeError(
            f"Chroma returned {len(all_ids)} chunks but count()={total}"
        )

    logger.info("Encoding sparse weights for %d chunks", len(all_docs))
    all_weights = encode_texts_sparse(all_docs, batch_size=16)

    # Aggregate postings: term -> list of (doc_index, weight)
    postings: Dict[int, List[Tuple[int, float]]] = {}
    for doc_index, weights in enumerate(all_weights):
        for token_id, weight in weights.items():
            postings.setdefault(int(token_id), []).append((doc_index, weight))

    token_ids_sorted = sorted(postings.keys())
    token_ids = np.array(token_ids_sorted, dtype=np.int64)
    indptr = np.zeros(len(token_ids_sorted) + 1, dtype=np.int64)
    doc_index_parts: List[np.ndarray] = []
    weight_parts: List[np.ndarray] = []
    for i, term in enumerate(token_ids_sorted):
        pairs = postings[term]
        indptr[i + 1] = indptr[i] + len(pairs)
        doc_index_parts.append(np.array([p[0] for p in pairs], dtype=np.int32))
        weight_parts.append(np.array([p[1] for p in pairs], dtype=np.float32))

    doc_indices = np.concatenate(doc_index_parts) if doc_index_parts else np.zeros(0, dtype=np.int32)
    weights = np.concatenate(weight_parts) if weight_parts else np.zeros(0, dtype=np.float32)

    index = SparseIndex(token_ids, indptr, doc_indices, weights, all_ids)
    logger.info(
        "Sparse index built: %d chunks, %d terms, %d postings",
        index.chunk_count,
        len(token_ids_sorted),
        index.posting_count,
    )
    return index


_index_lock_state: Dict[str, object] = {"index": None, "version": None}


def sparse_index_dir() -> Path:
    return Path(settings.sparse_index_dir)


def load_index(strict: bool = True) -> SparseIndex:
    """Load the persisted sparse index, verifying collection consistency.

    With ``strict=True`` a missing or stale index raises RuntimeError so
    hybrid retrieval fails loudly instead of silently returning dense-only
    results.
    """
    directory = sparse_index_dir()
    npz_path = directory / "index.npz"
    manifest_path = directory / "manifest.json"

    if not npz_path.exists() or not manifest_path.exists():
        message = (
            "Sparse index not found. Build it with "
            "`python -m scripts.build_sparse_index` before enabling hybrid "
            "retrieval."
        )
        if strict:
            raise RuntimeError(message)
        raise RuntimeError(message)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = payload.get("manifest", {})
    doc_ids = payload.get("doc_ids", [])

    cached = _index_lock_state["index"]
    if cached is not None and _index_lock_state["version"] == collection_version():
        return cached  # type: ignore[return-value]

    data = np.load(npz_path)
    index = SparseIndex(
        token_ids=data["token_ids"],
        indptr=data["indptr"],
        doc_indices=data["doc_indices"],
        weights=data["weights"],
        doc_ids=doc_ids,
    )

    current_version = collection_version()
    if manifest.get("collection_version") != current_version:
        raise RuntimeError(
            "Sparse index was built against a different Chroma collection "
            f"version ({manifest.get('collection_version')!r} vs current "
            f"{current_version!r}). Rebuild with `python -m scripts.build_sparse_index`."
        )
    if manifest.get("chunk_count") != index.chunk_count:
        raise RuntimeError(
            "Sparse index manifest chunk_count does not match stored doc ids."
        )

    _index_lock_state["index"] = index
    _index_lock_state["version"] = current_version
    logger.info(
        "Sparse index loaded: %d chunks, %d postings",
        index.chunk_count,
        index.posting_count,
    )
    return index


def clear_index_cache() -> None:
    """Test/admin hook; never deletes files on disk."""
    _index_lock_state["index"] = None
    _index_lock_state["version"] = None
