"""Cross-encoder reranker wrapper for hybrid retrieval.

Loads ``BAAI/bge-reranker-v2-m3`` once and reuses it across requests. The
model never silently degrades: if it is configured but cannot be loaded,
:func:`get_reranker` raises so callers can surface an explicit error. When
reranking is disabled the module is never touched.
"""
from __future__ import annotations

import threading
from typing import Dict, List

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_reranker_lock = threading.Lock()
_reranker = None


def resolve_reranker_model_path() -> str:
    """Resolve the reranker to a local directory without network access."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(settings.reranker_model, local_files_only=True)
    except Exception as e:
        raise RuntimeError(
            f"No local snapshot of {settings.reranker_model!r} found in the "
            "HuggingFace cache. Download the model first or point "
            "RERANKER_MODEL at a local directory."
        ) from e


class BgeReranker:
    """Thin wrapper around FlagEmbedding's FlagReranker for bge-reranker-v2-m3."""

    def __init__(self, model_name: str) -> None:
        from FlagEmbedding import FlagReranker

        self._model_name = model_name
        model_path = resolve_reranker_model_path()
        self._model = FlagReranker(model_path, use_fp16=False)
        logger.info("Reranker %s loaded from %s", model_name, model_path)

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self, query: str, hits: List[Dict], final_k: int
    ) -> List[Dict]:
        """Score (query, chunk_text) pairs and return top ``final_k`` hits.

        Each returned hit is a copy of the input hit dict with a
        ``rerank_score`` field added. Metadata is preserved unchanged.
        """
        if not hits:
            return []

        pairs = [[query, str(h.get("text", ""))] for h in hits]
        scores = self._model.compute_score(pairs)
        if not isinstance(scores, list):
            scores = [scores]

        scored = sorted(
            zip(scores, range(len(hits))), key=lambda item: item[0], reverse=True
        )[:final_k]

        reordered: List[Dict] = []
        for score, original_index in scored:
            hit = dict(hits[original_index])
            hit["rerank_score"] = float(score)
            reordered.append(hit)
        return reordered


def reranker_available() -> bool:
    """Best-effort check that the reranker model files exist locally."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(settings.reranker_model, local_files_only=True)
        return True
    except Exception:
        return False


def get_reranker() -> BgeReranker:
    """Load the reranker once. Raises RuntimeError if it cannot be loaded."""
    global _reranker
    with _reranker_lock:
        if _reranker is not None:
            return _reranker

        try:
            _reranker = BgeReranker(settings.reranker_model)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load reranker {settings.reranker_model!r}: {e}. "
                "Download the model or disable reranking (RERANKER_ENABLED=false)."
            ) from e
        return _reranker
