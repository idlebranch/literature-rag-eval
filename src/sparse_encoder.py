"""BGE-M3 sparse (lexical weights) encoder via the official FlagEmbedding path.

The project's local BGE-M3 snapshot is a sentence-transformers export used for
dense retrieval. FlagEmbedding's ``BGEM3FlagModel`` is the official
implementation for BGE-M3 sparse lexical weights; pointing it at the local
snapshot directory loads the same base weights plus ``sparse_linear.pt``
without any repository download.

This module never falls back to dense: if the model or head files are missing,
loading raises so callers fail loudly instead of silently degrading.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Dict, List

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_encoder_lock = threading.Lock()
_encoder = None
_encoder_model_path: str | None = None


@contextlib.contextmanager
def _dtype_kwarg_shim():
    """Translate FlagEmbedding's ``dtype=`` kwarg to ``torch_dtype=``.

    FlagEmbedding's model runner passes ``dtype`` to ``AutoModel.from_pretrained``,
    which older transformers releases reject. This shim only renames that one
    keyword so the official FlagEmbedding code path runs unchanged.
    """
    from transformers import AutoModel

    original = AutoModel.from_pretrained

    def patched(*args, **kwargs):
        if "dtype" in kwargs and "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return original(*args, **kwargs)

    AutoModel.from_pretrained = patched
    try:
        yield
    finally:
        AutoModel.from_pretrained = original


def resolve_bge_m3_model_path() -> str:
    """Return a local directory containing BGE-M3 weights and head files."""
    if settings.bge_m3_local_dir:
        return settings.bge_m3_local_dir

    # Resolve the cached snapshot without touching the network.
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            settings.embedding_model,
            local_files_only=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"No local snapshot of {settings.embedding_model!r} found in the "
            "HuggingFace cache. Set BGE_M3_LOCAL_DIR to a directory containing "
            "the model weights, or download the model first."
        ) from e


def get_sparse_encoder():
    """Load the official BGEM3FlagModel once (sparse-only usage).

    Returns a FlagEmbedding model configured for sparse lexical weights.
    Raises RuntimeError when the model or sparse head cannot be loaded.
    """
    global _encoder, _encoder_model_path
    with _encoder_lock:
        if _encoder is not None:
            return _encoder

        model_path = resolve_bge_m3_model_path()

        import os

        sparse_head = os.path.join(model_path, "sparse_linear.pt")
        if not os.path.exists(sparse_head):
            raise RuntimeError(
                f"sparse_linear.pt not found under {model_path!r}. The sparse "
                "head is required for BGE-M3 sparse retrieval; download it from "
                "the official BAAI/bge-m3 repository."
            )

        from FlagEmbedding import BGEM3FlagModel

        try:
            with _dtype_kwarg_shim():
                encoder = BGEM3FlagModel(
                    model_path,
                    use_fp16=False,  # CPU build: fp16 on CPU is slower/unsupported
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load BGE-M3 sparse encoder from {model_path!r}: {e}"
            ) from e

        _encoder = encoder
        _encoder_model_path = model_path
        logger.info("BGE-M3 sparse encoder loaded from %s", model_path)
        return _encoder


def sparse_model_path() -> str | None:
    return _encoder_model_path


def encode_query_sparse(query: str) -> Dict[str, float]:
    """Encode a query into BGE-M3 lexical weights {token_id: weight}."""
    encoder = get_sparse_encoder()
    out = encoder.encode_queries(
        [query],
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
        max_length=settings.sparse_max_length,
    )
    weights = out["lexical_weights"]
    if isinstance(weights, list):
        weights = weights[0]
    return {str(token_id): float(w) for token_id, w in weights.items()}


def encode_texts_sparse(texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
    """Encode passages into BGE-M3 lexical weights for index building."""
    encoder = get_sparse_encoder()
    results: List[Dict[str, float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        out = encoder.encode_corpus(
            batch,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
            max_length=settings.sparse_max_length,
        )
        weights_list = out["lexical_weights"]
        for weights in weights_list:
            results.append(
                {str(token_id): float(w) for token_id, w in weights.items()}
            )
    return results
