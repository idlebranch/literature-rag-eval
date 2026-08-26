import os
# Disable Chroma telemetry before import to avoid "capture() takes 1 positional argument but 3 were given"
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_OTEL_ENABLED", "False")

# ── Monkey-patch posthog to prevent chromadb 0.5.23 → posthog 7.x API mismatch ──
# chromadb 0.5.23 calls posthog.capture(user_id, event_name, properties) with 3
# positional args, but posthog ≥4.x accepts only capture(event, **kwargs).
# Since we disable telemetry anyway, replace capture with a no-op.
import posthog as _posthog
_posthog.capture = lambda *a, **kw: None

from functools import lru_cache
from typing import Dict, List
import chromadb
from chromadb import Settings as ChromaSettings  # chromadb.config.Settings → chromadb.Settings (v0.5+)
from src.config import settings


@lru_cache(maxsize=4)
def _get_client(chroma_dir: str):
    return chromadb.PersistentClient(
        path=chroma_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_client():
    """Return one persistent Chroma client per configured database path."""
    return _get_client(settings.chroma_dir)


@lru_cache(maxsize=8)
def _get_collection(chroma_dir: str, collection_name: str):
    client = _get_client(chroma_dir)
    return client.get_or_create_collection(name=collection_name)


def get_collection():
    """Return one collection handle per database path/name pair."""
    return _get_collection(settings.chroma_dir, settings.collection_name)


def clear_vectorstore_caches() -> None:
    """Test/admin hook; never deletes the on-disk index."""
    _get_collection.cache_clear()
    _get_client.cache_clear()


def reset_collection():
    """Create the configured collection only when no active collection exists.

    Destructive in-place resets are intentionally disabled. Build a candidate
    index in a different CHROMA_DIR and validate it before any manual swap.
    """
    client = get_client()
    try:
        client.get_collection(name=settings.collection_name)
    except Exception:
        return client.create_collection(name=settings.collection_name)
    raise RuntimeError(
        "Refusing to overwrite the active Chroma collection. "
        "Set CHROMA_DIR to a new candidate directory and build there."
    )


def add_chunks(chunks: List[Dict], embeddings: List[List[float]], reset: bool = True):
    collection = reset_collection() if reset else get_collection()

    ids = [c["id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]

    batch_size = 128
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=docs[i : i + batch_size],
            metadatas=metas[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
        )

    return collection


def get_collection_count() -> int:
    """Return the number of chunks in the Chroma collection, or 0 if unavailable."""
    try:
        return get_collection().count()
    except Exception:
        return 0


def search(query_embedding: List[float], top_k: int):
    collection = get_collection()
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc_id, doc, meta, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append(
            {
                "id": doc_id,
                "text": doc,
                "metadata": meta,
                "distance": dist,
            }
        )
    return hits
