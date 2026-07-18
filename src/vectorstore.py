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

from typing import Dict, List
import chromadb
from chromadb import Settings as ChromaSettings  # chromadb.config.Settings → chromadb.Settings (v0.5+)
from src.config import settings


def get_client():
    return chromadb.PersistentClient(
        path=settings.chroma_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection():
    client = get_client()
    return client.get_or_create_collection(name=settings.collection_name)


def reset_collection():
    client = get_client()
    try:
        client.delete_collection(name=settings.collection_name)
    except Exception:
        pass
    return client.get_or_create_collection(name=settings.collection_name)


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
        client = get_client()
        collection = client.get_collection(name=settings.collection_name)
        return collection.count()
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
    for doc, meta, dist in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "metadata": meta,
                "distance": dist,
            }
        )
    return hits
