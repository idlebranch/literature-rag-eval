from functools import lru_cache
import threading
from typing import List
from sentence_transformers import SentenceTransformer
from src.config import settings


_model_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(
        model_name,
        local_files_only=settings.embedding_local_files_only,
    )


def get_embedding_model() -> SentenceTransformer:
    """Load each configured embedding model at most once, including first-use races."""
    with _model_lock:
        return _load_embedding_model(settings.embedding_model)


# Preserve the cache inspection API used by health checks and tests.
get_embedding_model.cache_info = _load_embedding_model.cache_info  # type: ignore[attr-defined]
get_embedding_model.cache_clear = _load_embedding_model.cache_clear  # type: ignore[attr-defined]


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    return embed_texts([query])[0]
