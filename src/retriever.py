from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import re
import threading
import time
from typing import Callable, Dict, List
from src.config import settings
from src.embedder import embed_query
from src.vectorstore import search

RETRIEVAL_MODES = ("dense_only", "hybrid_dense_sparse", "hybrid_reranker")


_cache_lock = threading.RLock()
_embedding_inference_lock = threading.Lock()
_embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
_retrieval_cache: OrderedDict[tuple[str, int, str], tuple[List[Dict], int]] = OrderedDict()


@dataclass
class RetrievalResult:
    hits: List[Dict]
    expanded_query: str
    query_rewrite_ms: float
    query_embedding_ms: float
    chroma_search_ms: float
    filter_diversify_ms: float
    embedding_cache_hit: bool | None
    retrieval_cache_hit: bool
    raw_hit_count: int
    context_estimated_tokens: int
    collection_version: str
    retrieval_mode: str = "dense_only"
    sparse_search_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0


def expand_query(query: str) -> str:
    """Rule-based bilingual query expansion for environmental RAG."""
    q = query
    lower_q = query.lower()

    expansions = []

    if "高级氧化" in q or "aop" in lower_q:
        expansions.append(
            "advanced oxidation processes AOPs emerging contaminants micropollutants hydroxyl radical sulfate radical reactive oxygen species"
        )

    if "pms" in lower_q or "pds" in lower_q or "过硫酸" in q or "单过硫酸" in q:
        expansions.append(
            "peroxymonosulfate PMS peroxydisulfate PDS persulfate activation sulfate radical SO4 hydroxyl radical singlet oxygen antibiotics bisphenol A BPA"
        )

    if "臭氧" in q or "ozone" in lower_q or "ozonation" in lower_q:
        expansions.append(
            "ozonation catalytic ozonation ozone O3 hydroxyl radical micropollutants emerging contaminants water treatment"
        )

    if "光催化" in q or "photocatal" in lower_q:
        expansions.append(
            "photocatalysis TiO2 g-C3N4 visible light pharmaceuticals personal care products PPCPs degradation"
        )

    if "fenton" in lower_q or "芬顿" in q:
        expansions.append(
            "Fenton photo-Fenton heterogeneous Fenton iron catalyst hydroxyl radical emerging contaminants wastewater"
        )

    if "pfas" in lower_q or "pfoa" in lower_q or "pfos" in lower_q or "全氟" in q:
        expansions.append(
            "PFAS PFOA PFOS per- and polyfluoroalkyl substances removal treatment adsorption membrane ion exchange activated carbon destruction limitations full-scale application"
        )

    if "新污染物" in q:
        expansions.append(
            "emerging contaminants micropollutants PPCPs pharmaceuticals endocrine disrupting compounds PFAS antibiotics water treatment"
        )

    if expansions:
        return query + "\n" + "\n".join(expansions)

    return query


def diversify_hits(hits: List[Dict], top_k: int, max_per_source: int = 2) -> List[Dict]:
    """Deduplicate overlapping neighbours and preserve source diversity."""
    selected = []
    source_count = {}

    for hit in hits:
        metadata = hit.get("metadata") or {}
        source = str(metadata.get("source", ""))
        count = source_count.get(source, 0)

        if count < max_per_source and not _is_adjacent_duplicate(hit, selected):
            selected.append(hit)
            source_count[source] = count + 1

        if len(selected) >= top_k:
            break

    return selected


def _text_shingles(text: str, width: int = 3) -> set[str]:
    normalized = re.sub(r"\s+", "", text).casefold()
    if len(normalized) <= width:
        return {normalized} if normalized else set()
    return {normalized[i : i + width] for i in range(len(normalized) - width + 1)}


def _is_adjacent_duplicate(candidate: Dict, selected: List[Dict]) -> bool:
    meta = candidate.get("metadata") or {}
    source = str(meta.get("source", ""))
    try:
        chunk_index = int(meta.get("chunk_index"))
    except (TypeError, ValueError):
        return False
    candidate_shingles: set[str] | None = None
    for existing in selected:
        existing_meta = existing.get("metadata") or {}
        if str(existing_meta.get("source", "")) != source:
            continue
        try:
            existing_index = int(existing_meta.get("chunk_index"))
        except (TypeError, ValueError):
            continue
        if abs(existing_index - chunk_index) > 1:
            continue
        candidate_shingles = candidate_shingles or _text_shingles(str(candidate.get("text", "")))
        existing_shingles = _text_shingles(str(existing.get("text", "")))
        union = candidate_shingles | existing_shingles
        similarity = len(candidate_shingles & existing_shingles) / len(union) if union else 1.0
        if similarity >= 0.90:
            return True
    return False


def _estimate_tokens(text: str) -> int:
    """Conservative language-aware estimate used only for the context budget."""
    ascii_count = sum(1 for char in text if ord(char) < 128)
    non_ascii_count = len(text) - ascii_count
    return max(1, math.ceil(ascii_count / 4 + non_ascii_count / 1.5))


def _apply_context_budget(hits: List[Dict], token_budget: int) -> tuple[List[Dict], int]:
    selected: List[Dict] = []
    used = 0
    for hit in hits:
        meta = hit.get("metadata") or {}
        block = (
            f"source={meta.get('source', '')} page={meta.get('page', '')} "
            f"chunk={meta.get('chunk_index', '')}\n{hit.get('text', '')}"
        )
        tokens = _estimate_tokens(block)
        # Keep chunks whole so citations never point at an arbitrary string cut.
        if selected and used + tokens > token_budget:
            continue
        selected.append(hit)
        used += tokens
    return selected, used


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().split()).casefold()


def collection_version() -> str:
    sqlite_file = Path(settings.chroma_dir) / "chroma.sqlite3"
    try:
        stat = sqlite_file.stat()
        disk_version = f"{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        disk_version = "missing"
    return f"{settings.collection_name}:{disk_version}"


def clear_retrieval_caches() -> None:
    """Clear only in-memory query caches; the Chroma index is untouched."""
    with _cache_lock:
        _embedding_cache.clear()
        _retrieval_cache.clear()


def _cached_embedding(expanded_query: str) -> tuple[List[float], bool]:
    key = _normalize_query(expanded_query)
    with _cache_lock:
        cached = _embedding_cache.get(key)
        if cached is not None:
            _embedding_cache.move_to_end(key)
            return list(cached), True
    with _embedding_inference_lock:
        # Recheck after waiting: another request may have filled the cache.
        with _cache_lock:
            cached = _embedding_cache.get(key)
            if cached is not None:
                _embedding_cache.move_to_end(key)
                return list(cached), True
        vector = embed_query(expanded_query)
        with _cache_lock:
            _embedding_cache[key] = list(vector)
            _embedding_cache.move_to_end(key)
            while len(_embedding_cache) > settings.query_cache_size:
                _embedding_cache.popitem(last=False)
        return vector, False


def _inherit_dense_distance(fused_hits: List[Dict]) -> List[Dict]:
    """Give sparse-only fused hits a dense-distance value for the evidence gate.

    The evidence gate in ``rag_chain`` is calibrated on Chroma dense distance.
    Sparse-only candidates have no dense distance of their own, so they inherit
    the best (minimum) dense distance present in the fused pool. The gate
    therefore keeps behaving exactly as the dense pool would decide it; sparse
    additions can only add candidates, never loosen the threshold.
    """
    dense_distances = [
        float(hit["distance"]) for hit in fused_hits if hit.get("distance") is not None
    ]
    if not dense_distances:
        return fused_hits
    fallback = min(dense_distances)
    for hit in fused_hits:
        if hit.get("distance") is None:
            hit["distance"] = fallback
    return fused_hits


def _retrieve_hybrid(
    query: str,
    expanded_query: str,
    top_k: int,
    *,
    rerank: bool,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[List[Dict], float, float, float, float, int]:
    """Run the hybrid pipeline. Returns (diversified_hits, dense_ms, sparse_ms,
    fusion_ms, rerank_ms, raw_hit_count). Raises on missing prerequisites."""
    from src.fusion import resolve_sparse_hits, rrf_fuse
    from src.sparse_encoder import encode_query_sparse
    from src.sparse_index import load_index

    if progress_callback:
        progress_callback("chroma")
    dense_started = time.perf_counter()
    query_embedding, _ = _cached_embedding(expanded_query)
    dense_hits = search(query_embedding, top_k=settings.hybrid_dense_k)
    dense_ms = (time.perf_counter() - dense_started) * 1000

    if progress_callback:
        progress_callback("sparse")
    sparse_started = time.perf_counter()
    index = load_index(strict=True)
    query_weights = encode_query_sparse(expanded_query)
    sparse_scored = index.search(query_weights, top_k=settings.hybrid_sparse_k)
    sparse_hits = resolve_sparse_hits(sparse_scored)
    sparse_ms = (time.perf_counter() - sparse_started) * 1000

    if progress_callback:
        progress_callback("fusion")
    fusion_started = time.perf_counter()
    fused = rrf_fuse(dense_hits, sparse_hits, fusion_k=settings.hybrid_fusion_k)
    fused = _inherit_dense_distance(fused)
    fusion_ms = (time.perf_counter() - fusion_started) * 1000

    rerank_ms = 0.0
    candidates = fused
    if rerank:
        if progress_callback:
            progress_callback("rerank")
        rerank_started = time.perf_counter()
        from src.reranker import get_reranker

        reranker = get_reranker()
        # The original information need drives cross-encoder scoring; the rule
        # expansion is kept for the lexical/dense recall paths only.
        candidates = reranker.rerank(query, fused, final_k=settings.reranker_final_k)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000

    if progress_callback:
        progress_callback("filter")
    diversified = diversify_hits(candidates, top_k=top_k, max_per_source=2)
    return diversified, dense_ms, sparse_ms, fusion_ms, rerank_ms, len(candidates)


def retrieve_with_metrics(
    query: str,
    top_k: int | None = None,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> RetrievalResult:
    top_k = top_k or settings.top_k
    mode = settings.retrieval_mode
    if mode not in RETRIEVAL_MODES:
        raise ValueError(
            f"Unknown RETRIEVAL_MODE {mode!r}; expected one of {RETRIEVAL_MODES}"
        )
    rewrite_started = time.perf_counter()
    expanded_query = expand_query(query)
    rewrite_ms = (time.perf_counter() - rewrite_started) * 1000
    version = collection_version()
    retrieval_key = (_normalize_query(query), top_k, mode, version)

    with _cache_lock:
        cached = _retrieval_cache.get(retrieval_key)
        if cached is not None:
            if progress_callback:
                progress_callback("retrieval_cache_hit")
            _retrieval_cache.move_to_end(retrieval_key)
            cached_hits, raw_hit_count = deepcopy(cached)
            context_hits, context_tokens = _apply_context_budget(
                cached_hits, settings.context_token_budget
            )
            return RetrievalResult(
                hits=context_hits,
                expanded_query=expanded_query,
                query_rewrite_ms=rewrite_ms,
                query_embedding_ms=0.0,
                chroma_search_ms=0.0,
                filter_diversify_ms=0.0,
                embedding_cache_hit=None,
                retrieval_cache_hit=True,
                raw_hit_count=raw_hit_count,
                context_estimated_tokens=context_tokens,
                collection_version=version,
                retrieval_mode=mode,
            )

    embedding_ms = 0.0
    sparse_ms = fusion_ms = rerank_ms = 0.0

    if mode == "dense_only":
        if progress_callback:
            progress_callback("embedding")
        embedding_started = time.perf_counter()
        query_embedding, embedding_hit = _cached_embedding(expanded_query)
        embedding_ms = (time.perf_counter() - embedding_started) * 1000

        if progress_callback:
            progress_callback("chroma")
        search_started = time.perf_counter()
        raw_hits = search(query_embedding, top_k=max(top_k * 4, 20))
        search_ms = (time.perf_counter() - search_started) * 1000

        if progress_callback:
            progress_callback("filter")
        filter_started = time.perf_counter()
        diversified = diversify_hits(raw_hits, top_k=top_k, max_per_source=2)
        filter_ms = (time.perf_counter() - filter_started) * 1000
        raw_hit_count = len(raw_hits)
        search_phase_ms = search_ms
    else:
        (
            diversified,
            search_phase_ms,
            sparse_ms,
            fusion_ms,
            rerank_ms,
            raw_hit_count,
        ) = _retrieve_hybrid(
            query,
            expanded_query,
            top_k,
            rerank=(mode == "hybrid_reranker"),
            progress_callback=progress_callback,
        )
        embedding_hit = None
        filter_ms = 0.0

    with _cache_lock:
        _retrieval_cache[retrieval_key] = (deepcopy(diversified), raw_hit_count)
        _retrieval_cache.move_to_end(retrieval_key)
        while len(_retrieval_cache) > settings.retrieval_cache_size:
            _retrieval_cache.popitem(last=False)

    context_hits, context_tokens = _apply_context_budget(
        diversified, settings.context_token_budget
    )
    return RetrievalResult(
        hits=context_hits,
        expanded_query=expanded_query,
        query_rewrite_ms=rewrite_ms,
        query_embedding_ms=embedding_ms,
        chroma_search_ms=search_phase_ms,
        filter_diversify_ms=filter_ms,
        embedding_cache_hit=embedding_hit if mode == "dense_only" else None,
        retrieval_cache_hit=False,
        raw_hit_count=raw_hit_count,
        context_estimated_tokens=context_tokens,
        collection_version=version,
        retrieval_mode=mode,
        sparse_search_ms=sparse_ms,
        fusion_ms=fusion_ms,
        rerank_ms=rerank_ms,
    )


def retrieve(
    query: str,
    top_k: int | None = None,
    *,
    with_metrics: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> List[Dict] | RetrievalResult:
    result = retrieve_with_metrics(
        query,
        top_k=top_k,
        progress_callback=progress_callback,
    )
    return result if with_metrics else result.hits


def format_context(hits: List[Dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        blocks.append(
            f"[S{i}]\n"
            f"document: {meta.get('source', '')}\n"
            f"page: {meta.get('page', '')}\n"
            f"chunk_id: {meta.get('chunk_index', '')}\n"
            "content:\n"
            f"{hit['text']}\n"
            f"[/S{i}]"
        )
    return "\n\n".join(blocks)
