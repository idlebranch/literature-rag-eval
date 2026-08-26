from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Explicit process environment variables win over local .env values. This is
# required for isolated candidate-index builds without editing the active config.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _project_path(env_name: str, default: str) -> str:
    configured = Path(os.getenv(env_name, default)).expanduser()
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    return str(configured.resolve())


@dataclass
class Settings:
    project_root: str = str(PROJECT_ROOT)

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_local_files_only: bool = _env_bool("EMBEDDING_LOCAL_FILES_ONLY", True)

    # Production demo defaults are the frozen final acceptance pipeline. Explicit
    # process environment values still win for isolated, non-production index work.
    chroma_dir: str = _project_path("CHROMA_DIR", "chroma_db_section_aware_270_gpu")
    collection_name: str = os.getenv("COLLECTION_NAME", "section_aware_270_gpu")

    top_k: int = int(os.getenv("TOP_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    # "fixed" keeps the legacy per-page fixed-size chunker; "section_aware"
    # enables section-aware + page-traceable chunking with fixed fallback.
    chunking_mode: str = os.getenv("CHUNKING_MODE", "section_aware")
    section_min_chunk: int = int(os.getenv("SECTION_MIN_CHUNK", "30"))
    max_retrieval_distance: float = float(os.getenv("MAX_RETRIEVAL_DISTANCE", "1.15"))
    context_token_budget: int = int(os.getenv("CONTEXT_TOKEN_BUDGET", "3000"))
    query_cache_size: int = int(os.getenv("QUERY_CACHE_SIZE", "128"))
    retrieval_cache_size: int = int(os.getenv("RETRIEVAL_CACHE_SIZE", "128"))

    # Retrieval pipeline mode. dense_only keeps the historical behavior;
    # hybrid_dense_sparse and hybrid_reranker fail loudly when their
    # prerequisites are missing instead of silently falling back to dense.
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid_dense_sparse")
    hybrid_dense_k: int = int(os.getenv("HYBRID_DENSE_K", "25"))
    hybrid_sparse_k: int = int(os.getenv("HYBRID_SPARSE_K", "25"))
    hybrid_fusion_k: int = int(os.getenv("HYBRID_FUSION_K", "25"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    sparse_index_dir: str = _project_path("SPARSE_INDEX_DIR", "sparse_index_section_aware_270_gpu")
    sparse_max_length: int = int(os.getenv("SPARSE_MAX_LENGTH", "512"))
    bge_m3_local_dir: str = os.getenv("BGE_M3_LOCAL_DIR", "")

    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranker_final_k: int = int(os.getenv("RERANKER_FINAL_K", "8"))
    reranker_batch_size: int = int(os.getenv("RERANKER_BATCH_SIZE", "16"))
    reranker_max_length: int = int(os.getenv("RERANKER_MAX_LENGTH", "512"))

    llm_connect_timeout: float = float(os.getenv("LLM_CONNECT_TIMEOUT", "10"))
    llm_read_timeout: float = float(os.getenv("LLM_READ_TIMEOUT", "120"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "1"))

    pdf_dir: str = _project_path("PDF_DIR", "data/papers/final_corpus")
    output_dir: str = _project_path("OUTPUT_DIR", "outputs")
    # CSV manifest mapping final_file -> title/doi; used to enrich chunk metadata.
    paper_manifest_path: str = _project_path("PAPER_MANIFEST", "data/papers/final_paper_manifest.csv")

    api_host: str = os.getenv("RAG_API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("RAG_API_PORT", "8010"))
    ui_port: int = int(os.getenv("RAG_UI_PORT", "8501"))
    eval_port: int = int(os.getenv("RAG_EVAL_PORT", "8502"))
    tracing_enabled: bool = _env_bool("TRACING_ENABLED", False)
    performance_debug: bool = _env_bool("PERFORMANCE_DEBUG", False)


settings = Settings()
