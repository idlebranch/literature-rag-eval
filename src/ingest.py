"""Ingestion pipeline: PDF -> chunks -> (optional) index eligibility -> embeddings -> Chroma.

Chunking strategy follows ``settings.chunking_mode``:

- ``fixed``          -> legacy per-page fixed-size chunker (load_pdfs).
- ``section_aware``  -> line-preserving page extraction (load_pdf_pages) +
                        section-aware page-traceable chunking.

Chunk metadata is enriched to a uniform schema (paper_id, title, doi, section,
page_start, page_end, chunk_id, chunking_mode) and then the *index eligibility*
policy excludes low-value sections (references / acknowledgments) from the
Dense/Sparse index while keeping them in the raw chunk list.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from src.chunking import build_chunks, is_indexable
from src.config import settings
from src.embedder import embed_texts
from src.pdf_loader import load_pdf_pages, load_pdfs
from src.utils.logging import get_logger
from src.vectorstore import add_chunks

logger = get_logger(__name__)

REQUIRED_METADATA = (
    "paper_id", "title", "doi", "section",
    "page_start", "page_end", "chunk_id", "chunking_mode",
)


def load_paper_meta(manifest_path: str | None = None) -> Dict[str, Dict[str, str]]:
    """Read final_paper_manifest.csv -> {key: {"title", "doi"}}.

    Keys are both the full filename (``final_file``) and its stem, so chunks
    keyed by paper_id (stem) or source (filename) both resolve.
    """
    path = Path(manifest_path or settings.paper_manifest_path)
    meta: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        logger.warning("Manifest not found: %s (title/doi will be empty)", path)
        return meta
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ff = (row.get("final_file") or "").strip()
            if not ff:
                continue
            info = {"title": row.get("title") or "", "doi": row.get("doi") or ""}
            meta[ff] = info
            meta[Path(ff).stem] = info
    return meta


def load_pages(pdf_dir: str, mode: str) -> List[Dict]:
    """Extract page records using the loader appropriate for ``mode``."""
    if mode == "section_aware":
        records: List[Dict] = []
        for pdf in sorted(Path(pdf_dir).rglob("*.pdf")):
            records.extend(load_pdf_pages(str(pdf)))
        return records
    return load_pdfs(pdf_dir)


def enrich_chunks(chunks: List[Dict], paper_meta: Dict[str, Dict[str, str]],
                  mode: str) -> List[Dict]:
    """Ensure every chunk carries the uniform metadata schema.

    ``fixed`` chunks only carry paper_id/source/page/chunk_index, so title, doi,
    section, page_start, page_end, chunk_id and chunking_mode are filled here.
    The chunk text and ids are never modified.
    """
    for c in chunks:
        m = c.get("metadata") or {}
        pid = m.get("paper_id", "")
        info = paper_meta.get(pid) or paper_meta.get(m.get("source", "")) or {}
        m.setdefault("title", info.get("title", "") or "")
        m.setdefault("doi", info.get("doi", "") or "")
        m.setdefault("section", "")
        if "page_start" not in m:
            m["page_start"] = m.get("page")
        if "page_end" not in m:
            m["page_end"] = m.get("page")
        m.setdefault("chunk_index", 0)
        m["chunk_id"] = c["id"]
        m["chunking_mode"] = mode
        c["metadata"] = m
    return chunks


def partition_indexable(chunks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split chunks into (indexable, excluded) by the section policy.

    Excluded chunks (references / acknowledgments) are NOT deleted: they remain
    in the raw chunk list / manifest; only the Dense+Sparse index drops them.
    """
    indexable: List[Dict] = []
    excluded: List[Dict] = []
    for c in chunks:
        section = (c.get("metadata") or {}).get("section", "")
        if is_indexable(section):
            indexable.append(c)
        else:
            excluded.append(c)
    return indexable, excluded


def ingest(pdf_dir: str | None = None, mode: str | None = None,
           paper_meta: Dict[str, Dict[str, str]] | None = None,
           progress_callback=None) -> Dict:
    """Run dense ingestion for the current settings; returns build stats.

    Does NOT build the sparse index (call build_sparse_index afterwards so it
    reads the freshly-written Chroma collection).

    ``progress_callback`` (optional) is invoked after each embedding batch with
    {"stage": "dense", "progress_current", "progress_total", "batch_latency"}.
    """
    pdf_dir = pdf_dir or settings.pdf_dir
    mode = mode or settings.chunking_mode
    if paper_meta is None:
        paper_meta = load_paper_meta()

    started = time.perf_counter()
    pages = load_pages(pdf_dir, mode)
    logger.info("Loaded pages: %d", len(pages))
    if not pages:
        raise RuntimeError("No extractable PDF text found. Check PDF_DIR.")

    chunks = build_chunks(pages, chunking_mode=mode, paper_meta=paper_meta)
    chunks = enrich_chunks(chunks, paper_meta, mode)
    indexable, excluded = partition_indexable(chunks)
    logger.info("chunks total=%d indexable=%d excluded=%d",
                len(chunks), len(indexable), len(excluded))

    texts = [c["text"] for c in indexable]
    embeddings: List[List[float]] = []
    batch_size = 32
    total_batches = -(-len(texts) // batch_size) if texts else 0
    for bi, i in enumerate(tqdm(range(0, len(texts), batch_size), desc="Embedding")):
        b_started = time.perf_counter()
        embeddings.extend(embed_texts(texts[i:i + batch_size]))
        if progress_callback:
            progress_callback({
                "stage": "dense",
                "progress_current": bi + 1,
                "progress_total": total_batches,
                "batch_latency": round(time.perf_counter() - b_started, 3),
            })

    add_chunks(indexable, embeddings, reset=True)

    paper_ids = sorted({p["paper_id"] for p in pages})
    return {
        "pdf_count": len(paper_ids),
        "paper_ids": paper_ids,
        "raw_chunks": len(chunks),
        "indexable_chunks": len(indexable),
        "excluded_total": len(excluded),
        "excluded_references": _count_excluded(excluded, "references"),
        "excluded_acknowledgments": _count_excluded(excluded, "acknowledgments"),
        "dense_count": len(indexable),
        "build_seconds": round(time.perf_counter() - started, 1),
    }


def _count_excluded(excluded: List[Dict], section: str) -> int:
    return sum(
        1 for c in excluded
        if ((c.get("metadata") or {}).get("section") or "").strip().lower() == section
    )


def main() -> None:
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    stats = ingest()
    logger.info("Done. Dense collection built.")
    logger.info("Collection name: %s", settings.collection_name)
    logger.info("Chroma dir: %s", settings.chroma_dir)
    logger.info("Stats: %s", stats)


if __name__ == "__main__":
    main()
