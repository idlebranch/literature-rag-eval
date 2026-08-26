"""Build fixed_270 and section_aware_270 Dense + Sparse indexes on final_corpus.

Two isolated indexes, identical except for the chunking strategy:

    fixed_270          -> chroma_db_fixed_270 / sparse_index_fixed_270
    section_aware_270  -> chroma_db_section_aware_270 / sparse_index_section_aware_270

Each build: fresh-delete the target dirs -> chunk (mode-dependent) -> apply the
index-eligibility policy (exclude references/acknowledgments from the index but
keep them in the raw chunks) -> dense Chroma -> BGE-M3 sparse index -> sanity
queries. Writes data/processed/index_build_report.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.ingest import ingest, load_paper_meta
from src.vectorstore import clear_vectorstore_caches, search
from src.sparse_index import build_sparse_index, load_index, clear_index_cache, sparse_index_dir
from src.embedder import embed_query

FINAL_CORPUS = ROOT / "data" / "papers" / "final_corpus"
REPORT = ROOT / "data" / "processed" / "index_build_report.md"

INDEXES = {
    "fixed_270_gpu": "fixed",
    "section_aware_270_gpu": "section_aware",
}


def preflight() -> None:
    """Fail loudly unless running under the project .venv on the RTX 5060 GPU."""
    import torch

    exe = Path(sys.executable).resolve()
    if ".venv" not in exe.parts:
        raise RuntimeError(f"Must run with the project .venv; got {exe}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available (CPU-only torch?) — refusing to build on CPU")
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != "NVIDIA GeForce RTX 5060 Laptop GPU":
        raise RuntimeError(f"Unexpected GPU: {gpu_name!r}")

    from src.embedder import get_embedding_model
    model = get_embedding_model()
    if not str(model.device).startswith("cuda"):
        raise RuntimeError(f"Embedding model on {model.device}, expected cuda:0")

    print(f"PREFLIGHT OK: {exe}")
    print(f"  torch={torch.__version__} cuda={torch.version.cuda} device={model.device}")
    print(f"  GPU={gpu_name} batch_size=32")

SANITY_QUERIES = [
    ("adsorption", "adsorption of organic pollutants onto activated carbon"),
    ("PMS/AOP", "peroxymonosulfate PMS activation for antibiotic degradation"),
    ("membrane", "membrane fouling mechanisms and control in water treatment"),
    ("biological", "biological nitrogen removal via anammox in wastewater"),
    ("emerging contaminant", "removal of emerging contaminants and PFAS from drinking water"),
]


def _write_progress(progress_file: str | None, rec: dict) -> None:
    if not progress_file:
        return
    rec = dict(rec)
    rec["updated_at"] = time.time()
    Path(progress_file).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")


def _build_one(name: str, mode: str, paper_meta: dict, progress_file: str | None = None) -> dict:
    chroma_dir = ROOT / f"chroma_db_{name}"
    sparse_dir = ROOT / f"sparse_index_{name}"

    # Fresh, deterministic rebuild: no cross-index overwrite, no duplicate writes.
    shutil.rmtree(chroma_dir, ignore_errors=True)
    shutil.rmtree(sparse_dir, ignore_errors=True)

    settings.chroma_dir = str(chroma_dir)
    settings.collection_name = name
    settings.sparse_index_dir = str(sparse_dir)
    settings.chunking_mode = mode
    settings.pdf_dir = str(FINAL_CORPUS)

    clear_vectorstore_caches()
    clear_index_cache()

    print(f"\n=== building {name} (mode={mode}) ===")
    _write_progress(progress_file, {"stage": "dense", "progress_current": 0, "progress_total": 1})
    dense_started = time.perf_counter()
    stats = ingest(
        paper_meta=paper_meta,
        progress_callback=(lambda rec: _write_progress(progress_file, rec)) if progress_file else None,
    )
    dense_seconds = round(time.perf_counter() - dense_started, 1)
    _write_progress(progress_file, {"stage": "sparse", "progress_current": 0, "progress_total": 1})

    # Free the dense model from VRAM before loading the separate sparse encoder,
    # so both phases fit sequentially in the 8GB RTX 5060 (no model/config change).
    import gc
    import torch as _torch
    from src import embedder as _embedder
    _embedder.get_embedding_model.cache_clear()
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()

    sparse_started = time.perf_counter()
    index = build_sparse_index(verbose=True)
    index.save(
        sparse_dir,
        manifest_extra={"builder": "scripts.build_indexes", "index_name": name,
                        "chunking_mode": mode},
    )
    sparse_seconds = round(time.perf_counter() - sparse_started, 1)
    _write_progress(progress_file, {"stage": "completed", "progress_current": 1, "progress_total": 1})

    stats["dense_seconds"] = dense_seconds
    stats["sparse_seconds"] = sparse_seconds
    stats["sparse_count"] = index.chunk_count
    stats["chroma_dir"] = str(chroma_dir)
    stats["sparse_index_dir"] = str(sparse_dir)

    # sanity: dense queries + sparse loadability
    clear_vectorstore_caches()
    query_results = []
    for label, q in SANITY_QUERIES:
        emb = embed_query(q)
        hits = search(emb, top_k=5)
        query_results.append((label, len(hits), hits[0]["metadata"].get("source", "") if hits else ""))
    stats["sanity"] = query_results
    stats["sparse_load_ok"] = _sparse_ok()
    return stats


def _sparse_ok() -> bool:
    try:
        load_index(strict=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [sparse load check] {e}")
        return False


def _paper_coverage_matches(a: dict, b: dict) -> bool:
    return a.get("paper_ids") == b.get("paper_ids")


def _render(stats_by_index: dict) -> str:
    lines = ["# Index Build Report", "",
             f"Corpus: `{FINAL_CORPUS}`",
             f"Embedding model: `{settings.embedding_model}` (BGE-M3 dense + sparse)",
             "chunk_size=1000, overlap=150; retrieval policy excludes references/acknowledgments.", ""]

    for name, s in stats_by_index.items():
        lines += [
            f"## {name}",
            "",
            f"- PDF count: **{s['pdf_count']}**",
            f"- raw chunk count: **{s['raw_chunks']}**",
            f"- indexable chunk count: **{s['indexable_chunks']}**",
            f"- Dense count: **{s['dense_count']}**",
            f"- Sparse count: **{s['sparse_count']}**",
            f"- excluded References count: **{s['excluded_references']}**",
            f"- excluded Acknowledgments count: **{s['excluded_acknowledgments']}**",
            f"- build time: dense={s['dense_seconds']}s sparse={s['sparse_seconds']}s",
            f"- chroma dir: `{s['chroma_dir']}`",
            f"- sparse index dir: `{s['sparse_index_dir']}`",
            f"- sparse index loads cleanly: **{s['sparse_load_ok']}**",
            "",
        ]

    a, b = stats_by_index["fixed_270_gpu"], stats_by_index["section_aware_270_gpu"]
    lines += [
        "## Cross-index checks",
        "",
        f"- paper_id coverage identical: **{_paper_coverage_matches(a, b)}** "
        f"({a['pdf_count']} == {b['pdf_count']})",
        f"- failed papers: **{0 if a['pdf_count'] == 270 and b['pdf_count'] == 270 else 'CHECK'}**",
        f"- stale sparse index: **{not a['sparse_load_ok'] or not b['sparse_load_ok']}**",
        "",
        "## Sanity queries (dense top-5, both indexes)",
        "",
    ]
    for name, s in stats_by_index.items():
        lines.append(f"### {name}")
        for label, n_hits, top_source in s["sanity"]:
            lines.append(f"- `{label}` -> {n_hits} hits, top source `{top_source}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(INDEXES), default=None,
                    help="build a single index (e.g. section_aware_270_gpu)")
    ap.add_argument("--progress-file", default=str(ROOT / "data" / "processed" / "build_progress.json"),
                    help="write per-batch progress JSON for the watchdog")
    args = ap.parse_args()

    preflight()
    meta = load_paper_meta()
    names = [args.only] if args.only else list(INDEXES)
    stats_by_index = {name: _build_one(name, INDEXES[name], meta, args.progress_file)
                      for name in names}

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if len(stats_by_index) == 2:
        REPORT.write_text(_render(stats_by_index), encoding="utf-8")
        print(f"\nWrote {REPORT}")
        a, b = stats_by_index["fixed_270_gpu"], stats_by_index["section_aware_270_gpu"]
        print(f"paper_id coverage identical: {_paper_coverage_matches(a, b)} "
              f"({a['pdf_count']} vs {b['pdf_count']})")
    for name, s in stats_by_index.items():
        print(f"{name}: dense={s['dense_count']} sparse={s['sparse_count']} "
              f"pdfs={s['pdf_count']} excl_refs={s['excluded_references']} "
              f"excl_ack={s['excluded_acknowledgments']}")


if __name__ == "__main__":
    main()
