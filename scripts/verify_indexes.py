"""Post-build verification of the two Dense+Sparse indexes.

For fixed_270 and section_aware_270, verify:
  - Dense Chroma count == Sparse index chunk count.
  - paper_id coverage matches across the two indexes (expect 270).
  - No references/acknowledgments chunk leaked into the Dense index.
  - Sparse index loads cleanly against the Dense collection version (not stale).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings
from src.sparse_index import clear_index_cache, load_index
from src.vectorstore import clear_vectorstore_caches, get_collection

INDEXES = {"fixed_270_gpu": "fixed", "section_aware_270_gpu": "section_aware"}
EXCLUDED = {"references", "acknowledgments"}


def verify_one(name: str) -> dict:
    settings.chroma_dir = str(ROOT / f"chroma_db_{name}")
    settings.collection_name = name
    settings.sparse_index_dir = str(ROOT / f"sparse_index_{name}")
    clear_vectorstore_caches()
    clear_index_cache()

    collection = get_collection()
    dense_count = collection.count()
    sparse = load_index(strict=True)
    sparse_count = sparse.chunk_count

    paper_ids: set[str] = set()
    leaked = 0
    batch = 512
    for offset in range(0, dense_count, batch):
        got = collection.get(limit=batch, offset=offset, include=["metadatas"])
        for meta in got["metadatas"]:
            paper_ids.add(meta.get("paper_id"))
            if (meta.get("section") or "").strip().lower() in EXCLUDED:
                leaked += 1

    return {
        "name": name,
        "dense_count": dense_count,
        "sparse_count": sparse_count,
        "dense_sparse_match": dense_count == sparse_count,
        "paper_ids": paper_ids,
        "excluded_leaked": leaked,
    }


def main() -> int:
    results = {name: verify_one(name) for name in INDEXES}
    for r in results.values():
        print(f"{r['name']}: dense={r['dense_count']} sparse={r['sparse_count']} "
              f"match={r['dense_sparse_match']} papers={len(r['paper_ids'])} "
              f"refs_acks_in_index={r['excluded_leaked']}")

    a, b = results["fixed_270_gpu"], results["section_aware_270_gpu"]
    coverage_identical = a["paper_ids"] == b["paper_ids"]
    print(f"paper_id coverage identical: {coverage_identical} "
          f"({len(a['paper_ids'])} vs {len(b['paper_ids'])})")

    ok = (
        all(r["dense_sparse_match"] for r in results.values())
        and coverage_identical
        and len(a["paper_ids"]) == 270
        and all(r["excluded_leaked"] == 0 for r in results.values())
    )
    print("VERDICT:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
