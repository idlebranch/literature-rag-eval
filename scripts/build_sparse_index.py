"""Build the BGE-M3 sparse lexical index from the current Chroma collection.

Usage:
    python -m scripts.build_sparse_index

Writes index.npz + manifest.json into the configured SPARSE_INDEX_DIR
(default: sparse_index/). Never touches the Chroma collection itself.
"""
from __future__ import annotations

from src.config import settings
from src.sparse_index import build_sparse_index, sparse_index_dir


def main() -> None:
    print(f"Building sparse index into {sparse_index_dir()}")
    index = build_sparse_index()
    index.save(
        sparse_index_dir(),
        manifest_extra={"builder": "scripts.build_sparse_index"},
    )
    print(
        f"Done: {index.chunk_count} chunks, "
        f"{index.posting_count} postings, "
        f"{len(index.token_ids)} unique terms."
    )


if __name__ == "__main__":
    main()
