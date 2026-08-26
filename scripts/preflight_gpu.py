"""Preflight checks for GPU long tasks (embedding / sparse / reranker / benchmark).

Run before any GPU build. Read-only: never fixes the environment, only checks.

Hard fail (non-zero exit):
  1. sys.executable must be the project .venv python.
  2. torch.cuda.is_available() and torch.version.cuda must be present.
  3. GPU must be "NVIDIA GeForce RTX 5060 Laptop GPU".
  4. BGE-M3 model smoke test (device cuda:0, embedding shape, no CUDA error).
  5. final_corpus + final_paper_manifest.csv must exist.

Warnings (still exit 0):
  - corpus PDF count != expected frozen count.
  - low disk space.
  - target index path already exists (would be overwritten).

Usage:
    .venv/Scripts/python.exe scripts/preflight_gpu.py [--index NAME] [--expected-pdfs 270]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = (ROOT / ".venv" / "Scripts" / "python.exe").resolve()
EXPECTED_GPU = "NVIDIA GeForce RTX 5060 Laptop GPU"


def _fail(msg: str) -> "NoReturn":  # noqa: F821
    print(f"PREFLIGHT FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def _check_python() -> None:
    exe = Path(sys.executable).resolve()
    if exe != VENV_PYTHON:
        _fail(f"wrong interpreter {exe} (expected {VENV_PYTHON})")
    print(f"[ok] python: {exe}")


def _check_torch() -> None:
    import torch
    if not torch.cuda.is_available():
        _fail("torch.cuda.is_available() == False (CPU-only torch?)")
    if torch.version.cuda is None:
        _fail("torch.version.cuda is None")
    gpu = torch.cuda.get_device_name(0)
    if gpu != EXPECTED_GPU:
        _fail(f"unexpected GPU: {gpu!r} (expected {EXPECTED_GPU!r})")
    print(f"[ok] torch={torch.__version__} cuda={torch.version.cuda} gpu={gpu}")


def _check_model() -> None:
    import torch
    from src.embedder import embed_texts, get_embedding_model

    model = get_embedding_model()
    dev = str(model.device)
    if not dev.startswith("cuda"):
        _fail(f"model on {dev}, expected cuda:0")
    try:
        vecs = embed_texts(["adsorption of organic pollutants",
                            "membrane fouling control",
                            "PFAS removal from water"])
        torch.cuda.synchronize()  # surface any async CUDA error
    except Exception as e:  # noqa: BLE001
        _fail(f"model smoke test raised: {e}")
    dims = {len(v) for v in vecs}
    if dims != {1024}:
        _fail(f"unexpected embedding dims {dims}")
    print(f"[ok] model device={dev} shape=[{len(vecs)},1024] smoke passed")


def _check_data(args: argparse.Namespace) -> None:
    corpus = Path(args.corpus)
    manifest = Path(args.manifest)
    if not corpus.is_dir():
        _fail(f"final_corpus missing: {corpus}")
    if not manifest.is_file():
        _fail(f"manifest missing: {manifest}")

    pdfs = sorted(corpus.glob("*.pdf"))
    n = len(pdfs)
    print(f"[ok] corpus={corpus} pdfs={n}")
    if n != args.expected_pdfs:
        print(f"[warn] corpus pdf count {n} != expected {args.expected_pdfs}")

    free_gb = shutil.disk_usage(str(ROOT)).free / 1e9
    print(f"[info] disk free={free_gb:.1f} GB")
    if free_gb < 5:
        print("[warn] low disk space (<5 GB)")

    if args.index:
        for sub in (f"chroma_db_{args.index}", f"sparse_index_{args.index}"):
            p = ROOT / sub
            if p.exists() and any(p.iterdir()):
                print(f"[warn] target index path exists (would be overwritten): {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=None, help="index name for overwrite check")
    ap.add_argument("--corpus", default=str(ROOT / "data" / "papers" / "final_corpus"))
    ap.add_argument("--manifest", default=str(ROOT / "data" / "papers" / "final_paper_manifest.csv"))
    ap.add_argument("--expected-pdfs", type=int, default=270)
    args = ap.parse_args()

    _check_python()
    _check_torch()
    _check_model()
    _check_data(args)
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
