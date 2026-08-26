"""Fetch only the BGE-M3 head files missing from the local snapshot."""
from huggingface_hub import snapshot_download

path = snapshot_download(
    "BAAI/bge-m3",
    allow_patterns=["sparse_linear.pt", "colbert_linear.pt"],
)
print("snapshot:", path)

import os
for name in ("sparse_linear.pt", "colbert_linear.pt"):
    full = os.path.join(path, name)
    print(name, os.path.getsize(full), "bytes")
