"""Download bge-reranker-v2-m3 with endpoint fallback and per-file retries."""
import os
import sys
import time

from huggingface_hub import snapshot_download

ENDPOINTS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]

last_error = None
for endpoint in ENDPOINTS:
    os.environ["HF_ENDPOINT"] = endpoint
    for attempt in range(1, 4):
        try:
            print(f"trying {endpoint} attempt {attempt}", flush=True)
            path = snapshot_download(
                "BAAI/bge-reranker-v2-m3",
                allow_patterns=[
                    "*.json", "*.txt", "sentencepiece.bpe.model",
                    "model.safetensors", "pytorch_model.bin",
                ],
                max_workers=2,
            )
            print("OK:", path, flush=True)
            sys.exit(0)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_error = e
            print(f"failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
            time.sleep(3)

print("ALL_ENDPOINTS_FAILED:", repr(last_error)[:300], file=sys.stderr)
sys.exit(1)
