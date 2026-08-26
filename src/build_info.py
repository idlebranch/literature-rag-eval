"""Public build identity shared by health checks and the Windows launcher."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_DEFAULT = {
    "project_id": "literature-rag-eval",
    "application_version": "1.0.0",
    "build_id": "v1.0.0-final",
    "prompt_version": "rag_answer_prompt_v2",
}


def load_build_manifest(root: Path | None = None) -> dict[str, str]:
    project_root = root or Path(__file__).resolve().parents[1]
    try:
        raw: Any = json.loads((project_root / "build_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return dict(_DEFAULT)
    return {
        key: str(raw.get(key) or fallback)
        for key, fallback in _DEFAULT.items()
    }


BUILD_INFO = load_build_manifest()
