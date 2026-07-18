"""Preflight health check for the OpenAI-compatible API.

Catches the V2 fiasco (15 consecutive 401 errors silently written to CSV) by
failing loudly *before* the long batch starts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HealthStatus:
    ok: bool
    message: str
    model: str = ""


def check_llm(timeout_seconds: float = 15.0) -> HealthStatus:
    """Send one trivial request to verify api_key + base_url + model are valid."""
    try:
        from src.config import settings
        from src.llm_client import get_llm_client
    except Exception as e:
        return HealthStatus(ok=False, message=f"config/llm_client import failed: {e}")

    if not settings.openai_api_key:
        return HealthStatus(
            ok=False,
            message="OPENAI_API_KEY is empty in .env. Copy .env.example to .env and fill it.",
        )

    try:
        client = get_llm_client()
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=4,
            temperature=0,
            timeout=timeout_seconds,
        )
        content = resp.choices[0].message.content or ""
        return HealthStatus(ok=True, message=f"OK ({len(content)} chars)", model=settings.llm_model)
    except Exception as e:
        return HealthStatus(
            ok=False,
            message=f"LLM call failed: {type(e).__name__}: {e}",
            model=settings.llm_model,
        )


def require_healthy() -> None:
    """Raise SystemExit with a clear remediation if the LLM is unreachable."""
    status = check_llm()
    if status.ok:
        print(f"[health] LLM OK · model={status.model}")
        return
    raise SystemExit(
        "\n".join(
            [
                "[health] LLM preflight FAILED. Aborting before wasting batch time.",
                f"  reason: {status.message}",
                "  fix:",
                "    1. Check OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL in .env",
                "    2. Curl-test: curl $OPENAI_BASE_URL/models -H 'Authorization: Bearer $OPENAI_API_KEY'",
                "    3. If the model name is wrong, set LLM_MODEL=<valid model> in .env",
            ]
        )
    )


if __name__ == "__main__":
    status = check_llm()
    print(status)
    raise SystemExit(0 if status.ok else 1)
