"""Run the two permitted live answer-mode checks without logging answer text.

This script intentionally records only timings, counts, prompt identity, evidence
status, and deterministic citation-validation results. It never prints the API
key, generated answer, or retrieved document contents.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Any


DEFAULT_QUESTION = "PFAS 水处理工程化面临哪些主要限制？"


def _stream_once(base_url: str, question: str, answer_mode: str) -> dict[str, Any]:
    payload = json.dumps(
        {"question": question, "top_k": 5, "answer_mode": answer_mode},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/stream",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    started = time.perf_counter()
    client_first_token_ms: float | None = None
    final_result: dict[str, Any] | None = None
    with urllib.request.urlopen(request, timeout=240) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event.get("type") == "token" and client_first_token_ms is None:
                client_first_token_ms = (time.perf_counter() - started) * 1000
            if event.get("type") == "error":
                raise RuntimeError(str(event.get("message") or "stream failed"))
            if event.get("type") == "final":
                final_result = dict(event.get("result") or {})
    client_total_ms = (time.perf_counter() - started) * 1000
    if final_result is None:
        raise RuntimeError("stream ended without a final result")

    performance = dict(final_result.get("performance") or {})
    validation = dict(final_result.get("citation_validation") or {})
    answer = str(final_result.get("answer") or "")
    contexts = list(final_result.get("contexts") or [])
    used_ids = list(validation.get("used_source_ids") or [])
    invalid_ids = list(validation.get("invalid_source_ids") or [])
    warnings = list(validation.get("warnings") or [])
    return {
        "answer_mode": answer_mode,
        "fallback": bool(final_result.get("fallback")),
        "fallback_reason": final_result.get("fallback_reason"),
        "evidence_status": final_result.get("evidence_status"),
        "answer_chars": len(answer),
        "context_count": len(contexts),
        "used_citation_count": len(used_ids),
        "unused_source_count": len(validation.get("unused_source_ids") or []),
        "invalid_citation_count": len(invalid_ids),
        "citation_validation_status": validation.get("status"),
        "citation_warning_count": len(warnings),
        "llm_calls": performance.get("llm_calls"),
        "retry_count": performance.get("retry_count"),
        "prompt_tokens": performance.get("prompt_tokens"),
        "prompt_tokens_source": performance.get("prompt_tokens_source"),
        "server_ttft_ms": performance.get("llm_ttft_ms"),
        "client_first_token_ms": client_first_token_ms,
        "llm_full_generation_ms": performance.get("llm_full_generation_ms"),
        "server_total_ms": performance.get("total_ms"),
        "client_total_ms": client_total_ms,
        "prompt_version": final_result.get("prompt_version"),
        "prompt_hash": final_result.get("prompt_hash"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    args = parser.parse_args()
    results = [
        _stream_once(args.base_url, args.question, "quick"),
        _stream_once(args.base_url, args.question, "detailed"),
    ]
    print(json.dumps({"question_id": "pfas_engineering_limits", "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
