from dataclasses import dataclass
from datetime import datetime
import threading
import time
from typing import Iterator

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from src.config import settings


_last_success_at: str | None = None
_last_error_type: str | None = None
_client: OpenAI | None = None
_client_signature: tuple[str, str] | None = None
_client_lock = threading.Lock()


@dataclass
class LLMResult:
    """Content plus transport-level metadata for one completion call."""

    content: str
    usage: dict | None
    model: str
    retry_count: int = 0
    client_prepare_ms: float = 0.0
    full_generation_ms: float = 0.0
    finish_reason: str | None = None


@dataclass
class LLMStreamState:
    """Mutable timing/transport state populated while a stream is consumed."""

    model: str
    retry_count: int = 0
    client_prepare_ms: float = 0.0
    request_establish_ms: float | None = None
    ttft_ms: float | None = None
    full_generation_ms: float | None = None
    usage: dict | None = None
    finish_reason: str | None = None


def _extract_usage(resp) -> dict | None:
    """Best-effort token usage extraction.

    OpenAI-compatible proxies vary: some return a CompletionUsage model, some a
    dict, some nothing. Any failure yields None (recorded as null in the trace).
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    try:
        if hasattr(usage, "model_dump"):
            data = usage.model_dump()
        elif isinstance(usage, dict):
            data = usage
        else:
            data = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    out = {k: data.get(k) for k in keys if k in data}
    return out or None


def get_llm_client() -> OpenAI:
    global _client, _client_signature
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is empty. Copy .env.example to .env and fill your API key."
        )

    signature = (settings.openai_api_key, settings.openai_base_url)
    with _client_lock:
        if _client is None or _client_signature != signature:
            _client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=httpx.Timeout(
                    connect=settings.llm_connect_timeout,
                    read=settings.llm_read_timeout,
                    write=30.0,
                    pool=10.0,
                ),
                # Retries are explicit below so the trace can report every attempt.
                max_retries=0,
            )
            _client_signature = signature
        return _client


def clear_llm_client_cache() -> None:
    """Close the pooled client and clear singleton state (primarily for tests)."""
    global _client, _client_signature
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass
        _client = None
        _client_signature = None


def _is_retriable(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    return False


def _create_with_retry(client: OpenAI, **kwargs):
    retries = 0
    while True:
        try:
            return client.chat.completions.create(**kwargs), retries
        except Exception as exc:
            if retries >= settings.llm_max_retries or not _is_retriable(exc):
                raise
            retries += 1
            time.sleep(min(0.4 * (2 ** (retries - 1)), 2.0))


def get_llm_runtime_state() -> dict[str, str | bool | None]:
    return {
        "network_checked": _last_success_at is not None or _last_error_type is not None,
        "last_success_at": _last_success_at,
        "last_error_type": _last_error_type,
    }


def chat_messages_result(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> LLMResult:
    global _last_success_at, _last_error_type
    client_started = time.perf_counter()
    client = get_llm_client()
    client_prepare_ms = (time.perf_counter() - client_started) * 1000
    used_model = model or settings.llm_model
    generation_started = time.perf_counter()
    try:
        request_kwargs = {
            "model": used_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        resp, retry_count = _create_with_retry(client, **request_kwargs)
    except Exception as exc:
        _last_error_type = type(exc).__name__
        raise
    _last_success_at = datetime.now().astimezone().isoformat(timespec="seconds")
    _last_error_type = None
    content = resp.choices[0].message.content or ""
    return LLMResult(
        content=content,
        usage=_extract_usage(resp),
        model=used_model,
        retry_count=retry_count,
        client_prepare_ms=client_prepare_ms,
        full_generation_ms=(time.perf_counter() - generation_started) * 1000,
        finish_reason=getattr(resp.choices[0], "finish_reason", None),
    )


def stream_chat_messages(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> tuple[Iterator[str], LLMStreamState]:
    """Return a real model token stream plus timing state; no simulated chunks."""
    client_started = time.perf_counter()
    client = get_llm_client()
    used_model = model or settings.llm_model
    state = LLMStreamState(
        model=used_model,
        client_prepare_ms=(time.perf_counter() - client_started) * 1000,
    )
    request_started = time.perf_counter()
    try:
        request_kwargs = {
            "model": used_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        stream, state.retry_count = _create_with_retry(client, **request_kwargs)
    except Exception as exc:
        global _last_error_type
        _last_error_type = type(exc).__name__
        raise
    state.request_establish_ms = (time.perf_counter() - request_started) * 1000

    def consume() -> Iterator[str]:
        global _last_success_at, _last_error_type
        try:
            for chunk in stream:
                usage = _extract_usage(chunk)
                if usage:
                    state.usage = usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                finish_reason = getattr(choices[0], "finish_reason", None)
                if finish_reason:
                    state.finish_reason = finish_reason
                content = getattr(choices[0].delta, "content", None) or ""
                if content:
                    if state.ttft_ms is None:
                        state.ttft_ms = (time.perf_counter() - request_started) * 1000
                    yield content
            _last_success_at = datetime.now().astimezone().isoformat(timespec="seconds")
            _last_error_type = None
        except GeneratorExit:
            raise
        except Exception as exc:
            _last_error_type = type(exc).__name__
            raise
        finally:
            state.full_generation_ms = (time.perf_counter() - request_started) * 1000
            try:
                stream.close()
            except Exception:
                pass

    return consume(), state


def stream_chat_completion(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int | None = None,
) -> tuple[Iterator[str], LLMStreamState]:
    return stream_chat_messages(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=settings.llm_model,
        temperature=0.2,
        max_tokens=max_tokens,
    )


def chat_messages(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
) -> str:
    return chat_messages_result(messages, model=model, temperature=temperature).content


def chat_completion_result(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int | None = None,
) -> LLMResult:
    return chat_messages_result(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=settings.llm_model,
        temperature=0.2,
        max_tokens=max_tokens,
    )


def chat_completion(system_prompt: str, user_prompt: str) -> str:
    return chat_completion_result(system_prompt, user_prompt).content
