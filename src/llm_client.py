from dataclasses import dataclass

from openai import OpenAI
from src.config import settings


@dataclass
class LLMResult:
    """Content plus transport-level metadata for one completion call."""

    content: str
    usage: dict | None
    model: str


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
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is empty. Copy .env.example to .env and fill your API key."
        )

    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def chat_messages_result(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
) -> LLMResult:
    client = get_llm_client()
    used_model = model or settings.llm_model
    resp = client.chat.completions.create(
        model=used_model,
        messages=messages,
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    return LLMResult(content=content, usage=_extract_usage(resp), model=used_model)


def chat_messages(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
) -> str:
    return chat_messages_result(messages, model=model, temperature=temperature).content


def chat_completion_result(system_prompt: str, user_prompt: str) -> LLMResult:
    return chat_messages_result(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=settings.llm_model,
        temperature=0.2,
    )


def chat_completion(system_prompt: str, user_prompt: str) -> str:
    return chat_completion_result(system_prompt, user_prompt).content
