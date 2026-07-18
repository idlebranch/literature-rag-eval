from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.llm_client import chat_messages
from src.tracing.instrumentation import traced_chat


app = FastAPI(
    title="Literature RAG API",
    description="HTTP wrapper for local literature RAG, used by Promptfoo red team evaluation.",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    question: str = Field(..., description="User question sent by Promptfoo or other clients.")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Number of retrieved chunks.")


class ChatResponse(BaseModel):
    question: str
    answer: str
    contexts: list[dict[str, Any]]
    context_count: int
    trace_id: str


class Message(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant.")
    content: str = Field(..., description="Message content.")


class LLMChatRequest(BaseModel):
    prompt: str | None = Field(default=None, description="Shortcut user prompt.")
    messages: list[Message] | None = Field(
        default=None,
        description="OpenAI-style chat messages. If set, this takes priority over prompt.",
    )
    model: str | None = Field(default=None, description="Optional model override.")
    temperature: float = Field(default=0.2, ge=0, le=2)


class LLMChatResponse(BaseModel):
    model: str
    answer: str


def simplify_context(hit: dict[str, Any]) -> dict[str, Any]:
    """
    Keep only JSON-safe fields for API output.
    Promptfoo mainly needs `answer`, but contexts are useful for later debugging.
    """
    metadata = hit.get("metadata") or {}

    return {
        "text": str(hit.get("text", "")),
        "distance": hit.get("distance"),
        "metadata": {
            "source": str(metadata.get("source", "")),
            "page": metadata.get("page"),
            "chunk_index": metadata.get("chunk_index"),
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "default_top_k": settings.top_k,
        "llm_model": settings.llm_model,
        "openai_base_url": settings.openai_base_url,
    }


@app.post("/llm/chat", response_model=LLMChatResponse)
def llm_chat(req: LLMChatRequest) -> LLMChatResponse:
    if req.messages:
        messages = [message.dict() for message in req.messages]
    elif req.prompt:
        messages = [{"role": "user", "content": req.prompt}]
    else:
        raise HTTPException(status_code=400, detail="Either prompt or messages must be provided.")

    model = req.model or settings.llm_model
    answer = chat_messages(messages, model=model, temperature=req.temperature)
    return LLMChatResponse(model=model, answer=answer)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    top_k = req.top_k or settings.top_k
    result, trace_id = traced_chat(req.question, top_k=top_k)

    contexts = [simplify_context(hit) for hit in result.get("contexts", [])]

    return ChatResponse(
        question=result.get("question", req.question),
        answer=str(result.get("answer", "")),
        contexts=contexts,
        context_count=len(contexts),
        trace_id=trace_id,
    )
