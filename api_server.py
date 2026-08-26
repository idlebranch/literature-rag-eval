from __future__ import annotations

from contextlib import asynccontextmanager
import json
import time
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.build_info import BUILD_INFO
from src.llm_client import chat_messages
from src.prompts import DIRECT_LLM_SYSTEM_PROMPT
from src.status import get_runtime_status
from src.tracing.instrumentation import traced_chat, traced_stream
from src.utils.logging import get_logger
from src.warmup import start_warmup


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Background warmup loads only local Embedding/Chroma resources. No LLM call.
    start_warmup()
    yield


app = FastAPI(
    title="Literature RAG API",
    description="HTTP wrapper for local literature RAG, used by Promptfoo red team evaluation.",
    version=BUILD_INFO["application_version"],
    lifespan=lifespan,
)


@app.middleware("http")
async def mark_request_arrival(request: Request, call_next):
    request.state.received_at = time.perf_counter()
    return await call_next(request)


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="Question for the literature knowledge base.",
    )
    top_k: int | None = Field(default=None, ge=1, le=20, description="Number of retrieved chunks.")
    answer_mode: Literal["quick", "detailed"] = Field(
        default="quick",
        description="Answer depth. Evidence and citation rules are identical in both modes.",
    )


class ChatResponse(BaseModel):
    question: str
    answer: str
    contexts: list[dict[str, Any]]
    context_count: int
    trace_id: str
    query_rewrite: str
    fallback: bool
    fallback_reason: str | None = None
    answer_mode: str = "quick"
    evidence_status: str = "available"
    citation_validation: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str = ""
    performance: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: str = Field(
        ...,
        pattern="^(system|user|assistant)$",
        description="Message role: system, user, or assistant.",
    )
    content: str = Field(..., min_length=1, max_length=16000, description="Message content.")


class LLMChatRequest(BaseModel):
    prompt: str | None = Field(default=None, min_length=1, max_length=16000, description="Shortcut user prompt.")
    messages: list[Message] | None = Field(
        default=None,
        max_length=50,
        description="OpenAI-style chat messages. If set, this takes priority over prompt.",
    )
    model: str | None = Field(default=None, description="Optional model override.")
    temperature: float = Field(default=0.2, ge=0, le=2)


class LLMChatResponse(BaseModel):
    model: str
    answer: str
    mode: str = "direct_llm"
    citations_supported: bool = False


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
    status = get_runtime_status()
    status["default_top_k"] = settings.top_k
    return status


@app.get("/knowledge-base/status")
def knowledge_base_status() -> dict[str, Any]:
    status = get_runtime_status(force=True)
    return {
        "status": status["status"],
        "rag_ready": status["rag_ready"],
        "project_id": status["project_id"],
        "build_id": status["build_id"],
        "prompt_version": status["prompt_version"],
        "prewarmed": status["prewarmed"],
        "knowledge_base": status["knowledge_base"],
        "vector_index": status["vector_index"],
        "embedding": status["embedding"],
    }


@app.post("/llm/chat", response_model=LLMChatResponse)
def llm_chat(req: LLMChatRequest) -> LLMChatResponse:
    if req.messages:
        messages = [
            message.model_dump() if hasattr(message, "model_dump") else message.dict()
            for message in req.messages
        ]
    elif req.prompt:
        messages = [
            {"role": "system", "content": DIRECT_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": req.prompt.strip()},
        ]
    else:
        raise HTTPException(status_code=400, detail="Either prompt or messages must be provided.")

    model = req.model or settings.llm_model
    try:
        answer = chat_messages(messages, model=model, temperature=req.temperature)
    except Exception as exc:
        logger.error("Direct LLM request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="直接 LLM 服务暂时不可用，请检查模型配置或网络。") from None
    return LLMChatResponse(model=model, answer=answer)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    parse_ms = (time.perf_counter() - request.state.received_at) * 1000
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空。")
    top_k = req.top_k or settings.top_k
    try:
        result, trace_id = traced_chat(
            question,
            top_k=top_k,
            request_parse_ms=parse_ms,
            answer_mode=req.answer_mode,
        )
    except Exception as exc:
        logger.error("RAG request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="RAG 服务暂时不可用，请检查索引、Embedding 或模型配置。") from None

    contexts = [simplify_context(hit) for hit in result.get("contexts", [])]

    return ChatResponse(
        question=result.get("question", question),
        answer=str(result.get("answer", "")),
        contexts=contexts,
        context_count=len(contexts),
        trace_id=trace_id,
        query_rewrite=str(result.get("query_rewrite", question)),
        fallback=bool(result.get("fallback", False)),
        fallback_reason=result.get("fallback_reason"),
        answer_mode=str(result.get("answer_mode", req.answer_mode)),
        evidence_status=str(result.get("evidence_status", "available")),
        citation_validation=result.get("citation_validation") or {},
        prompt_version=str(result.get("prompt_version", BUILD_INFO["prompt_version"])),
        performance=result.get("performance") or {},
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """SSE endpoint carrying real model tokens followed by verified contexts."""
    parse_ms = (time.perf_counter() - request.state.received_at) * 1000
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空。")
    top_k = req.top_k or settings.top_k
    stream, trace_id = traced_stream(
        question,
        top_k,
        request_parse_ms=parse_ms,
        answer_mode=req.answer_mode,
    )

    def event_source():
        try:
            for event in stream:
                outbound = dict(event)
                if event.get("type") == "final":
                    result = dict(event.get("result") or {})
                    result["contexts"] = [
                        simplify_context(hit) for hit in result.get("contexts", [])
                    ]
                    result["context_count"] = len(result["contexts"])
                    result["trace_id"] = trace_id
                    outbound["result"] = result
                yield "data: " + json.dumps(outbound, ensure_ascii=False) + "\n\n"
        except GeneratorExit:
            raise
        except Exception as exc:
            logger.error("Streaming RAG request failed: %s", type(exc).__name__)
            error = {
                "type": "error",
                "message": "RAG 服务暂时不可用，请检查索引、Embedding 或模型配置。",
            }
            yield "data: " + json.dumps(error, ensure_ascii=False) + "\n\n"
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
