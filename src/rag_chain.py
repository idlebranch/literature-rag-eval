from __future__ import annotations

import hashlib
import math
import queue
import re
import sys
import threading
import time
from typing import Any, Dict, Generator, Iterator, List

from src.answerability import ACTION_INSTRUCTION, Action, classify_action
from src.citation_validation import validate_citations
from src.config import settings
from src.llm_client import chat_completion_result, stream_chat_completion
from src.prompts import (
    AnswerMode,
    RAG_ANSWER_PROMPT_VERSION,
    build_answer_system_prompt,
    build_answer_user_prompt,
)
from src.retriever import RetrievalResult, expand_query, format_context, retrieve
from src.warmup import get_warmup_state


# Backwards-compatible exports used by tracing/evaluation. The source of truth is
# src.prompts; SYSTEM_PROMPT is the default quick-mode prompt only.
PROMPT_VERSION = RAG_ANSWER_PROMPT_VERSION
SYSTEM_PROMPT = build_answer_system_prompt("quick")
PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]

_ANSWER_MAX_TOKENS: dict[AnswerMode, int] = {"quick": 1200, "detailed": 2200}
_query_state_lock = threading.Lock()
_first_query_seen = False

_INJECTION_PATTERNS = (
    r"忽略(之前|以上|前面).{0,12}(指令|规则|提示)",
    r"(显示|泄露|输出).{0,12}(系统提示|api\s*key|密钥|凭据)",
    r"(伪造|编造).{0,8}(引用|文献|作者|页码)",
    r"ignore\s+(all\s+)?(previous|above).{0,24}(instructions?|rules?|prompts?)",
    r"(reveal|show|print).{0,24}(system\s+prompt|api\s*key|secret|credential)",
    r"(fabricate|invent|fake).{0,16}(citation|reference|source)",
)
_AMBIGUOUS_SHORT_RE = re.compile(
    r"(处理效果|处理性能|这个方法|这种方法|这个技术|这种技术|效果|效率|性能|限制|机理)"
)
_DOMAIN_ANCHOR_RE = re.compile(
    r"(PFAS|PFOA|PFOS|AOP|PMS|PDS|臭氧|光催化|Fenton|芬顿|膜|吸附|催化剂|"
    r"污染物|废水|饮用水|DOM|NOM|抗生素|双酚|活性炭)",
    flags=re.IGNORECASE,
)
_EXACT_VALUE_REQUEST_RE = re.compile(
    r"(具体数值|精确数值|准确数值|确切数值|多少\s*(mg|g|mol|%|℃|°C|分钟|小时)|"
    r"去除率是多少|最佳浓度|最佳剂量|精确实验条件)",
    flags=re.IGNORECASE,
)
_NUMERIC_EVIDENCE_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|mg/?L|g/?L|mol/?L|mmol/?L|µg/?L|μg/?L|℃|°C|min|h|分钟|小时)",
    flags=re.IGNORECASE,
)
_CONFLICT_PAIRS = (
    ("提高", "降低"),
    ("增加", "减少"),
    ("促进", "抑制"),
    ("有效", "无效"),
    ("稳定", "失活"),
    ("higher", "lower"),
    ("increase", "decrease"),
    ("effective", "ineffective"),
)
_CONFLICT_METRIC_RE = re.compile(
    r"(反应速率|去除率|降解率|矿化率|吸附容量|通量|能耗|成本|毒性|稳定性|寿命|"
    r"rate|removal|degradation|mineralization|capacity|flux|energy|cost|toxicity|stability)",
    flags=re.IGNORECASE,
)


def _normalize_answer_mode(answer_mode: str) -> AnswerMode:
    if answer_mode not in _ANSWER_MAX_TOKENS:
        raise ValueError("answer_mode must be 'quick' or 'detailed'")
    return answer_mode  # type: ignore[return-value]


def _prompt_hash(answer_mode: AnswerMode) -> str:
    prompt = build_answer_system_prompt(answer_mode)
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def _guardrail_reason(question: str) -> str | None:
    normalized = " ".join(question.strip().split())
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in _INJECTION_PATTERNS):
        return "prompt_injection"
    if re.fullmatch(
        r"(这个|那个|它|上述|前者|后者)(效果|方法|过程|结果)?(怎么样|如何|是什么|为什么|呢)?[？?]?",
        normalized,
    ):
        return "needs_clarification"
    if (
        len(normalized) <= 24
        and _AMBIGUOUS_SHORT_RE.search(normalized)
        and not _DOMAIN_ANCHOR_RE.search(normalized)
    ):
        return "needs_clarification"
    return None


def _empty_validation(status: str = "not_applicable") -> dict[str, Any]:
    return {
        "status": status,
        "allowed_source_ids": [],
        "used_source_ids": [],
        "unused_source_ids": [],
        "invalid_source_ids": [],
        "warnings": [],
    }


def _fallback_result(
    question: str,
    reason: str,
    *,
    answer_mode: AnswerMode,
    query_rewrite: str | None = None,
    performance: dict[str, Any] | None = None,
) -> Dict[str, Any]:
    messages = {
        "prompt_injection": "该请求试图绕过知识库规则或要求伪造信息，我不能执行。请改为询问可由当前文献证据支持的问题。",
        "needs_clarification": "这个问题缺少关键限定。请补充具体污染物、处理工艺、实验条件或评价指标后再查询。",
        "insufficient_evidence": "当前知识库中没有足够证据回答该问题。",
    }
    return {
        "question": question,
        "answer": messages[reason],
        "answer_mode": answer_mode,
        "contexts": [],
        "usage": None,
        "model": settings.llm_model,
        "query_rewrite": query_rewrite or question,
        "fallback": True,
        "fallback_reason": reason,
        "evidence_status": "insufficient",
        "citation_validation": _empty_validation(),
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": _prompt_hash(answer_mode),
        "performance": performance or {},
    }


def _cold_start_flag() -> bool:
    global _first_query_seen
    prewarmed = bool(get_warmup_state().get("prewarmed"))
    with _query_state_lock:
        cold = not prewarmed and not _first_query_seen
        _first_query_seen = True
    return cold


def reset_query_runtime_state_for_tests() -> None:
    global _first_query_seen
    with _query_state_lock:
        _first_query_seen = False


def _estimate_prompt_tokens(*parts: str) -> int:
    text = "\n".join(parts)
    ascii_count = sum(1 for char in text if ord(char) < 128)
    return max(1, math.ceil(ascii_count / 4 + (len(text) - ascii_count) / 1.5))


def _coerce_retrieval_result(value: Any, question: str) -> RetrievalResult:
    """Keep backwards-compatible injected retrievers used by existing tests."""
    if isinstance(value, RetrievalResult):
        return value
    hits = value if isinstance(value, list) else []
    return RetrievalResult(
        hits=hits,
        expanded_query=expand_query(question),
        query_rewrite_ms=0.0,
        query_embedding_ms=0.0,
        chroma_search_ms=0.0,
        filter_diversify_ms=0.0,
        embedding_cache_hit=None,
        retrieval_cache_hit=False,
        raw_hit_count=len(hits),
        context_estimated_tokens=(
            _estimate_prompt_tokens(*(str(hit.get("text", "")) for hit in hits))
            if hits
            else 0
        ),
        collection_version="injected",
    )


def _evidence_status(question: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "insufficient"
    combined = "\n".join(str(hit.get("text", "")) for hit in hits)
    if _EXACT_VALUE_REQUEST_RE.search(question) and not _NUMERIC_EVIDENCE_RE.search(combined):
        return "insufficient"
    source_sentences = [
        [part.casefold() for part in re.split(r"[。！？!?;；]", str(hit.get("text", ""))) if part]
        for hit in hits
    ]
    for positive, negative in _CONFLICT_PAIRS:
        positive_sentences = [
            (index, sentence)
            for index, sentences in enumerate(source_sentences)
            for sentence in sentences
            if positive.casefold() in sentence
        ]
        negative_sentences = [
            (index, sentence)
            for index, sentences in enumerate(source_sentences)
            for sentence in sentences
            if negative.casefold() in sentence
        ]
        for positive_index, positive_sentence in positive_sentences:
            positive_metrics = set(_CONFLICT_METRIC_RE.findall(positive_sentence))
            for negative_index, negative_sentence in negative_sentences:
                if positive_index == negative_index:
                    continue
                negative_metrics = set(_CONFLICT_METRIC_RE.findall(negative_sentence))
                if positive_metrics & negative_metrics:
                    return "conflicting"
    return "available"


def _base_performance(*, cold_start: bool, request_parse_ms: float) -> dict[str, Any]:
    return {
        "request_parse_ms": request_parse_ms,
        "query_rewrite_ms": 0.0,
        "query_embedding_ms": 0.0,
        "chroma_search_ms": 0.0,
        "filter_diversify_ms": 0.0,
        "prompt_build_ms": 0.0,
        "llm_client_prepare_ms": 0.0,
        "llm_request_establish_ms": None,
        "llm_ttft_ms": None,
        "llm_full_generation_ms": 0.0,
        "citation_organize_ms": 0.0,
        "total_ms": 0.0,
        "local_processing_ms": 0.0,
        "external_model_ms": 0.0,
        "raw_retrieved_chunks": 0,
        "final_context_chunks": 0,
        "context_estimated_tokens": 0,
        "prompt_tokens": None,
        "prompt_tokens_source": None,
        "llm_calls": 0,
        "retry_count": 0,
        "embedding_cache_hit": None,
        "retrieval_cache_hit": False,
        "cold_start": cold_start,
        "prewarmed": bool(get_warmup_state().get("prewarmed")),
        "retrieval_mode": None,
        "sparse_search_ms": 0.0,
        "fusion_ms": 0.0,
        "rerank_ms": 0.0,
    }


def _set_retrieval_performance(performance: dict[str, Any], result: RetrievalResult) -> None:
    performance.update(
        {
            "query_rewrite_ms": result.query_rewrite_ms,
            "query_embedding_ms": result.query_embedding_ms,
            "chroma_search_ms": result.chroma_search_ms,
            "filter_diversify_ms": result.filter_diversify_ms,
            "raw_retrieved_chunks": result.raw_hit_count,
            "final_context_chunks": len(result.hits),
            "context_estimated_tokens": result.context_estimated_tokens,
            "embedding_cache_hit": result.embedding_cache_hit,
            "retrieval_cache_hit": result.retrieval_cache_hit,
            "collection_version": result.collection_version,
            "retrieval_mode": result.retrieval_mode,
            "sparse_search_ms": result.sparse_search_ms,
            "fusion_ms": result.fusion_ms,
            "rerank_ms": result.rerank_ms,
        }
    )


def _finish_performance(performance: dict[str, Any], request_started: float) -> None:
    performance["total_ms"] = (
        (time.perf_counter() - request_started) * 1000
        + float(performance.get("request_parse_ms") or 0.0)
    )
    local_keys = (
        "request_parse_ms",
        "query_rewrite_ms",
        "query_embedding_ms",
        "chroma_search_ms",
        "filter_diversify_ms",
        "prompt_build_ms",
        "llm_client_prepare_ms",
        "citation_organize_ms",
    )
    performance["local_processing_ms"] = sum(
        float(performance.get(key) or 0.0) for key in local_keys
    )
    performance["external_model_ms"] = float(performance.get("llm_full_generation_ms") or 0.0)


def _prepare_retrieval(
    question: str,
    top_k: int,
    performance: dict[str, Any],
) -> tuple[RetrievalResult, str]:
    retrieval = _coerce_retrieval_result(
        retrieve(question, top_k=top_k, with_metrics=True),
        question,
    )
    _set_retrieval_performance(performance, retrieval)
    hits = retrieval.hits
    best_distance = min((float(hit.get("distance", 999.0)) for hit in hits), default=999.0)
    status = _evidence_status(question, hits)
    if best_distance > settings.max_retrieval_distance:
        status = "insufficient"
    return retrieval, status


def answer_question(
    question: str,
    top_k: int | None = None,
    *,
    answer_mode: str = "quick",
    request_parse_ms: float = 0.0,
) -> Dict[str, Any]:
    mode = _normalize_answer_mode(answer_mode)
    request_started = time.perf_counter()
    performance = _base_performance(
        cold_start=_cold_start_flag(),
        request_parse_ms=request_parse_ms,
    )
    guardrail_reason = _guardrail_reason(question)
    if guardrail_reason:
        result = _fallback_result(
            question,
            guardrail_reason,
            answer_mode=mode,
            performance=performance,
        )
        _finish_performance(performance, request_started)
        return result

    retrieval, evidence_status = _prepare_retrieval(
        question,
        top_k or settings.top_k,
        performance,
    )
    hits = retrieval.hits
    best_distance = min((float(hit.get("distance", 999.0)) for hit in hits), default=999.0)
    action, action_reason = classify_action(question, hits, evidence_status, best_distance)

    if action == Action.REFUSE:
        result = _fallback_result(
            question,
            "insufficient_evidence",
            answer_mode=mode,
            query_rewrite=retrieval.expanded_query,
            performance=performance,
        )
        result["action"] = action.value
        result["action_reason"] = action_reason
        _finish_performance(performance, request_started)
        return result
    if action == Action.CLARIFY:
        result = _fallback_result(
            question,
            "needs_clarification",
            answer_mode=mode,
            query_rewrite=retrieval.expanded_query,
            performance=performance,
        )
        result["action"] = action.value
        result["action_reason"] = action_reason
        _finish_performance(performance, request_started)
        return result

    prompt_started = time.perf_counter()
    system_prompt = build_answer_system_prompt(mode)
    user_prompt = build_answer_user_prompt(
        question,
        format_context(hits),
        evidence_status=evidence_status,
        action=action,
    )
    performance["prompt_build_ms"] = (time.perf_counter() - prompt_started) * 1000
    estimated_prompt_tokens = _estimate_prompt_tokens(system_prompt, user_prompt)
    llm = chat_completion_result(
        system_prompt,
        user_prompt,
        max_tokens=_ANSWER_MAX_TOKENS[mode],
    )
    performance.update(
        {
            "llm_client_prepare_ms": float(getattr(llm, "client_prepare_ms", 0.0)),
            "llm_full_generation_ms": float(getattr(llm, "full_generation_ms", 0.0)),
            "llm_calls": 1,
            "retry_count": int(getattr(llm, "retry_count", 0)),
        }
    )
    usage = getattr(llm, "usage", None)
    exact_prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    performance["prompt_tokens"] = exact_prompt_tokens or estimated_prompt_tokens
    performance["prompt_tokens_source"] = "api" if exact_prompt_tokens else "estimated"

    citation_started = time.perf_counter()
    answer, validation = validate_citations(
        llm.content,
        hits,
        generation_truncated=getattr(llm, "finish_reason", None) == "length",
    )
    result = {
        "question": question,
        "answer": answer,
        "answer_mode": mode,
        "contexts": hits,
        "usage": usage,
        "model": llm.model,
        "query_rewrite": retrieval.expanded_query,
        "fallback": False,
        "fallback_reason": None,
        "evidence_status": evidence_status,
        "action": action.value,
        "action_reason": action_reason,
        "citation_validation": validation,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": _prompt_hash(mode),
        "performance": performance,
    }
    performance["citation_organize_ms"] = (time.perf_counter() - citation_started) * 1000
    _finish_performance(performance, request_started)
    return result


_STREAM_STATUS = {
    "embedding": ("embedding", "正在生成查询向量"),
    "chroma": ("retrieval", "正在检索文献"),
    "sparse": ("retrieval", "正在执行词面检索"),
    "fusion": ("fusion", "正在融合两路检索结果"),
    "rerank": ("rerank", "正在重排序候选片段"),
    "filter": ("filter", "正在去重并筛选相关片段"),
    "retrieval_cache_hit": ("retrieval", "正在读取已缓存的检索结果"),
}


def _stream_retrieval(
    question: str,
    top_k: int,
) -> Generator[dict[str, Any], None, RetrievalResult]:
    events: queue.Queue[tuple[str, Any]] = queue.Queue()

    def progress(stage: str) -> None:
        events.put(("progress", stage))

    def worker() -> None:
        try:
            value = retrieve(
                question,
                top_k=top_k,
                with_metrics=True,
                progress_callback=progress,
            )
            events.put(("result", value))
        except BaseException as exc:
            events.put(("error", exc))

    threading.Thread(target=worker, name="rag-stream-retrieval", daemon=True).start()
    while True:
        kind, value = events.get()
        if kind == "progress":
            stage, message = _STREAM_STATUS.get(value, (value, str(value)))
            yield {"type": "status", "stage": stage, "message": message}
        elif kind == "error":
            raise value
        else:
            return _coerce_retrieval_result(value, question)


class _CitationDisplayFilter:
    """Withhold unvalidated [Sx] markers from live display until final validation."""

    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, text: str) -> str:
        self.buffer += text
        output: list[str] = []
        while self.buffer:
            opening = self.buffer.find("[")
            if opening < 0:
                output.append(self.buffer)
                self.buffer = ""
                break
            output.append(self.buffer[:opening])
            self.buffer = self.buffer[opening:]
            closing = self.buffer.find("]")
            if closing < 0:
                break
            candidate = self.buffer[: closing + 1]
            self.buffer = self.buffer[closing + 1 :]
            if not re.fullmatch(r"\[S\d+\]", candidate, flags=re.IGNORECASE):
                output.append(candidate)
        return "".join(output)

    def flush(self) -> str:
        remaining = self.buffer
        self.buffer = ""
        return "" if re.fullmatch(r"\[S\d+\]", remaining, flags=re.IGNORECASE) else remaining


def stream_answer_question(
    question: str,
    top_k: int | None = None,
    *,
    answer_mode: str = "quick",
    request_parse_ms: float = 0.0,
) -> Iterator[dict[str, Any]]:
    """Stream real answer text; expose citation markers only after validation."""
    mode = _normalize_answer_mode(answer_mode)
    request_started = time.perf_counter()
    performance = _base_performance(
        cold_start=_cold_start_flag(),
        request_parse_ms=request_parse_ms,
    )
    yield {"type": "status", "stage": "understanding", "message": "正在理解问题"}
    guardrail_reason = _guardrail_reason(question)
    if guardrail_reason:
        result = _fallback_result(
            question,
            guardrail_reason,
            answer_mode=mode,
            performance=performance,
        )
        _finish_performance(performance, request_started)
        yield {"type": "final", "result": result}
        return

    retrieval = yield from _stream_retrieval(question, top_k or settings.top_k)
    _set_retrieval_performance(performance, retrieval)
    hits = retrieval.hits
    evidence_status = _evidence_status(question, hits)
    best_distance = min((float(hit.get("distance", 999.0)) for hit in hits), default=999.0)
    action, action_reason = classify_action(question, hits, evidence_status, best_distance)
    yield {
        "type": "status",
        "stage": "retrieved",
        "message": f"已找到 {len(hits)} 个相关片段",
        "chunk_count": len(hits),
    }
    if action == Action.REFUSE:
        result = _fallback_result(
            question,
            "insufficient_evidence",
            answer_mode=mode,
            query_rewrite=retrieval.expanded_query,
            performance=performance,
        )
        result["action"] = action.value
        result["action_reason"] = action_reason
        _finish_performance(performance, request_started)
        yield {"type": "final", "result": result}
        return
    if action == Action.CLARIFY:
        result = _fallback_result(
            question,
            "needs_clarification",
            answer_mode=mode,
            query_rewrite=retrieval.expanded_query,
            performance=performance,
        )
        result["action"] = action.value
        result["action_reason"] = action_reason
        _finish_performance(performance, request_started)
        yield {"type": "final", "result": result}
        return

    prompt_started = time.perf_counter()
    system_prompt = build_answer_system_prompt(mode)
    user_prompt = build_answer_user_prompt(
        question,
        format_context(hits),
        evidence_status=evidence_status,
        action=action,
    )
    performance["prompt_build_ms"] = (time.perf_counter() - prompt_started) * 1000
    estimated_prompt_tokens = _estimate_prompt_tokens(system_prompt, user_prompt)
    yield {"type": "status", "stage": "generating", "message": "正在生成回答"}
    token_stream, llm_state = stream_chat_completion(
        system_prompt,
        user_prompt,
        max_tokens=_ANSWER_MAX_TOKENS[mode],
    )
    answer_parts: list[str] = []
    display_filter = _CitationDisplayFilter()
    for token in token_stream:
        answer_parts.append(token)
        safe_display = display_filter.feed(token)
        if safe_display:
            yield {"type": "token", "content": safe_display}
    remaining = display_filter.flush()
    if remaining:
        yield {"type": "token", "content": remaining}

    performance.update(
        {
            "llm_client_prepare_ms": llm_state.client_prepare_ms,
            "llm_request_establish_ms": llm_state.request_establish_ms,
            "llm_ttft_ms": llm_state.ttft_ms,
            "llm_full_generation_ms": llm_state.full_generation_ms or 0.0,
            "llm_calls": 1,
            "retry_count": llm_state.retry_count,
        }
    )
    exact_prompt_tokens = (
        llm_state.usage.get("prompt_tokens")
        if isinstance(llm_state.usage, dict)
        else None
    )
    performance["prompt_tokens"] = exact_prompt_tokens or estimated_prompt_tokens
    performance["prompt_tokens_source"] = "api" if exact_prompt_tokens else "estimated"

    yield {"type": "status", "stage": "citations", "message": "正在校验并整理引用"}
    citation_started = time.perf_counter()
    answer, validation = validate_citations(
        "".join(answer_parts),
        hits,
        generation_truncated=getattr(llm_state, "finish_reason", None) == "length",
    )
    result = {
        "question": question,
        "answer": answer,
        "answer_mode": mode,
        "contexts": hits,
        "usage": llm_state.usage,
        "model": llm_state.model,
        "query_rewrite": retrieval.expanded_query,
        "fallback": False,
        "fallback_reason": None,
        "evidence_status": evidence_status,
        "action": action.value,
        "action_reason": action_reason,
        "citation_validation": validation,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": _prompt_hash(mode),
        "performance": performance,
    }
    performance["citation_organize_ms"] = (time.perf_counter() - citation_started) * 1000
    _finish_performance(performance, request_started)
    yield {"type": "final", "result": result}


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m src.rag_chain "你的问题"')
        return
    question = " ".join(sys.argv[1:])
    result = answer_question(question)
    print("\n=== Answer ===\n")
    print(result["answer"])
    print("\n=== Retrieved Sources ===\n")
    for index, hit in enumerate(result["contexts"], start=1):
        metadata = hit["metadata"]
        print(
            f"[S{index}] {metadata['source']} | page {metadata['page']} "
            f"| distance={hit['distance']:.4f}"
        )


if __name__ == "__main__":
    main()
