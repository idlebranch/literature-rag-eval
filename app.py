from __future__ import annotations

import json
from typing import Any, Iterator

import httpx
import streamlit as st

from src.config import settings


API_BASE = f"http://{settings.api_host}:{settings.api_port}"


@st.cache_resource
def api_client() -> httpx.Client:
    """Reuse one keep-alive pool for Streamlit-to-API traffic."""
    return httpx.Client(
        base_url=API_BASE,
        timeout=httpx.Timeout(connect=3.0, read=settings.llm_read_timeout + 15, write=10.0, pool=5.0),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    )


def read_health() -> dict[str, Any] | None:
    try:
        response = api_client().get("/health")
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except (httpx.HTTPError, ValueError):
        return None


def ms(value: Any) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric / 1000:.2f}s" if numeric >= 1000 else f"{numeric:.0f}ms"


def cache_label(value: Any) -> str:
    if value is True:
        return "hit"
    if value is False:
        return "miss"
    return "skipped"


def stream_rag(
    question: str,
    top_k: int,
    answer_mode: str,
    run_status,
    final_holder: dict[str, Any],
) -> Iterator[str]:
    with api_client().stream(
        "POST",
        "/chat/stream",
        json={"question": question, "top_k": top_k, "answer_mode": answer_mode},
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            event_type = event.get("type")
            if event_type == "status":
                run_status.update(label=str(event.get("message", "处理中…")), state="running")
            elif event_type == "token":
                yield str(event.get("content", ""))
            elif event_type == "final":
                final_holder["result"] = event.get("result") or {}
            elif event_type == "error":
                raise RuntimeError(str(event.get("message", "流式请求失败。")))


st.set_page_config(page_title="Literature RAG for Water Treatment", page_icon="📚", layout="wide")
st.title("Literature RAG for Water Treatment")
st.caption(
    "Scientific PDF → Section-aware Parsing → Dense + Sparse Retrieval → RRF → "
    "Evidence-aware Answering → Citation Validation"
)

health = read_health()
st.sidebar.header("📊 Runtime Status")
if health:
    knowledge = health.get("knowledge_base") or {}
    vector = health.get("vector_index") or {}
    sparse = vector.get("sparse") or {}
    retrieval = health.get("retrieval") or {}
    embedding = health.get("embedding") or {}
    llm = health.get("llm") or {}
    st.sidebar.metric("📄 Knowledge Base", f"{knowledge.get('document_count', '—')} PDFs")
    st.sidebar.metric("🧩 Dense Chunks", vector.get("chunk_count", "—"))
    st.sidebar.caption(f"Corpus: {knowledge.get('path', '—')}")
    st.sidebar.caption(
        f"Retrieval: {retrieval.get('pipeline', '—')}"
        + (f" · {retrieval.get('fusion')}" if retrieval.get('fusion') else "")
    )
    st.sidebar.caption(
        f"Baseline: {retrieval.get('label', retrieval.get('mode', '—'))} · "
        f"Chunking: {retrieval.get('chunking', '—')}"
    )
    st.sidebar.caption(
        f"Dense index: {vector.get('status', 'unknown')} · {vector.get('path', '—')}"
    )
    st.sidebar.caption(
        f"Sparse index: {sparse.get('status', 'unknown')} · {sparse.get('chunk_count', '—')} chunks"
    )
    st.sidebar.caption(f"Embedding: {embedding.get('status', 'unknown')} · {embedding.get('model', '—')}")
    st.sidebar.caption(f"LLM: {llm.get('status', 'unknown')} · {llm.get('model', '—')}")
    st.sidebar.caption(
        f"Release: v{health.get('application_version', health.get('version', 'unknown'))} · "
        f"Build: {health.get('build_id', 'unknown')}"
    )
    if health.get("prewarmed"):
        st.sidebar.success("预热完成")
    else:
        st.sidebar.info("预热进行中或尚未完成")
    st.sidebar.caption(
        "Tracing: " + ("enabled" if (health.get("tracing") or {}).get("enabled") else "disabled by default")
    )
else:
    st.sidebar.error("后端不可用，请通过启动器启动或执行健康检查。")

st.sidebar.metric("🔍 Default Top-K", settings.top_k)
st.sidebar.info("索引为冻结的 release artifact；页面不会自动重建或切换索引。")
st.sidebar.divider()

question = st.text_area(
    "输入你的问题",
    value="What are the main engineering limitations of PFAS treatment reported in this corpus?",
    height=100,
    max_chars=8000,
)
top_k = st.slider("Top-K", min_value=4, max_value=15, value=settings.top_k)
answer_mode_label = st.radio(
    "回答模式",
    options=["快速回答", "详细回答"],
    index=0,
    horizontal=True,
    help="两种模式使用相同事实与引用规则；详细模式仅增加跨文献展开和条件差异。",
)
answer_mode = "quick" if answer_mode_label == "快速回答" else "detailed"
st.caption(f"当前模式：{answer_mode_label}")
comparison_mode = st.toggle(
    "RAG / 直接 LLM 对照模式",
    value=False,
    help="开启后才会额外调用一次直接 LLM，可能增加费用和等待时间。",
)
debug_mode = st.toggle(
    "性能调试模式",
    value=settings.performance_debug,
    help="显示逐阶段耗时、缓存与重试信息；不会显示 API Key 或完整日志。",
)

if st.button("Run RAG", type="primary"):
    normalized_question = question.strip()
    if not normalized_question:
        st.warning("问题不能为空。")
        st.stop()
    if not health:
        st.error("后端不可用，请先在启动器中启动项目。")
        st.stop()

    run_status = st.status("正在理解问题", expanded=True)
    st.subheader("📝 Answer")
    final_holder: dict[str, Any] = {}
    answer_placeholder = st.empty()
    streamed_answer = ""
    try:
        for content in stream_rag(
            normalized_question,
            top_k,
            answer_mode,
            run_status,
            final_holder,
        ):
            streamed_answer += content
            answer_placeholder.markdown(streamed_answer + "▌")
    except Exception as exc:
        run_status.update(label="请求失败", state="error", expanded=True)
        st.error(f"RAG 服务暂时不可用（{type(exc).__name__}）。请在启动器中运行健康检查。")
        st.stop()

    result = final_holder.get("result") or {}
    if not result:
        run_status.update(label="响应不完整", state="error", expanded=True)
        st.error("后端流未返回最终结果。")
        st.stop()
    # Replace the provisional citation-free stream with the validated final answer.
    answer_placeholder.markdown(str(result.get("answer", streamed_answer)))
    validation = result.get("citation_validation") or {}
    if validation.get("status") in {"corrected", "failed"}:
        st.warning(
            "引用校验状态："
            + str(validation.get("status"))
            + "。请结合下方来源核对；系统未伪造缺失引用。"
        )
    run_status.update(label="本次问答完成", state="complete", expanded=False)

    performance = result.get("performance") or {}
    rag_llm_calls = int(performance.get("llm_calls") or 0)
    total_llm_calls = rag_llm_calls
    comparison_answer: str | None = None
    if comparison_mode:
        try:
            with st.spinner("Running direct LLM comparison..."):
                response = api_client().post(
                    "/llm/chat",
                    json={"prompt": normalized_question},
                )
                response.raise_for_status()
                comparison_answer = str(response.json().get("answer", ""))
                total_llm_calls += 1
        except (httpx.HTTPError, ValueError) as exc:
            st.warning(f"RAG 已完成，但直接 LLM 对照失败（{type(exc).__name__}）。")
        else:
            st.subheader("🤖 Direct LLM Comparison")
            st.write(comparison_answer)
            st.caption("直接 LLM 未检索本地知识库，不提供 RAG 引用；仅用于效果对照。")

    st.caption(
        f"总耗时 {ms(performance.get('total_ms'))} · "
        f"使用 {performance.get('final_context_chunks', len(result.get('contexts') or []))} chunks · "
        f"本次 LLM 调用 {total_llm_calls} 次 · {answer_mode_label}"
    )

    if debug_mode:
        columns = st.columns(4)
        columns[0].metric("总耗时", ms(performance.get("total_ms")))
        columns[1].metric("Chroma 检索", ms(performance.get("chroma_search_ms")))
        columns[2].metric("LLM 首字", ms(performance.get("llm_ttft_ms")))
        columns[3].metric("LLM 完整生成", ms(performance.get("llm_full_generation_ms")))
        columns = st.columns(4)
        columns[0].metric(
            "Chunks（检索/使用）",
            f"{performance.get('raw_retrieved_chunks', '—')}/{performance.get('final_context_chunks', '—')}",
        )
        columns[1].metric("Prompt Tokens", performance.get("prompt_tokens", "—"))
        columns[2].metric("LLM 调用", total_llm_calls)
        columns[3].metric(
            "Cache",
            f"E:{cache_label(performance.get('embedding_cache_hit'))} / R:{cache_label(performance.get('retrieval_cache_hit'))}",
        )
        with st.expander("性能详情", expanded=False):
            stage_rows = {
                "请求解析": ms(performance.get("request_parse_ms")),
                "Query Rewrite（规则）": ms(performance.get("query_rewrite_ms")),
                "Query Embedding": ms(performance.get("query_embedding_ms")),
                "Chroma 检索": ms(performance.get("chroma_search_ms")),
                "去重/过滤": ms(performance.get("filter_diversify_ms")),
                "Prompt 构建": ms(performance.get("prompt_build_ms")),
                "LLM client 准备": ms(performance.get("llm_client_prepare_ms")),
                "LLM 请求建立": ms(performance.get("llm_request_establish_ms")),
                "LLM TTFT": ms(performance.get("llm_ttft_ms")),
                "LLM 完整生成": ms(performance.get("llm_full_generation_ms")),
                "引用整理": ms(performance.get("citation_organize_ms")),
                "本地处理合计": ms(performance.get("local_processing_ms")),
                "外部模型耗时": ms(performance.get("external_model_ms")),
            }
            st.table([{"阶段": key, "耗时": value} for key, value in stage_rows.items()])
            st.caption(
                f"Embedding cache: {cache_label(performance.get('embedding_cache_hit'))} · "
                f"Retrieval cache: {cache_label(performance.get('retrieval_cache_hit'))} · "
                f"Retries: {performance.get('retry_count', 0)} · "
                f"Cold start: {performance.get('cold_start', False)} · "
                f"Prewarmed: {performance.get('prewarmed', False)}"
            )
            st.caption(f"Trace ID: {result.get('trace_id', '—')}")
            st.caption(
                f"Prompt: {result.get('prompt_version', '—')} · "
                f"Evidence: {result.get('evidence_status', '—')} · "
                f"Citation validation: {validation.get('status', '—')}"
            )
            st.code(result.get("query_rewrite", normalized_question), language="text")

    contexts = result.get("contexts") or []
    with st.expander(f"📚 Retrieved Contexts ({len(contexts)})", expanded=False):
        for index, hit in enumerate(contexts, start=1):
            meta = hit.get("metadata") or {}
            distance = float(hit.get("distance") or 0.0)
            relevance = "🟢 High" if distance <= 0.60 else ("🟡 Medium" if distance <= 1.00 else "🔴 Low")
            st.markdown(
                f"**[S{index}] {meta.get('source', 'unknown')} · Page {meta.get('page', '?')} · {relevance}**"
            )
            provenance = [
                f"Paper: {meta.get('paper_id', '—')}",
                f"Section: {meta.get('section', '—')}",
            ]
            page_start, page_end = meta.get("page_start"), meta.get("page_end")
            if page_start is not None:
                provenance.append(
                    f"Pages: {page_start}" if page_start == page_end or page_end is None else f"Pages: {page_start}–{page_end}"
                )
            st.caption(
                " · ".join(provenance)
                + f" · Chunk: {meta.get('chunk_index', '?')} · Distance: {distance:.4f}"
            )
            st.markdown(f"> {hit.get('text', '')}")
            st.divider()
