from __future__ import annotations

from types import SimpleNamespace

import httpx

import src.llm_client as llm_client
import src.rag_chain as rag_chain
import src.retriever as retriever
import src.vectorstore as vectorstore
from src.config import settings


def _hit(source: str, chunk: int, text: str, distance: float = 0.3):
    return {
        "text": text,
        "metadata": {"source": source, "page": chunk + 1, "chunk_index": chunk},
        "distance": distance,
    }


def test_chroma_client_and_collection_are_reused(monkeypatch):
    created = []

    class FakeClient:
        def __init__(self):
            self.collection = object()

        def get_or_create_collection(self, name):
            return self.collection

    def build_client(**kwargs):
        created.append(kwargs)
        return FakeClient()

    vectorstore.clear_vectorstore_caches()
    monkeypatch.setattr(vectorstore.chromadb, "PersistentClient", build_client)
    first_client = vectorstore.get_client()
    second_client = vectorstore.get_client()
    first_collection = vectorstore.get_collection()
    second_collection = vectorstore.get_collection()

    assert first_client is second_client
    assert first_collection is second_collection
    assert len(created) == 1
    vectorstore.clear_vectorstore_caches()


def test_llm_client_reuses_pool_and_disables_hidden_sdk_retries(monkeypatch):
    clients = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            clients.append(kwargs)

        def close(self):
            pass

    monkeypatch.setattr(settings, "openai_api_key", "test-only-key")
    monkeypatch.setattr(settings, "openai_base_url", "https://example.invalid/v1")
    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAI)
    llm_client.clear_llm_client_cache()

    assert llm_client.get_llm_client() is llm_client.get_llm_client()
    assert len(clients) == 1
    assert clients[0]["max_retries"] == 0
    assert isinstance(clients[0]["timeout"], httpx.Timeout)
    llm_client.clear_llm_client_cache()


def test_query_and_retrieval_cache_are_bounded_and_versioned(monkeypatch):
    calls = {"embed": 0, "search": 0}
    version = {"value": "v1"}

    def fake_embed(query):
        calls["embed"] += 1
        return [0.1, 0.2]

    def fake_search(vector, top_k):
        calls["search"] += 1
        return [
            _hit("a.pdf", 1, "完整证据 A"),
            _hit("b.pdf", 2, "完整证据 B"),
        ]

    retriever.clear_retrieval_caches()
    monkeypatch.setattr(retriever, "embed_query", fake_embed)
    monkeypatch.setattr(retriever, "search", fake_search)
    monkeypatch.setattr(retriever, "collection_version", lambda: version["value"])

    first = retriever.retrieve_with_metrics("  PFAS   限制 ", top_k=2)
    second = retriever.retrieve_with_metrics("PFAS 限制", top_k=2)
    version["value"] = "v2"
    third = retriever.retrieve_with_metrics("PFAS 限制", top_k=2)

    assert first.retrieval_cache_hit is False
    assert second.retrieval_cache_hit is True
    assert third.retrieval_cache_hit is False
    assert third.embedding_cache_hit is True
    assert calls == {"embed": 1, "search": 2}
    assert third.hits[0]["metadata"]["source"] == "a.pdf"
    retriever.clear_retrieval_caches()


def test_adjacent_duplicates_are_removed_without_truncating_chunks(monkeypatch):
    duplicate_text = "PFAS 工程证据。" * 80
    unique_text = "膜处理成本与浓缩液管理证据。" * 30
    raw = [
        _hit("same.pdf", 3, duplicate_text, 0.1),
        _hit("same.pdf", 4, duplicate_text, 0.11),
        _hit("other.pdf", 2, unique_text, 0.2),
    ]
    monkeypatch.setattr(retriever, "embed_query", lambda query: [0.1])
    monkeypatch.setattr(retriever, "search", lambda vector, top_k: raw)
    monkeypatch.setattr(retriever, "collection_version", lambda: "dedupe-test")
    monkeypatch.setattr(settings, "context_token_budget", 10_000)
    retriever.clear_retrieval_caches()

    result = retriever.retrieve_with_metrics("PFAS", top_k=3)

    assert [hit["text"] for hit in result.hits] == [duplicate_text, unique_text]
    assert result.hits[0]["text"] == duplicate_text
    retriever.clear_retrieval_caches()


def test_normal_rag_has_one_llm_call_and_stage_metrics(monkeypatch):
    hit = _hit("pfas.pdf", 3, "PFAS 处理存在能耗与浓缩液限制。", 0.2)
    calls = {"llm": 0}

    monkeypatch.setattr(rag_chain, "retrieve", lambda *args, **kwargs: [hit])

    def fake_llm(*args, **kwargs):
        calls["llm"] += 1
        return SimpleNamespace(
            content="回答 [S1]",
            usage={"prompt_tokens": 100, "completion_tokens": 10},
            model="fake",
            retry_count=0,
            client_prepare_ms=0.1,
            full_generation_ms=2.0,
        )

    monkeypatch.setattr(rag_chain, "chat_completion_result", fake_llm)
    result = rag_chain.answer_question("PFAS 有什么限制？")

    assert calls["llm"] == 1
    assert result["performance"]["llm_calls"] == 1
    assert result["performance"]["prompt_tokens"] == 100
    assert result["contexts"] == [hit]


def test_streaming_emits_real_tokens_before_final_citations(monkeypatch):
    hit = _hit("pfas.pdf", 3, "PFAS 工程限制证据。", 0.2)
    retrieval_result = retriever.RetrievalResult(
        hits=[hit],
        expanded_query="PFAS expanded",
        query_rewrite_ms=0.1,
        query_embedding_ms=1.0,
        chroma_search_ms=2.0,
        filter_diversify_ms=0.1,
        embedding_cache_hit=False,
        retrieval_cache_hit=False,
        raw_hit_count=1,
        context_estimated_tokens=20,
        collection_version="v1",
    )

    def fake_retrieve(*args, **kwargs):
        callback = kwargs.get("progress_callback")
        if callback:
            callback("embedding")
            callback("chroma")
        return retrieval_result

    state = SimpleNamespace(
        client_prepare_ms=0.1,
        request_establish_ms=0.2,
        ttft_ms=1.5,
        full_generation_ms=3.0,
        retry_count=0,
        usage=None,
        model="fake",
    )
    monkeypatch.setattr(rag_chain, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        rag_chain,
        "stream_chat_completion",
        lambda *args, **kwargs: (iter(["回答", " [S1]"]), state),
    )

    events = list(rag_chain.stream_answer_question("PFAS 有什么限制？", top_k=1))
    token_indexes = [i for i, event in enumerate(events) if event["type"] == "token"]
    final_index = next(i for i, event in enumerate(events) if event["type"] == "final")
    final = events[final_index]["result"]

    assert token_indexes and max(token_indexes) < final_index
    assert all("contexts" not in events[i] for i in token_indexes)
    assert final["answer"] == "回答 [S1]"
    assert final["contexts"] == [hit]
    assert final["performance"]["llm_ttft_ms"] == 1.5
    assert final["performance"]["llm_calls"] == 1
