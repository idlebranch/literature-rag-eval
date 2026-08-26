from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.rag_chain as rag_chain
from src.citation_validation import validate_citations
from src.prompts import (
    JUDGE_SYSTEM_PROMPT,
    RAG_ANSWER_PROMPT_VERSION,
    build_answer_system_prompt,
    build_judge_user_prompt,
)
from src.retriever import format_context


def hit(source: str, chunk: int, text: str, distance: float = 0.2):
    return {
        "text": text,
        "metadata": {"source": source, "page": chunk + 1, "chunk_index": chunk},
        "distance": distance,
    }


def fake_result(content: str):
    return SimpleNamespace(
        content=content,
        usage={"prompt_tokens": 100},
        model="fake",
        retry_count=0,
        client_prepare_ms=0.0,
        full_generation_ms=1.0,
        finish_reason="stop",
    )


def test_clear_answer_uses_one_llm_and_valid_citation(monkeypatch):
    contexts = [hit("a.pdf", 1, "PFAS 吸附受成本和再生限制。")]
    calls = []
    monkeypatch.setattr(rag_chain, "retrieve", lambda *args, **kwargs: contexts)
    monkeypatch.setattr(
        rag_chain,
        "chat_completion_result",
        lambda *args, **kwargs: calls.append((args, kwargs)) or fake_result("存在成本限制 [S1]。"),
    )

    result = rag_chain.answer_question("PFAS 吸附的限制是什么？")

    assert len(calls) == 1
    assert result["citation_validation"]["status"] == "passed"
    assert result["prompt_version"] == RAG_ANSWER_PROMPT_VERSION
    assert result["answer_mode"] == "quick"


def test_cross_chunk_context_preserves_stable_source_boundaries(monkeypatch):
    contexts = [
        hit("a.pdf", 1, "成本证据"),
        hit("b.pdf", 2, "稳定性证据"),
    ]
    captured = {}
    monkeypatch.setattr(rag_chain, "retrieve", lambda *args, **kwargs: contexts)

    def complete(system, user, **kwargs):
        captured["user"] = user
        return fake_result("成本受限 [S1]；稳定性也受限 [S2]。")

    monkeypatch.setattr(rag_chain, "chat_completion_result", complete)
    result = rag_chain.answer_question("综合说明 PFAS 处理的成本与稳定性限制")

    assert "[S1]" in captured["user"] and "[S2]" in captured["user"]
    assert "document: a.pdf" in captured["user"]
    assert result["citation_validation"]["used_source_ids"] == ["S1", "S2"]


def test_no_answer_low_score_does_not_call_llm(monkeypatch):
    contexts = [hit("x.pdf", 1, "仅主题相似", rag_chain.settings.max_retrieval_distance + 0.2)]
    monkeypatch.setattr(rag_chain, "retrieve", lambda *args, **kwargs: contexts)
    monkeypatch.setattr(
        rag_chain,
        "chat_completion_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM must not run")),
    )
    result = rag_chain.answer_question("知识库没有的结论")
    assert result["answer"] == "当前知识库中没有足够证据回答该问题。"
    assert result["performance"]["llm_calls"] == 0


def test_exact_value_request_without_numeric_evidence_falls_back(monkeypatch):
    contexts = [hit("a.pdf", 1, "该工艺受到水体基质影响，但未报告定量条件。")]
    monkeypatch.setattr(rag_chain, "retrieve", lambda *args, **kwargs: contexts)
    monkeypatch.setattr(
        rag_chain,
        "chat_completion_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM must not run")),
    )
    result = rag_chain.answer_question("请给出最佳剂量的精确实验条件")
    assert result["fallback_reason"] == "insufficient_evidence"


def test_opposite_evidence_not_auto_conflict(monkeypatch):
    # The keyword-based conflict heuristic is removed: opposite-direction phrases
    # across sources no longer auto-flag "conflicting" evidence status.
    contexts = [
        hit("a.pdf", 1, "DOM 会提高反应速率。"),
        hit("b.pdf", 2, "DOM 会降低反应速率。"),
    ]
    captured = {}
    monkeypatch.setattr(rag_chain, "retrieve", lambda *args, **kwargs: contexts)

    def complete(system, user, **kwargs):
        captured["user"] = user
        return fake_result("来源结果不同：提高 [S1]，降低 [S2]。")

    monkeypatch.setattr(rag_chain, "chat_completion_result", complete)
    result = rag_chain.answer_question("DOM 对反应速率有什么影响？", answer_mode="detailed")
    assert result["evidence_status"] == "available"
    assert "自动冲突判定" in captured["user"]


def test_ambiguous_question_stops_before_retrieval(monkeypatch):
    monkeypatch.setattr(
        rag_chain,
        "retrieve",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not retrieve")),
    )
    result = rag_chain.answer_question("处理效果怎么样？")
    assert result["fallback_reason"] == "needs_clarification"
    assert "补充" in result["answer"]


@pytest.mark.parametrize(
    "mode,max_tokens,mode_text",
    [("quick", 1200, "快速回答"), ("detailed", 2200, "详细回答")],
)
def test_answer_modes_share_fact_rules_and_only_change_depth(
    monkeypatch, mode, max_tokens, mode_text
):
    contexts = [hit("a.pdf", 1, "证据")]
    captured = {}
    monkeypatch.setattr(rag_chain, "retrieve", lambda *args, **kwargs: contexts)

    def complete(system, user, **kwargs):
        captured.update(system=system, kwargs=kwargs)
        return fake_result("结论 [S1]")

    monkeypatch.setattr(rag_chain, "chat_completion_result", complete)
    result = rag_chain.answer_question("PFAS 限制？", answer_mode=mode)
    assert "只能使用本次提供" in captured["system"]
    assert mode_text in captured["system"]
    assert captured["kwargs"]["max_tokens"] == max_tokens
    assert result["answer_mode"] == mode


def test_citation_validator_removes_out_of_range_and_flags_unmapped_page():
    contexts = [hit("a.pdf", 1, "证据")]
    answer, validation = validate_citations("结论 [S9]，见第 99 页。", contexts)
    assert "[S9]" not in answer
    assert "[页码无法核验]" in answer
    assert validation["status"] == "failed"
    assert validation["invalid_source_ids"] == ["S9"]


def test_citation_validator_removes_unsupported_literature_claim():
    contexts = [hit("a.pdf", 1, "证据")]
    answer, validation = validate_citations("研究表明该方法有效。另有证据 [S1]。", contexts)
    assert "研究表明" not in answer
    assert validation["claims_removed_without_citation"] == 1


def test_stream_withholds_citations_until_validated_final(monkeypatch):
    contexts = [hit("a.pdf", 1, "证据")]
    monkeypatch.setattr(rag_chain, "retrieve", lambda *args, **kwargs: contexts)
    state = SimpleNamespace(
        client_prepare_ms=0.0,
        request_establish_ms=0.1,
        ttft_ms=1.0,
        full_generation_ms=2.0,
        retry_count=0,
        usage=None,
        model="fake",
        finish_reason="stop",
    )
    monkeypatch.setattr(
        rag_chain,
        "stream_chat_completion",
        lambda *args, **kwargs: (iter(["结论 ", "[", "S1", "]。"]), state),
    )
    events = list(rag_chain.stream_answer_question("PFAS 限制？"))
    streamed = "".join(event["content"] for event in events if event["type"] == "token")
    final = next(event["result"] for event in events if event["type"] == "final")
    assert "[S1]" not in streamed
    assert "[S1]" in final["answer"]
    assert final["citation_validation"]["status"] == "passed"


def test_prompt_blocks_document_instructions_and_judge_is_evidence_aware():
    system = build_answer_system_prompt("quick")
    judge = build_judge_user_prompt(
        question="Q",
        ideal_answer="I",
        model_answer="M",
        retrieved_sources="S",
    )
    assert "外部不可信数据" in system
    assert "思维过程" in system
    assert "语言流畅" in JUDGE_SYSTEM_PROMPT
    assert "correctness_score" in judge
    assert "正确常识若没有证据支持" in judge


def test_context_formatter_excludes_unneeded_metadata():
    context = format_context(
        [
            {
                "text": "正文",
                "metadata": {
                    "source": "a.pdf",
                    "page": 3,
                    "chunk_index": 7,
                    "paper_id": "unused",
                    "secret": "drop-me",
                },
                "distance": 0.2,
            }
        ]
    )
    assert "document: a.pdf" in context
    assert "page: 3" in context
    assert "chunk_id: 7" in context
    assert "paper_id" not in context and "drop-me" not in context
