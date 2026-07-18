"""V3 end-to-end runner: groundtruth → canonical JSON.

Two steps:
    1. run_rag    — call retriever + LLM on each question, capture chunks + answer.
    2. judge_run  — populate JudgeScore via LLM-as-judge (separate API).

Both write incrementally: every question that completes is flushed to JSON,
so a mid-batch crash (API key revoked, rate limit, Ctrl+C) preserves partial
results. Re-run with `judge_run(... only_missing=True)` to resume.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.config import settings
from src.eval.health import require_healthy
from src.eval.io import compute_summary, load_run, save_run
from src.eval.schema import (
    EvalRun,
    JudgeScore,
    QuestionResult,
    RetrievedSource,
    RunConfig,
)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def load_groundtruth(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
    return rows


def parse_judge_json(text: str) -> Dict[str, Any]:
    """Robust extractor: try strict JSON first, then regex on first {...} block."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Cannot parse JSON from judge output: {text[:300]}")


def build_judge_prompt(qr: QuestionResult, sources_block: str) -> str:
    """LLM-as-judge prompt. Returns plain text expecting JSON back."""
    return f"""你是一个严谨的 RAG 答案评测员，任务是评估专业文献 RAG 系统的回答质量。

请根据：
1. 用户问题
2. ideal_answer
3. model_answer
4. retrieved_sources

对回答进行评分。

评分维度：
- faithfulness_score：回答是否忠实于 retrieved_sources，1-5 分。
- completeness_score：回答是否覆盖 ideal_answer 的关键要点，1-5 分。
- citation_score：回答中的引用是否支撑对应结论，1-5 分。
- overall_score：综合评分，1-5 分。

错误类型 error_type 只能从以下选：
none, retrieval_error, generation_error, citation_error, incomplete_answer, hallucination, insufficient_context

评分标准：
5 = 很好，基本可直接使用
4 = 较好，只有轻微遗漏
3 = 可用但不完整
2 = 有明显错误、遗漏或引用问题
1 = 不可用、答非所问或幻觉严重

请只输出 JSON，不要输出额外解释。

用户问题：
{qr.question}

ideal_answer：
{qr.ideal_answer}

model_answer：
{qr.model_answer}

retrieved_sources：
{sources_block}

输出 JSON 格式：
{{
  "faithfulness_score": 1,
  "completeness_score": 1,
  "citation_score": 1,
  "overall_score": 1,
  "error_type": "none",
  "judge_reason": "一句话说明评分理由"
}}
"""


def _format_sources_block(sources: List[RetrievedSource]) -> str:
    if not sources:
        return "(无检索结果)"
    blocks = []
    for s in sources:
        snippet = (s.chunk_text or "").replace("\n", " ")[:900]
        blocks.append(
            f"[{s.sid}] {s.source} | page {s.page} | distance={s.distance:.4f}\n{snippet}"
        )
    return "\n\n".join(blocks)


def run_rag(
    groundtruth: Path,
    run_id: str,
    output_path: Path,
    prompt_version: str = "",
    sleep_secs: float = 0.5,
    limit: Optional[int] = None,
    skip_preflight: bool = False,
) -> EvalRun:
    """Run RAG over all groundtruth questions, save EvalRun as JSON (incrementally)."""
    from src.rag_chain import SYSTEM_PROMPT, answer_question

    if not skip_preflight:
        require_healthy()

    cases = load_groundtruth(groundtruth)
    if limit:
        cases = cases[:limit]

    config = RunConfig(
        rag_model=settings.llm_model,
        rag_prompt_version=prompt_version,
        rag_prompt_hash=prompt_hash(SYSTEM_PROMPT),
        embedding_model=settings.embedding_model,
        top_k=settings.top_k,
        groundtruth_file=str(groundtruth),
    )
    run = EvalRun(
        run_id=run_id,
        timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        config=config,
        notes=f"prompt_version={prompt_version}",
    )

    for case in tqdm(cases, desc="RAG"):
        qid = case.get("id", "")
        question = case.get("question", "")
        try:
            result = answer_question(question)
            retrieved = [
                RetrievedSource(
                    sid=f"S{i+1}",
                    source=h["metadata"]["source"],
                    page=int(h["metadata"]["page"]),
                    distance=float(h["distance"]),
                    chunk_text=(h.get("text") or "")[:2000],
                )
                for i, h in enumerate(result["contexts"])
            ]
            qr = QuestionResult(
                qid=qid,
                question=question,
                ideal_answer=case.get("ideal_answer", ""),
                model_answer=result["answer"],
                answer_type=case.get("answer_type", ""),
                retrieved=retrieved,
            )
        except Exception as e:
            qr = QuestionResult(
                qid=qid,
                question=question,
                ideal_answer=case.get("ideal_answer", ""),
                model_answer="",
                answer_type=case.get("answer_type", ""),
                error=f"[ERROR] {type(e).__name__}: {e}",
            )
        run.results.append(qr)
        run.summary = compute_summary(run)
        save_run(run, output_path)
        sleep(sleep_secs)

    return run


def _get_judge_client(judge_model: Optional[str]):
    from openai import OpenAI

    api_key = os.getenv("JUDGE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("JUDGE_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = judge_model or os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL")
    if not api_key:
        raise SystemExit("Missing JUDGE_OPENAI_API_KEY or OPENAI_API_KEY in .env")
    if not model:
        raise SystemExit("Missing JUDGE_MODEL or LLM_MODEL in .env")
    return OpenAI(api_key=api_key, base_url=base_url), model


def judge_run(
    run_path: Path,
    judge_model: Optional[str] = None,
    sleep_secs: float = 0.5,
    only_missing: bool = True,
    request_timeout: float = 60.0,
) -> EvalRun:
    """Populate JudgeScore on an existing EvalRun JSON (in-place, incremental)."""
    run = load_run(run_path)
    client, model = _get_judge_client(judge_model)

    for qr in tqdm(run.results, desc="Judge"):
        if only_missing and qr.judge is not None:
            continue
        if not qr.model_answer:
            # RAG never produced an answer — nothing to score.
            continue
        # Strip any prior "[JUDGE ERROR] ..." lines so a re-judge can succeed cleanly.
        if qr.error and "[JUDGE ERROR]" in qr.error:
            kept = [ln for ln in qr.error.splitlines() if "[JUDGE ERROR]" not in ln]
            qr.error = "\n".join(kept).strip() or None
        sources_block = _format_sources_block(qr.retrieved)
        prompt = build_judge_prompt(qr, sources_block)

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个严格、保守、专业的RAG评测员。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                timeout=request_timeout,
            )
            if not hasattr(resp, "choices"):
                # Some proxies return HTML (anti-bot) or plain strings on error;
                # surface a clear message instead of letting AttributeError leak.
                raise RuntimeError(
                    f"Unexpected response type from judge endpoint: {type(resp).__name__}; "
                    f"first 200 chars: {str(resp)[:200]!r}"
                )
            content = resp.choices[0].message.content or ""
            j = parse_judge_json(content)
            qr.judge = JudgeScore(
                faithfulness=int(j.get("faithfulness_score", 0) or 0),
                completeness=int(j.get("completeness_score", 0) or 0),
                citation=int(j.get("citation_score", 0) or 0),
                overall=int(j.get("overall_score", 0) or 0),
                error_type=j.get("error_type", "none") or "none",
                reason=j.get("judge_reason", "") or "",
                judge_model=model,
                judged_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            )
        except Exception as e:
            prev = qr.error or ""
            qr.error = (prev + "\n" if prev else "") + f"[JUDGE ERROR] {type(e).__name__}: {e}"

        run.summary = compute_summary(run)
        run.config.judge_model = model
        save_run(run, run_path)
        sleep(sleep_secs)

    return run
