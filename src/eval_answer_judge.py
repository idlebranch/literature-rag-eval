"""Legacy CSV judge entrypoint using the canonical versioned judge prompt."""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from time import sleep

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.prompts import (
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_prompt,
)
from src.rag_chain import PROMPT_VERSION, answer_question


load_dotenv()
GROUNDTRUTH_PATH = Path("./groundtruth/groundtruth.example.jsonl")
OUTPUT_CSV = Path("./outputs/answer_judge_eval.csv")
OUTPUT_MD = Path("./outputs/answer_judge_report.md")


def load_groundtruth(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    return rows


def get_judge_client():
    api_key = os.getenv("JUDGE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("JUDGE_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL")
    if not api_key or not base_url or not model:
        raise RuntimeError("Judge API key, base URL, and model must be configured.")
    return OpenAI(api_key=api_key, base_url=base_url), model


def format_sources(contexts):
    items = []
    for index, hit in enumerate(contexts, start=1):
        metadata = hit["metadata"]
        text = hit["text"][:900].replace("\n", " ")
        items.append(
            f"[S{index}] {metadata.get('source')} | page {metadata.get('page')} "
            f"| distance={hit.get('distance'):.4f}\n{text}"
        )
    return "\n\n".join(items)


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return json.loads(match.group(0))
    raise ValueError(f"Cannot parse JSON from judge output: {text[:500]}")


def judge_answer(client, model, case, model_answer, retrieved_sources):
    prompt = build_judge_user_prompt(
        question=case["question"],
        ideal_answer=case.get("ideal_answer", ""),
        model_answer=model_answer,
        retrieved_sources=retrieved_sources,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return extract_json(response.choices[0].message.content or "")


def main():
    cases = load_groundtruth(GROUNDTRUTH_PATH)
    client, judge_model = get_judge_client()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in tqdm(cases, desc="Answer + Judge"):
        try:
            rag_result = answer_question(case["question"])
            sources = format_sources(rag_result["contexts"])
            judge = judge_answer(client, judge_model, case, rag_result["answer"], sources)
            row = {
                "id": case.get("id", ""),
                "answer_type": case.get("answer_type", ""),
                "question": case["question"],
                "ideal_answer": case.get("ideal_answer", ""),
                "model_answer": rag_result["answer"],
                "retrieved_sources": sources,
                "rag_prompt_version": rag_result.get("prompt_version", PROMPT_VERSION),
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "correctness_score": judge.get("correctness_score", ""),
                "evidence_relevance_score": judge.get("evidence_relevance_score", ""),
                "faithfulness_score": judge.get("faithfulness_score", ""),
                "completeness_score": judge.get("completeness_score", ""),
                "citation_score": judge.get("citation_score", ""),
                "overall_score": judge.get("overall_score", ""),
                "error_type": judge.get("error_type", ""),
                "judge_reason": judge.get("judge_reason", ""),
            }
        except Exception as exc:
            row = {
                "id": case.get("id", ""),
                "answer_type": case.get("answer_type", ""),
                "question": case.get("question", ""),
                "ideal_answer": case.get("ideal_answer", ""),
                "model_answer": f"[ERROR] {type(exc).__name__}",
                "retrieved_sources": "",
                "rag_prompt_version": PROMPT_VERSION,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "correctness_score": "",
                "evidence_relevance_score": "",
                "faithfulness_score": "",
                "completeness_score": "",
                "citation_score": "",
                "overall_score": "",
                "error_type": "script_error",
                "judge_reason": type(exc).__name__,
            }
        rows.append(row)
        sleep(0.5)

    fieldnames = list(rows[0].keys()) if rows else []
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with OUTPUT_MD.open("w", encoding="utf-8") as file:
        file.write(
            f"# RAG Answer Judge Report\n\nRAG Prompt: `{PROMPT_VERSION}`  "
            f"Judge Prompt: `{JUDGE_PROMPT_VERSION}`\n\n"
        )
        for row in rows:
            file.write(f"## {row['id']}｜{row['question']}\n\n")
            file.write(
                "Scores: "
                f"correctness={row['correctness_score']}, "
                f"evidence_relevance={row['evidence_relevance_score']}, "
                f"faithfulness={row['faithfulness_score']}, "
                f"completeness={row['completeness_score']}, "
                f"citation={row['citation_score']}, overall={row['overall_score']}\n\n"
            )
            file.write(f"Error: `{row['error_type']}` — {row['judge_reason']}\n\n---\n\n")
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved Markdown report: {OUTPUT_MD}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
