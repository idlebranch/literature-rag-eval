import csv
import json
import os
import re
from pathlib import Path
from time import sleep

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.rag_chain import answer_question

load_dotenv()

GROUNDTRUTH_PATH = Path("./groundtruth/groundtruth.example.jsonl")
OUTPUT_CSV = Path("./outputs/answer_judge_eval.csv")
OUTPUT_MD = Path("./outputs/answer_judge_report.md")


def load_groundtruth(path: Path):
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


def get_judge_client():
    api_key = os.getenv("JUDGE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("JUDGE_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL")

    if not api_key:
        raise RuntimeError("Missing JUDGE_OPENAI_API_KEY or OPENAI_API_KEY in .env")
    if not base_url:
        raise RuntimeError("Missing JUDGE_OPENAI_BASE_URL or OPENAI_BASE_URL in .env")
    if not model:
        raise RuntimeError("Missing JUDGE_MODEL or LLM_MODEL in .env")

    return OpenAI(api_key=api_key, base_url=base_url), model


def format_sources(contexts):
    items = []
    for i, hit in enumerate(contexts, start=1):
        meta = hit["metadata"]
        text = hit["text"][:900].replace("\n", " ")
        items.append(
            f"[S{i}] {meta.get('source')} | page {meta.get('page')} | distance={hit.get('distance'):.4f}\n{text}"
        )
    return "\n\n".join(items)


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Cannot parse JSON from judge output: {text[:500]}")


def judge_answer(client, model, case, model_answer, retrieved_sources):
    prompt = f"""
你是一个严谨的 RAG 答案评测员，任务是评估“专业文献 RAG 系统”的回答质量。

请根据：
1. 用户问题
2. ideal_answer
3. model_answer
4. retrieved_sources

对回答进行评分。

评分维度：
- faithfulness_score：回答是否忠实于 retrieved_sources，1-5分。
- completeness_score：回答是否覆盖 ideal_answer 的关键要点，1-5分。
- citation_score：回答中的引用是否支撑对应结论，1-5分。
- overall_score：综合评分，1-5分。

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
{case["question"]}

ideal_answer：
{case.get("ideal_answer", "")}

model_answer：
{model_answer}

retrieved_sources：
{retrieved_sources}

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

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个严格、保守、专业的RAG评测员。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    content = resp.choices[0].message.content or ""
    return extract_json(content)


def main():
    cases = load_groundtruth(GROUNDTRUTH_PATH)
    client, judge_model = get_judge_client()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for case in tqdm(cases, desc="Answer + Judge"):
        qid = case.get("id", "")
        question = case["question"]

        try:
            rag_result = answer_question(question)
            model_answer = rag_result["answer"]
            retrieved_sources = format_sources(rag_result["contexts"])

            judge = judge_answer(
                client=client,
                model=judge_model,
                case=case,
                model_answer=model_answer,
                retrieved_sources=retrieved_sources,
            )

            row = {
                "id": qid,
                "answer_type": case.get("answer_type", ""),
                "question": question,
                "ideal_answer": case.get("ideal_answer", ""),
                "model_answer": model_answer,
                "retrieved_sources": retrieved_sources,
                "faithfulness_score": judge.get("faithfulness_score", ""),
                "completeness_score": judge.get("completeness_score", ""),
                "citation_score": judge.get("citation_score", ""),
                "overall_score": judge.get("overall_score", ""),
                "error_type": judge.get("error_type", ""),
                "judge_reason": judge.get("judge_reason", ""),
            }

        except Exception as e:
            row = {
                "id": qid,
                "answer_type": case.get("answer_type", ""),
                "question": question,
                "ideal_answer": case.get("ideal_answer", ""),
                "model_answer": f"[ERROR] {e}",
                "retrieved_sources": "",
                "faithfulness_score": "",
                "completeness_score": "",
                "citation_score": "",
                "overall_score": "",
                "error_type": "script_error",
                "judge_reason": str(e),
            }

        rows.append(row)
        sleep(0.5)

    fieldnames = [
        "id",
        "answer_type",
        "question",
        "ideal_answer",
        "model_answer",
        "retrieved_sources",
        "faithfulness_score",
        "completeness_score",
        "citation_score",
        "overall_score",
        "error_type",
        "judge_reason",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with OUTPUT_MD.open("w", encoding="utf-8") as f:
        f.write("# RAG Answer Judge Report\n\n")
        for r in rows:
            f.write(f"## {r['id']}｜{r['question']}\n\n")
            f.write(f"**Answer Type:** {r['answer_type']}\n\n")
            f.write(f"**Scores:** faithfulness={r['faithfulness_score']}, completeness={r['completeness_score']}, citation={r['citation_score']}, overall={r['overall_score']}\n\n")
            f.write(f"**Error Type:** {r['error_type']}\n\n")
            f.write(f"**Judge Reason:** {r['judge_reason']}\n\n")
            f.write("### Ideal Answer\n\n")
            f.write(str(r["ideal_answer"]) + "\n\n")
            f.write("### Model Answer\n\n")
            f.write(str(r["model_answer"]) + "\n\n")
            f.write("### Retrieved Sources\n\n")
            f.write(str(r["retrieved_sources"]) + "\n\n")
            f.write("---\n\n")

    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved Markdown report: {OUTPUT_MD}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
