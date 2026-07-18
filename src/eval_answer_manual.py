import argparse
import csv
import json
from pathlib import Path
from time import sleep

from tqdm import tqdm

from src.rag_chain import answer_question


GROUNDTRUTH_PATH = Path("./groundtruth/groundtruth.example.jsonl")
OUTPUT_PATH = Path("./outputs/answer_eval_manual.csv")


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


def format_sources(contexts):
    items = []
    for i, hit in enumerate(contexts, start=1):
        meta = hit["metadata"]
        items.append(
            f"[S{i}] {meta.get('source')} | page {meta.get('page')} | distance={hit.get('distance'):.4f}"
        )
    return " || ".join(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N cases")
    args = parser.parse_args()

    cases = load_groundtruth(GROUNDTRUTH_PATH)
    if args.limit:
        cases = cases[: args.limit]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

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
        "error_type",
        "notes",
    ]

    rows = []

    for case in tqdm(cases, desc="Generating answers"):
        question = case["question"]

        try:
            result = answer_question(question)
            model_answer = result["answer"]
            retrieved_sources = format_sources(result["contexts"])
        except Exception as e:
            model_answer = f"[ERROR] {e}"
            retrieved_sources = ""

        rows.append(
            {
                "id": case.get("id", ""),
                "answer_type": case.get("answer_type", ""),
                "question": question,
                "ideal_answer": case.get("ideal_answer", ""),
                "model_answer": model_answer,
                "retrieved_sources": retrieved_sources,
                "faithfulness_score": "",
                "completeness_score": "",
                "citation_score": "",
                "error_type": "",
                "notes": "",
            }
        )

        sleep(0.5)

    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
