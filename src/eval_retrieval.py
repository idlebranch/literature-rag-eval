import json
from pathlib import Path
import pandas as pd
from src.config import settings
from src.retriever import retrieve


GROUNDTRUTH_PATH = Path("./groundtruth/groundtruth.example.jsonl")
OUTPUT_PATH = Path(settings.output_dir) / "retrieval_eval.csv"


def load_groundtruth(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def expected_source_hit(hits, expected_sources):
    retrieved_sources = [h["metadata"]["source"] for h in hits]
    retrieved_papers = [h["metadata"]["paper_id"] for h in hits]

    for expected in expected_sources:
        for src, paper in zip(retrieved_sources, retrieved_papers):
            if expected in src or expected in paper:
                return True
    return False


def main():
    if not GROUNDTRUTH_PATH.exists():
        raise FileNotFoundError(f"Groundtruth file not found: {GROUNDTRUTH_PATH}")

    cases = load_groundtruth(GROUNDTRUTH_PATH)
    results = []

    for case in cases:
        question = case["question"]
        expected_sources = case.get("expected_sources", [])
        hits = retrieve(question, top_k=settings.top_k)

        hit = expected_source_hit(hits, expected_sources)

        results.append(
            {
                "id": case.get("id"),
                "question": question,
                "expected_sources": "; ".join(expected_sources),
                "hit_at_k": int(hit),
                "top1_source": hits[0]["metadata"]["source"] if hits else "",
                "top1_page": hits[0]["metadata"]["page"] if hits else "",
                "retrieved_sources": " | ".join(
                    [h["metadata"]["source"] for h in hits]
                ),
            }
        )

    df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(df)
    print("\nSummary:")
    print(f"Cases: {len(df)}")
    print(f"Hit@{settings.top_k}: {df['hit_at_k'].mean():.3f}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
