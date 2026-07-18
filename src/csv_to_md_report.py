"""Convert outputs/answer_eval_manual.csv into a human-readable Markdown report."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "outputs" / "answer_eval_manual.csv"
MD_PATH = ROOT / "outputs" / "answer_eval_manual.md"


def format_sources(raw: str) -> str:
    if not raw:
        return "_(无)_"
    items = [seg.strip() for seg in raw.split("||") if seg.strip()]
    return "\n".join(f"- {item}" for item in items)


def main() -> None:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    lines = [
        "# 人工评估报告：answer_eval_manual",
        "",
        f"共 {len(rows)} 个问题。每题预留 faithfulness / completeness / citation 三个评分字段，",
        "评分范围建议 0–2（0=不符合，1=部分符合，2=完全符合），评分人填入后保存即可。",
        "",
        "---",
        "",
    ]

    for idx, row in enumerate(rows, 1):
        qid = row.get("id", f"q{idx:03d}")
        qtype = row.get("answer_type", "").strip()
        question = (row.get("question") or "").strip()
        ideal = (row.get("ideal_answer") or "").strip()
        model = (row.get("model_answer") or "").strip()
        sources = format_sources((row.get("retrieved_sources") or "").strip())

        header = f"## {qid}"
        if qtype:
            header += f"  ·  _{qtype}_"
        lines.append(header)
        lines.append("")

        lines.append("### Question")
        lines.append(question if question else "_(空)_")
        lines.append("")

        lines.append("### Ideal Answer")
        lines.append(ideal if ideal else "_(空)_")
        lines.append("")

        lines.append("### Model Answer")
        lines.append(model if model else "_(空)_")
        lines.append("")

        lines.append("### Retrieved Sources")
        lines.append(sources)
        lines.append("")

        lines.append("### 人工评分")
        lines.append("")
        lines.append("| 维度 | 评分 (0/1/2) | 备注 |")
        lines.append("| --- | --- | --- |")
        lines.append("| faithfulness（事实一致性） |  |  |")
        lines.append("| completeness（要点覆盖度） |  |  |")
        lines.append("| citation（引用正确性） |  |  |")
        lines.append("")
        lines.append("**错误类型 / error_type：**")
        lines.append("")
        lines.append("**总体备注 / notes：**")
        lines.append("")
        lines.append("---")
        lines.append("")

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {MD_PATH} ({len(rows)} questions)")


if __name__ == "__main__":
    main()
