"""Convert the remaining 216-paper list to EndNote-importable formats (.ris and .enw)."""
from __future__ import annotations

import csv
from pathlib import Path

SRC = Path(r"C:\Users\10475\AI_PROJECT\literature_rag_eval_code\data\corpus_expansion\manual_download_remaining.csv")
OUT_RIS = SRC.parent / "remaining_for_endnote.ris"
OUT_ENW = SRC.parent / "remaining_for_endnote.enw"


def esc(s: str) -> str:
    return (s or "").strip()


rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
rows.sort(key=lambda r: (r.get("priority_final", "Z"), r.get("topic_cluster", ""), int(r.get("year") or 0)))

# ---------- RIS ----------
ris_lines = []
for r in rows:
    ris_lines.append("TY  - JOUR")
    ris_lines.append(f"TI  - {esc(r['title'])}")
    if r.get("year"):
        ris_lines.append(f"PY  - {esc(r['year'])}")
    if r.get("journal"):
        ris_lines.append(f"JO  - {esc(r['journal'])}")
        ris_lines.append(f"T2  - {esc(r['journal'])}")
    if r.get("doi"):
        ris_lines.append(f"DO  - {esc(r['doi'])}")
    url = r.get("publisher_url") or (f"https://doi.org/{r['doi']}" if r.get("doi") else "")
    if url:
        ris_lines.append(f"UR  - {url}")
    ris_lines.append(f"KW  - {esc(r['topic_cluster'])}")
    ris_lines.append(f"N1  - Priority {esc(r['priority_final'])} | {esc(r['route_suggestion'])}")
    ris_lines.append("ER  - ")
    ris_lines.append("")
OUT_RIS.write_text("\n".join(ris_lines), encoding="utf-8-sig")

# ---------- ENW (EndNote tagged import) ----------
enw_lines = []
for r in rows:
    enw_lines.append("%0 Journal Article")
    enw_lines.append(f"%T {esc(r['title'])}")
    if r.get("year"):
        enw_lines.append(f"%D {esc(r['year'])}")
    if r.get("journal"):
        enw_lines.append(f"%J {esc(r['journal'])}")
    if r.get("doi"):
        enw_lines.append(f"%R {esc(r['doi'])}")
    url = r.get("publisher_url") or (f"https://doi.org/{r['doi']}" if r.get("doi") else "")
    if url:
        enw_lines.append(f"%U {url}")
    enw_lines.append(f"%K {esc(r['topic_cluster'])}")
    enw_lines.append(f"%1 Priority {esc(r['priority_final'])} | {esc(r['route_suggestion'])}")
    enw_lines.append("")
OUT_ENW.write_text("\n".join(enw_lines), encoding="utf-8")

print(f"records: {len(rows)}")
print("RIS ->", OUT_RIS, OUT_RIS.stat().st_size, "bytes")
print("ENW ->", OUT_ENW, OUT_ENW.stat().st_size, "bytes")
