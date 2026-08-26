"""Local-only finalize (v2): produce the COMPLETE remaining manual-download list.

Remaining = main list (240) - already in manifest (downloaded) - rejected.
Covers both "confirmed no OA / failed" papers AND papers that were never
checked before the process was stopped, so the user gets one complete list.
Routing is derived from DOI publisher prefix (no network needed).
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "corpus_expansion"

main = list(csv.DictReader(open(EXP / "papers_to_download.csv", encoding="utf-8-sig")))
mani = list(csv.DictReader(open(EXP / "paper_manifest.csv", encoding="utf-8-sig"))) if (EXP / "paper_manifest.csv").exists() else []
rej = list(csv.DictReader(open(EXP / "rejected_papers.csv", encoding="utf-8-sig"))) if (EXP / "rejected_papers.csv").exists() else []

done_dois = {r["doi"].lower() for r in mani}
rej_dois = {r["doi"].lower() for r in rej}

state = json.loads((EXP / "step1_state.json").read_text(encoding="utf-8"))
checked = {r["doi"].lower(): r["status"] for r in state.get("results", [])}


def route(doi):
    if doi.startswith("10.1016/"):
        return "校园网 ScienceDirect 人工下载"
    if doi.startswith("10.1021/"):
        return "校园网 ACS Publications 人工下载"
    if doi.startswith("10.1002/"):
        return "校园网 Wiley 人工下载"
    if doi.startswith("10.1007/"):
        return "校园网 Springer 人工下载"
    if doi.startswith("10.1039/"):
        return "校园网 RSC 人工下载"
    return "CASHL/CALIS 文献传递"


rows = []
for r in main:
    doi = r["doi"].lower()
    if doi in done_dois or doi in rej_dois:
        continue
    status = checked.get(doi, "未检测")
    rows.append({
        "candidate_id": r["candidate_id"],
        "priority_final": r["priority_final"],
        "title": r["title"],
        "year": r["year"],
        "journal": r["journal"],
        "doi": r["doi"],
        "topic_cluster": r["topic_cluster"],
        "oa_check_result": status,
        "route_suggestion": route(r["doi"]),
        "publisher_url": f"https://doi.org/{r['doi']}",
        "oa_url": r.get("oa_url", ""),
    })

out = EXP / "manual_download_remaining.csv"
with out.open("w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)

from collections import Counter
print(f"total remaining manual: {len(rows)}")
print("by route:", dict(Counter(r["route_suggestion"] for r in rows)))
print("by priority:", dict(Counter(r["priority_final"] for r in rows)))
print("by oa_check_result:", dict(Counter(r["oa_check_result"] for r in rows)))
print(f"manifest rows now: {len(mani)}")
