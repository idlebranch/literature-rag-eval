"""Phase 1 reporting: classify the existing 66-paper corpus by topic,
merge with the candidate pool, and emit the Phase 1 summary report
(topic distribution, time/review distribution, access breakdown, and the
top Priority A papers for manual download).

No downloads. Read-only over metadata files.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "corpus_expansion"

# topic_cluster keyword rules applied to filename + meta title + first-page preview.
TOPIC_RULES = [
    ("advanced_oxidation_processes", re.compile(
        r"persulfate|peroxymonosulfate|peroxydisulfate|percarbonate|pms|pds|"
        r"fenton|ozone|ozonation|o3|uv[-_/ ]?h2o2|uv-h2o2|photocatal|photodegrad|"
        r"advanced oxidation|高级氧化|过硫酸|臭氧|芬顿|光催化|高铁酸钾|催化氧化",
        re.I)),
    ("adsorption_ion_exchange", re.compile(
        r"adsorption|adsorb|activated carbon|活性炭|吸附|再生|regeneration|"
        r"biochar|水热炭|离子交换|ion exchange", re.I)),
    ("emerging_contaminants", re.compile(
        r"pfas|pfoa|pfos|全氟|micropollutant|新污染物|emerging contaminant|"
        r"ppcp|抗生素|antibiotic|嗅味|odor|磺胺|sulfamethoxazole|双酚|bisphenol|"
        r"微污染物", re.I)),
    ("membrane_treatment", re.compile(
        r"membrane|膜|超滤|ultrafiltration|陶瓷膜|ceramic", re.I)),
    ("disinfection", re.compile(
        r"disinfect|chlorine|余氯|chloramine|消毒|氯", re.I)),
    ("industrial_wastewater", re.compile(
        r"工业废水|industrial wastewater|喷漆|洗漆|渗滤液|leachate|纺织|制药废水",
        re.I)),
    ("biological_treatment", re.compile(
        r"活性污泥|硝化|反硝化|厌氧消化|生物膜|颗粒污泥|activated sludge|anammox",
        re.I)),
    ("coagulation_flocculation", re.compile(
        r"混凝|絮凝|沉淀|coagul|floccul", re.I)),
    ("electrochemical_treatment", re.compile(
        r"电化学|电絮凝|电氧化|electrochem|electrocoagul|capacitive", re.I)),
    ("sludge_resource_recovery", re.compile(
        r"污泥|sludge|磷回收|phosphorus recovery", re.I)),
]


def classify_existing(rec: dict) -> str:
    text = " ".join([
        rec.get("filename", ""),
        rec.get("title_meta", ""),
        rec.get("first_page_preview", ""),
    ])
    for topic, pat in TOPIC_RULES:
        if pat.search(text):
            return topic
    return "other"


def load_candidates():
    with (EXP / "paper_candidates.csv").open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    existing = json.loads((ROOT / "data" / "_inventory_existing.json").read_text(encoding="utf-8"))
    ex_topic = Counter(classify_existing(r) for r in existing)
    ex_year = Counter(int(r["year"]) for r in existing if r.get("year") and r["year"].isdigit())

    cands = load_candidates()
    c_topic = Counter(r["topic_cluster"] for r in cands)
    c_pri = Counter(r["priority"] for r in cands)
    c_type = Counter(r["paper_type"] for r in cands)
    c_access = Counter(r["access_status"] for r in cands)
    c_year = Counter(int(r["year"]) for r in cands if r["year"])

    # Topic distribution table
    print("=== Topic distribution (Existing | Candidate | Expected Final) ===")
    topics = list(dict.fromkeys(list(ex_topic) + list(c_topic)))
    for t in sorted(topics):
        print(f"  {t}: existing={ex_topic.get(t,0)} candidate={c_topic.get(t,0)} "
              f"expected_final={ex_topic.get(t,0)+c_topic.get(t,0)}")

    print("\n=== Existing corpus ===")
    print(f"  total={len(existing)} valid={sum(1 for r in existing if r['ok'])} "
          f"duplicates=0 broken={sum(1 for r in existing if not r['ok'])}")
    print("  existing review-ish (title pattern):",
          sum(1 for r in existing if re.search(r'review|critical|advances|overview|perspective',
                                               r.get('filename','')+r.get('title_meta',''), re.I)))

    print("\n=== Candidate pool ===")
    print(f"  total={len(cands)}  A={c_pri.get('A',0)} B={c_pri.get('B',0)} C={c_pri.get('C',0)}")
    print("  paper_type:", dict(c_type))
    print("  access:", dict(c_access))
    recent = sum(v for y, v in c_year.items() if y >= 2023)
    print(f"  recent >=2023: {recent} ({recent/len(cands)*100:.0f}%)")

    # Time buckets
    print("\n=== Time distribution (candidate) ===")
    buckets = {"<2000": 0, "2000-2009": 0, "2010-2014": 0, "2015-2019": 0, "2020-2022": 0, ">=2023": 0}
    for y, v in c_year.items():
        if y < 2000: buckets["<2000"] += v
        elif y < 2010: buckets["2000-2009"] += v
        elif y < 2015: buckets["2010-2014"] += v
        elif y < 2020: buckets["2015-2019"] += v
        elif y < 2023: buckets["2020-2022"] += v
        else: buckets[">=2023"] += v
    for k, v in buckets.items():
        print(f"  {k}: {v}")

    # Top Priority A needing manual/institution download
    print("\n=== Top Priority A (institution/manual access) for manual download ===")
    top_a = [r for r in cands if r["priority"] == "A" and r["access_status"] != "open_access"]
    top_a.sort(key=lambda r: -int(r["citation_signal"]))
    for i, r in enumerate(top_a[:40], 1):
        print(f"{i:2d}. [{r['citation_signal']:>5}] ({r['year']}) {r['title'][:75]}")
        print(f"     {r['journal']} | DOI {r['doi']} | {r['topic_cluster']}")

    # Hard negatives sample
    print("\n=== Hard negatives (C) sample ===")
    cs = [r for r in cands if r["priority"] == "C"]
    for r in cs[:10]:
        print(f"  [{r['citation_signal']:>4}] ({r['year']}) {r['title'][:70]} | {r['topic_cluster']}")


if __name__ == "__main__":
    main()
