"""Phase 1 selection: score and curate the raw OpenAlex pool into a
350-450 candidate pool with Priority A/B/C, topic balance, review quota,
and diversity constraints. Outputs paper_candidates.csv + stats.

No downloads. Metadata only.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "corpus_expansion"

# Target journal signals (per prompt). Match by lowercase substring.
TOP_JOURNALS = [
    "nature water",
    "environmental science & technology",   # ES&T (also matches Letters/ES&T Water variants)
    "water research",                        # also matches Water Research X
    "journal of hazardous materials",
    "chemical engineering journal",
    "applied catalysis b",
    "journal of membrane science",
    "separation and purification technology",
    "environment international",
    "environmental pollution",
    "chemosphere",
    "science of the total environment",
    "bioresource technology",
    "desalination",
]
# Strong whitelist beyond TOP for institution-access likelihood
EXTRA_JOURNALS = [
    "acses&t water", "acs es&t engineering", "es&t water",
    "environmental science: water research & technology",
    "environmental science: nano", "journal of environmental management",
    "environmental research", "ecotoxicology and environmental safety",
    "journal of environmental chemical engineering",
    "water environment research", "environmental technology",
    "journal of water process engineering", "ultrasonics sonochemistry",
    "process safety and environmental protection",
]

CORE_TOPICS = {"advanced_oxidation_processes", "adsorption_ion_exchange", "emerging_contaminants"}
# per-topic target counts (candidate pool 350-450)
TOPIC_TARGETS = {
    "advanced_oxidation_processes": 58,
    "adsorption_ion_exchange": 52,
    "emerging_contaminants": 52,
    "membrane_treatment": 42,
    "biological_treatment": 40,
    "industrial_wastewater": 36,
    "disinfection": 30,
    "electrochemical_treatment": 26,
    "coagulation_flocculation": 24,
    "sludge_resource_recovery": 22,
    "hard_negative": 30,
}
MAX_PER_FIRST_AUTHOR = 3
MAX_PER_JOURNAL = 10
RECENT_RATIO = 0.22   # >=2023 share per topic
REVIEW_RATIO = 0.25   # review cap per topic
MIN_YEAR = 1990

REVIEW_PAT = re.compile(
    r"\b(review|critical review|systematic review|meta[- ]analysis|"
    r"perspective|minireview|tutorial|roadmap|state[- ]of[- ]the[- ]art)\b",
    re.IGNORECASE,
)


def journal_match(j: str) -> str:
    jl = (j or "").lower()
    for t in TOP_JOURNALS:
        if t in jl:
            return "top"
    for t in EXTRA_JOURNALS:
        if t in jl:
            return "whitelist"
    return ""


def paper_type_of(w: dict) -> str:
    oa_type = (w.get("type") or "").lower()
    if oa_type == "review" or REVIEW_PAT.search(w.get("title", "")):
        return "review"
    return "original_research"


def cite_score(cited: int, year: int) -> float:
    base = math.log10(cited + 1)
    if year >= 2023:
        # recent papers: fewer citations expected; boost novelty
        base += 0.6 if cited >= 20 else 0.2
    return base


def classify_priority(w: dict) -> tuple[str, str]:
    """Return (priority, reason). Stricter A/B split so the pool is layered."""
    cited = w.get("cited_by_count", 0)
    year = w.get("year") or 0
    jrank = journal_match(w.get("journal", ""))
    ptype = paper_type_of(w)
    topic = w.get("topic_cluster")

    if topic == "hard_negative":
        return "C", "Hard negative: topic-similar but answers a different question"

    # Priority A (core must-have): landmark papers, key reviews, recent breakthroughs.
    if cited >= 800:
        return "A", f"Landmark classic ({cited} citations), representative work"
    if cited >= 400 and jrank == "top":
        return "A", f"High-cited ({cited}) in core journal ({w['journal']})"
    if ptype == "review" and cited >= 300:
        return "A", f"Key review ({cited} citations) in {w['journal'] or 'the field'}"
    if year >= 2025 and cited >= 80 and jrank == "top":
        return "A", f"Important recent ({year}) work in {w['journal']} ({cited} citations)"

    # Priority B (important supplement): the bulk of coverage and diversity.
    if cited >= 150:
        return "B", f"Established work ({cited} citations), supplements coverage"
    if ptype == "review" and cited >= 60:
        return "B", f"Review in {w['journal'] or 'the field'}, broadens topic coverage"
    if year >= 2023 and cited >= 20:
        return "B", f"Recent ({year}) work ({cited} citations)"
    if cited >= 60 and jrank in ("top", "whitelist"):
        return "B", f"Moderately cited ({cited}) in {w['journal']}, diversifies conditions"
    if cited >= 30:
        return "B", f"Moderately cited ({cited}) work, diversifies materials/routes"
    return "", ""


def select():
    raw = json.loads((EXP / "candidates_raw.json").read_text(encoding="utf-8"))
    works = raw["works"]

    scored = []
    for w in works:
        year = w.get("year") or 0
        if year and year < MIN_YEAR:
            continue
        pri, reason = classify_priority(w)
        if not pri:
            continue
        jrank = journal_match(w.get("journal", ""))
        score = (
            cite_score(w.get("cited_by_count", 0), year) * 3
            + (3 if jrank == "top" else 1.5 if jrank == "whitelist" else 0)
            + (2 if pri == "A" else 1 if pri == "B" else 0.5)
        )
        w["_score"] = score
        w["_priority"] = pri
        w["_reason"] = reason
        w["_ptype"] = paper_type_of(w)
        scored.append(w)

    by_topic = defaultdict(list)
    for w in scored:
        by_topic[w["topic_cluster"]].append(w)

    selected = []
    for topic, target in TOPIC_TARGETS.items():
        pool = sorted(by_topic.get(topic, []), key=lambda x: -x["_score"])
        review_cap = max(2, round(target * REVIEW_RATIO))      # fixed cap
        recent_target = max(2, round(target * RECENT_RATIO))   # fixed >=2023 target

        author_cnt = Counter()
        journal_cnt = Counter()
        picked: list[dict] = []
        review_cnt = 0
        recent_cnt = 0

        def eligible(w):
            fa = w.get("first_author", "")
            j = w.get("journal", "")
            if fa and author_cnt[fa] >= MAX_PER_FIRST_AUTHOR:
                return False
            if j and journal_cnt[j] >= MAX_PER_JOURNAL:
                return False
            return True

        def take(w):
            nonlocal review_cnt, recent_cnt
            picked.append(w)
            if w["_ptype"] == "review":
                review_cnt += 1
            if (w.get("year") or 0) >= 2023:
                recent_cnt += 1
            author_cnt[w.get("first_author", "")] += 1
            journal_cnt[w.get("journal", "")] += 1

        # Reserve slots so recent (>=2023) and Priority B both get real room.
        max_a = target - recent_target

        # Pass 1: Priority A, up to the reserved A budget (respect review cap).
        for w in pool:
            if len(picked) >= max_a:
                break
            if w["_priority"] != "A" or not eligible(w):
                continue
            if w["_ptype"] == "review" and review_cnt >= review_cap:
                continue
            take(w)

        # Pass 2: guarantee recent (>=2023) coverage up to recent_target.
        for w in pool:
            if recent_cnt >= recent_target or len(picked) >= target:
                break
            if w in picked or not eligible(w):
                continue
            if (w.get("year") or 0) < 2023:
                continue
            if w["_ptype"] == "review" and review_cnt >= review_cap:
                continue
            take(w)

        # Pass 3: Priority B fill to target (respect review cap).
        for w in pool:
            if len(picked) >= target:
                break
            if w in picked or w["_priority"] != "B" or not eligible(w):
                continue
            if w["_ptype"] == "review" and review_cnt >= review_cap:
                continue
            take(w)

        # Pass 4 (hard_negative only): fill with remaining C.
        if topic == "hard_negative":
            for w in pool:
                if len(picked) >= target:
                    break
                if w in picked or not eligible(w):
                    continue
                take(w)

        # Pass 5: relax caps only if still short.
        for w in pool:
            if len(picked) >= target:
                break
            if w in picked or not eligible(w):
                continue
            take(w)

        selected.extend(picked)

    return selected


def access_status_of(w: dict) -> tuple[str, str]:
    if w.get("oa_url"):
        return "open_access", w["oa_url"]
    jrank = journal_match(w.get("journal", ""))
    jl = (w.get("journal") or "").lower()
    big_pub = any(k in jl for k in ["elsevier", "springer", "wiley", "acs", "rsc", "mdpi", "taylor"])
    doi = w.get("doi", "")
    if jrank in ("top", "whitelist") or big_pub:
        return "institution_access_possible", f"https://doi.org/{doi}" if doi else ""
    if doi:
        return "manual_required", f"https://doi.org/{doi}"
    return "unknown", ""


def main():
    selected = select()
    rows = []
    for i, w in enumerate(sorted(selected, key=lambda x: (x["topic_cluster"], -x["_score"])), 1):
        access, pub_url = access_status_of(w)
        rows.append({
            "candidate_id": f"C{i:04d}",
            "priority": w["_priority"],
            "title": w["title"],
            "authors": (w.get("first_author") or "") + " et al.",
            "year": w.get("year", ""),
            "journal": w.get("journal", ""),
            "doi": w.get("doi", ""),
            "topic_cluster": w["topic_cluster"],
            "paper_type": w["_ptype"],
            "citation_signal": w.get("cited_by_count", 0),
            "reason_for_inclusion": w["_reason"],
            "existing_or_new": "new",
            "access_status": access,
            "publisher_url": pub_url or (f"https://doi.org/{w['doi']}" if w.get("doi") else ""),
            "oa_url": w.get("oa_url", ""),
            "notes": f"score={w['_score']:.1f}; queries={'; '.join(w.get('queries', [])[:3])}",
        })

    out = EXP / "paper_candidates.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # stats
    print(f"selected={len(rows)}")
    print("\n--- priority ---")
    for k, v in Counter(r["priority"] for r in rows).most_common():
        print(f"  {k}: {v}")
    print("\n--- topic ---")
    for k, v in sorted(Counter(r["topic_cluster"] for r in rows).items()):
        print(f"  {k}: {v}")
    print("\n--- paper_type ---")
    for k, v in Counter(r["paper_type"] for r in rows).most_common():
        print(f"  {k}: {v}  ({v/len(rows)*100:.0f}%)")
    print("\n--- access ---")
    for k, v in Counter(r["access_status"] for r in rows).most_common():
        print(f"  {k}: {v}")
    print("\n--- year ---")
    years = Counter(int(r["year"]) for r in rows if r["year"])
    for y in sorted(years):
        print(f"  {y}: {years[y]}")
    recent = sum(v for y, v in years.items() if y >= 2023)
    print(f"  >=2023: {recent} ({recent/len(rows)*100:.0f}%)")
    print(f"\nwritten={out}")


if __name__ == "__main__":
    main()
