"""Phase 1.5: trim the 412-candidate pool into main / reserve / rejected.

Final corpus target ~280-320. Existing 66 all kept. New-paper quota per topic
= final_target - existing_count. Main list ~240, reserve ~40, rest rejected.

Priority A re-tightened: within each non-HN topic, top ~70% by score labelled A,
rest B; hard negatives labelled C. A total lands ~140-180.

Recent (2020-2026) reinforcement, heavier for membrane / biological /
electrochemical / emerging_contaminants.

Hard negatives must stay water-treatment-adjacent but NOT answer the treatment
question; soil remediation / pure epidemiology removed.

Suspected duplicates re-checked against existing corpus by DOI + Crossref title.

No downloads, no RAG-code changes.
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "corpus_expansion"

TOP_JOURNALS = [
    "nature water", "environmental science & technology", "water research",
    "journal of hazardous materials", "chemical engineering journal",
    "applied catalysis b", "journal of membrane science",
    "separation and purification technology", "environment international",
    "environmental pollution", "chemosphere", "science of the total environment",
    "bioresource technology", "desalination",
]

# existing corpus topic counts (phase1 report)
EXISTING = {
    "advanced_oxidation_processes": 35, "adsorption_ion_exchange": 22,
    "emerging_contaminants": 5, "membrane_treatment": 0, "biological_treatment": 0,
    "industrial_wastewater": 1, "disinfection": 0, "electrochemical_treatment": 0,
    "coagulation_flocculation": 0, "sludge_resource_recovery": 0, "hard_negative": 0,
}
# final corpus count per topic (midpoint of user's ranges)
FINAL_TARGET = {
    "advanced_oxidation_processes": 55, "adsorption_ion_exchange": 46,
    "emerging_contaminants": 35, "membrane_treatment": 30, "biological_treatment": 30,
    "industrial_wastewater": 22, "disinfection": 17, "electrochemical_treatment": 17,
    "coagulation_flocculation": 17, "sludge_resource_recovery": 12, "hard_negative": 22,
}
QUOTA = {t: FINAL_TARGET[t] - EXISTING.get(t, 0) for t in FINAL_TARGET}

MAX_PER_AUTHOR = 3
MAX_PER_JOURNAL = 8
REVIEW_CAP_RATIO = 0.25
A_RATIO = 0.70
RECENT_RATIO_DEFAULT = 0.30
RECENT_RATIO_BOOST = 0.45   # membrane / bio / electrochem / emerging
BOOST_TOPICS = {"membrane_treatment", "biological_treatment",
                "electrochemical_treatment", "emerging_contaminants"}
RESERVE_SIZE = 40
RESERVE_MAX_PER_TOPIC = 6


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def tokens(s: str) -> set:
    return set(norm(s).split())


def jaccard(a: str, b: str) -> float:
    sa, sb = tokens(a), tokens(b)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def jrank(j: str) -> str:
    jl = (j or "").lower()
    return "top" if any(t in jl for t in TOP_JOURNALS) else ""


def score_of(w) -> float:
    try:
        return float(w["notes"].split(";")[0].split("=")[1])
    except Exception:
        return 0.0


def load_rows():
    with (EXP / "paper_candidates.csv").open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_existing():
    inv = json.loads((ROOT / "data" / "_inventory_existing.json").read_text(encoding="utf-8"))
    dois = {(r.get("doi") or "").lower() for r in inv if r.get("doi")}
    # use filenames (contain real titles) + meta titles + preview
    texts = []
    for r in inv:
        texts.append(r.get("filename", "").rsplit(".", 1)[0])
        texts.append(r.get("title_meta", ""))
        texts.append(r.get("first_page_preview", "")[:160])
    return inv, dois, [t for t in texts if t and len(t) > 12]


def crossref_meta(doi: str):
    url = f"https://api.crossref.org/works/{urllib.request.quote(doi)}"
    req = urllib.request.Request(url, headers={"User-Agent": "corpus-trim/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            m = json.load(r).get("message", {})
            title = (m.get("title") or [""])[0]
            venue = (m.get("container-title") or [""])[0]
            year = (m.get("published", {}).get("date-parts", [[None]])[0][0])
            return title, venue, year
    except Exception as e:
        return None, None, None


HN_KEEP_SIG = re.compile(
    r"\b(occurrence|distribution|fate|burden|surveillance|detection|monitoring|"
    r"abundance|exposure|ecotoxicity|toxicity|presence|levels?|prevalence|"
    r"assessment|accumulation|transport|behavio[u]?r|sources?|tracing)\b", re.I)
HN_WATER_SIG = re.compile(
    r"\b(wastewater|water|aquatic|freshwater|marine|rivers?|lakes?|drinking|"
    r"effluent|sewage|treatment plant|groundwater|surface water|sediments?)\b", re.I)
HN_OFFTOPIC_SIG = re.compile(
    r"\b(soil|terrestrial|epidemiolog|clinical|patients?|food chain|"
    r"crops?|agricultur|air pollut|atmosphere|indoor dust)\b", re.I)
HN_ANSWERS_TREATMENT = re.compile(
    r"\b(removal|degradation|decomposition|elimination|treatment of|"
    r"for water treatment|adsorption of .+ from (aqueous|water)|"
    r"photocatalytic degradation|advanced oxidation of)\b", re.I)


def hn_verdict(w):
    """Return (keep: bool, reason: str).

    Valid hard negatives: occurrence/fate/measurement of pollutants IN water
    bodies (vocabulary overlaps water-treatment queries, but the paper answers
    "what is where / how much" rather than "how to treat"). Pure treatment
    papers without occurrence evidence, and soil/epidemiology papers, are
    rejected.
    """
    t = w["title"]
    if HN_OFFTOPIC_SIG.search(t):
        return False, "hn_off_topic(soil/epidemiology/non-water)"
    if not HN_WATER_SIG.search(t):
        return False, "hn_off_topic(not water-related)"
    if HN_KEEP_SIG.search(t):
        return True, ""
    if HN_ANSWERS_TREATMENT.search(t):
        return False, "hn_answers_same_question(is a treatment paper)"
    return False, "hn_not_occurrence_type"


def main():
    rows = load_rows()
    inv, existing_dois, existing_texts = load_existing()
    print(f"candidates={len(rows)} existing={len(inv)} existing_dois={len(existing_dois)}")

    # ---- 1. duplicate re-check vs existing corpus ----
    dup_dois = set()
    print("\n=== duplicate re-check (DOI exact + title jaccard + Crossref) ===")
    for w in rows:
        doi = w["doi"].lower()
        flagged = doi in existing_dois
        best_sim, best_text = 0.0, ""
        for t in existing_texts:
            s = jaccard(w["title"], t)
            if s > best_sim:
                best_sim, best_text = s, t
        if best_sim >= 0.8:
            flagged = True
        if not flagged:
            continue
        ct, cv, cy = crossref_meta(w["doi"])
        # confirm only if Crossref title strongly matches an existing text
        confirmed = False
        if ct:
            for t in existing_texts:
                if jaccard(ct, t) >= 0.9:
                    confirmed = True
                    break
        if doi in existing_dois:
            confirmed = True
        status = "DUPLICATE->reject" if confirmed else "similar-but-distinct->keep"
        if confirmed:
            dup_dois.add(w["doi"])
        print(f"\n[{status}] sim={best_sim:.2f} doi={w['doi']}")
        print(f"  cand: {w['title'][:85]} ({w['year']}, {w['journal'][:30]})")
        print(f"  existing_best: {best_text[:85]}")
        if ct:
            print(f"  crossref: {ct[:85]} | {cv} | {cy}")

    # ---- 2. hard negative filtering (original pool + supplement) ----
    hn_rows = [w for w in rows if w["priority"] == "C" and w["doi"] not in dup_dois]
    # merge in the occurrence-focused supplement pool
    supp_path = EXP / "hn_supplement.json"
    if supp_path.exists():
        supp = json.loads(supp_path.read_text(encoding="utf-8"))["works"]
        seen_hn = {w["doi"] for w in hn_rows}
        for s in supp:
            if s["doi"] in seen_hn:
                continue
            # reshape to candidate CSV row format
            oa = "open_access" if s.get("oa_url") else (
                "institution_access_possible" if jrank(s["journal"]) else "manual_required")
            hn_rows.append({
                "candidate_id": f"S{len(hn_rows)+1:03d}",
                "priority": "C",
                "title": s["title"],
                "authors": f"{s['first_author']} et al.",
                "year": str(s["year"] or ""),
                "journal": s["journal"],
                "doi": s["doi"],
                "topic_cluster": "hard_negative",
                "paper_type": "original_research",
                "citation_signal": str(s["cited_by_count"]),
                "reason_for_inclusion": "Hard negative supplement: occurrence-type study in water",
                "existing_or_new": "new",
                "access_status": oa,
                "publisher_url": f"https://doi.org/{s['doi']}",
                "oa_url": s.get("oa_url", ""),
                "notes": f"score=0.0; hn_supplement",
            })
            seen_hn.add(s["doi"])
        print(f"\nhard negatives after supplement merge: {len(hn_rows)}")
    hn_keep, hn_drop = [], []
    for w in hn_rows:
        ok, reason = hn_verdict(w)
        (hn_keep if ok else hn_drop).append((w, reason))
    print(f"\n=== hard negatives: keep={len(hn_keep)} drop={len(hn_drop)} ===")
    for w, reason in hn_drop:
        print(f"  DROP [{reason}] {w['title'][:80]}")

    # ---- 3. per-topic quota selection (non-HN) ----
    by_topic = defaultdict(list)
    for w in rows:
        if w["doi"] in dup_dois or w["priority"] == "C":
            continue
        by_topic[w["topic_cluster"]].append(w)

    main_list = []
    rejected = []
    for w, _ in hn_drop:
        rejected.append((w, "hn_off_topic"))
    for w in rows:
        if w["doi"] in dup_dois:
            rejected.append((w, "duplicate_of_existing_corpus"))

    for topic, quota in QUOTA.items():
        if topic == "hard_negative":
            continue
        pool = sorted(by_topic.get(topic, []), key=lambda x: -score_of(x))
        author_cnt, journal_cnt = Counter(), Counter()
        review_cnt = 0
        review_cap = max(2, round(quota * REVIEW_CAP_RATIO))
        recent_ratio = RECENT_RATIO_BOOST if topic in BOOST_TOPICS else RECENT_RATIO_DEFAULT
        recent_target = max(2, round(quota * recent_ratio))
        picked = []

        def eligible(w):
            fa = w["authors"].replace(" et al.", "").strip()
            if fa and author_cnt[fa] >= MAX_PER_AUTHOR:
                return False
            if journal_cnt[w["journal"]] >= MAX_PER_JOURNAL:
                return False
            if w["paper_type"] == "review" and review_cnt >= review_cap:
                return False
            return True

        def take(w):
            nonlocal review_cnt
            picked.append(dict(w))
            if w["paper_type"] == "review":
                review_cnt += 1
            fa = w["authors"].replace(" et al.", "").strip()
            author_cnt[fa] += 1
            journal_cnt[w["journal"]] += 1

        picked_dois = set()
        # recent reinforcement first (so 2020-2026 gets guaranteed slots)
        recent_added = 0
        for w in pool:
            if recent_added >= recent_target or len(picked) >= quota:
                break
            if int(w["year"] or 0) >= 2020 and eligible(w):
                take(w); picked_dois.add(w["doi"]); recent_added += 1
        # fill remaining by score
        for w in pool:
            if len(picked) >= quota:
                break
            if w["doi"] in picked_dois or not eligible(w):
                continue
            take(w); picked_dois.add(w["doi"])

        # assign A/B by score within topic (top A_RATIO -> A)
        picked_sorted = sorted(picked, key=lambda x: -score_of(x))
        a_budget = round(len(picked_sorted) * A_RATIO)
        for i, w in enumerate(picked_sorted):
            w["priority_final"] = "A" if i < a_budget else "B"
        main_list.extend(picked_sorted)
        nA = sum(1 for w in picked_sorted if w["priority_final"] == "A")
        nRecent = sum(1 for w in picked_sorted if int(w["year"] or 0) >= 2020)
        nRev = sum(1 for w in picked_sorted if w["paper_type"] == "review")
        print(f"\n[{topic}] quota={quota} picked={len(picked_sorted)} A={nA} "
              f"recent>=2020={nRecent} reviews={nRev}")

    # hard negatives into main list (kept ones, up to quota)
    hn_sorted = sorted(hn_keep, key=lambda x: -int(x[0]["citation_signal"]))
    for w, _ in hn_sorted[:QUOTA["hard_negative"]]:
        w2 = dict(w); w2["priority_final"] = "C"
        main_list.append(w2)
    for w, _ in hn_sorted[QUOTA["hard_negative"]:]:
        rejected.append((w, "hn_quota_exceeded"))
    print(f"\n[hard_negative] kept={min(len(hn_sorted), QUOTA['hard_negative'])}")

    # ---- 4. reserve from leftovers ----
    main_dois = {w["doi"] for w in main_list}
    rej_dois = {w["doi"] for w, _ in rejected}
    leftovers = [w for w in rows
                 if w["doi"] not in main_dois and w["doi"] not in rej_dois]
    leftovers.sort(key=lambda x: -score_of(x))
    reserve, res_topic = [], Counter()
    for w in leftovers:
        if len(reserve) >= RESERVE_SIZE:
            break
        if res_topic[w["topic_cluster"]] >= RESERVE_MAX_PER_TOPIC:
            continue
        reserve.append(w); res_topic[w["topic_cluster"]] += 1
    reserve_dois = {w["doi"] for w in reserve}
    for w in leftovers:
        if w["doi"] not in reserve_dois:
            rejected.append((w, "quota_exceeded"))

    # ---- 5. write CSVs ----
    base_fields = ["candidate_id", "priority", "title", "authors", "year", "journal",
                   "doi", "topic_cluster", "paper_type", "citation_signal",
                   "reason_for_inclusion", "existing_or_new", "access_status",
                   "publisher_url", "oa_url", "notes"]

    def emit(path, items, extra=None):
        fields = base_fields + (extra or [])
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            for w in items:
                row = {k: w.get(k, "") for k in base_fields}
                if extra:
                    for k in extra:
                        row[k] = w.get(k, "")
                wr.writerow(row)

    emit(EXP / "papers_to_download.csv", main_list, ["priority_final"])
    emit(EXP / "reserve_papers.csv", reserve)
    emit(EXP / "rejected_candidates.csv",
         [dict(w, reject_reason=r) for w, r in rejected], ["reject_reason"])

    # ---- 6. distributions ----
    print("\n================ FINAL DISTRIBUTIONS ================")
    n_main = len(main_list)
    print(f"main={n_main}  reserve={len(reserve)}  rejected={len(rejected)}")
    print(f"final corpus = existing 66 + main {n_main} = {66 + n_main}")
    nA = sum(1 for w in main_list if w["priority_final"] == "A")
    nB = sum(1 for w in main_list if w["priority_final"] == "B")
    nC = sum(1 for w in main_list if w["priority_final"] == "C")
    print(f"priority: A={nA} B={nB} C={nC}")

    print("\n--- topic: existing | main | reserve | final ---")
    mt = Counter(w["topic_cluster"] for w in main_list)
    rt = Counter(w["topic_cluster"] for w in reserve)
    for t in QUOTA:
        print(f"  {t:30s} {EXISTING.get(t,0):3d} | {mt.get(t,0):3d} | "
              f"{rt.get(t,0):2d} | {EXISTING.get(t,0)+mt.get(t,0):3d}")

    print("\n--- year buckets (main) ---")
    b = Counter()
    for w in main_list:
        y = int(w["year"] or 0)
        k = ("<2010" if y < 2010 else "2010-2014" if y < 2015 else "2015-2019"
             if y < 2020 else "2020-2022" if y < 2023 else "2023-2026")
        b[k] += 1
    for k in ["<2010", "2010-2014", "2015-2019", "2020-2022", "2023-2026"]:
        print(f"  {k}: {b[k]}")
    rec = b["2020-2022"] + b["2023-2026"]
    print(f"  >=2020 share: {rec}/{n_main} = {rec/n_main*100:.0f}%")

    print("\n--- paper_type (main) ---")
    pt = Counter(w["paper_type"] for w in main_list)
    for k, v in pt.most_common():
        print(f"  {k}: {v} ({v/n_main*100:.0f}%)")

    print("\n--- access (main) ---")
    for k, v in Counter(w["access_status"] for w in main_list).most_common():
        print(f"  {k}: {v}")

    print("\n--- reject reasons ---")
    for k, v in Counter(r for _, r in rejected).most_common():
        print(f"  {k}: {v}")

    # save a machine-readable summary for phase 2
    summary = {
        "main": n_main, "reserve": len(reserve), "rejected": len(rejected),
        "final_corpus": 66 + n_main, "A": nA, "B": nB, "C": nC,
        "topic_main": dict(mt), "topic_reserve": dict(rt),
        "access_main": dict(Counter(w["access_status"] for w in main_list)),
    }
    (EXP / "phase15_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: papers_to_download.csv / reserve_papers.csv / "
          f"rejected_candidates.csv / phase15_summary.json")


if __name__ == "__main__":
    main()
