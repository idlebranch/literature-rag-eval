"""Phase 1 candidate retrieval from OpenAlex (metadata only, no PDF downloads).

Strategy per the expansion prompt:
- 10 topic clusters, "core depth" (AOP / adsorption / emerging contaminants)
  get denser queries; breadth topics get representative queries.
- Each sub-query pulls top-cited classics + recent (2023+) works.
- Hard negatives: topic-similar papers that answer a different question.

Dedup against the existing corpus DOIs (from data/_inventory_existing.json).
Output: data/corpus_expansion/candidates_raw.json (intermediate).
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "corpus_expansion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

API = "https://api.openalex.org/works"
MAILTO = "corpus-survey@local.invalid"  # polite pool identifier

# topic_cluster -> list of sub-queries (OpenAlex title_abstract_search)
QUERIES = {
    "advanced_oxidation_processes": [
        "persulfate activation water treatment",
        "peroxymonosulfate activation contaminants degradation",
        "catalytic ozonation water wastewater",
        "UV/H2O2 water treatment",
        "Fenton wastewater treatment",
        "electro-Fenton degradation",
        "photocatalysis water treatment",
        "advanced oxidation transformation products toxicity",
    ],
    "adsorption_ion_exchange": [
        "activated carbon adsorption water",
        "biochar adsorption pollutants",
        "adsorption mechanism aqueous",
        "activated carbon regeneration",
        "ion exchange water treatment",
        "adsorption isotherms kinetics water",
    ],
    "emerging_contaminants": [
        "PPCPs removal water treatment",
        "antibiotics removal wastewater",
        "endocrine disrupting compounds water treatment",
        "pesticides removal water treatment",
        "PFAS removal water treatment",
        "microplastics removal wastewater treatment",
        "antibiotic resistance genes wastewater treatment",
    ],
    "membrane_treatment": [
        "nanofiltration water treatment",
        "reverse osmosis desalination",
        "membrane bioreactor wastewater",
        "membrane fouling control",
        "ceramic membrane water treatment",
        "ultrafiltration drinking water",
        "membrane cleaning regeneration",
    ],
    "biological_treatment": [
        "activated sludge wastewater treatment",
        "nitrification denitrification wastewater",
        "anaerobic digestion wastewater",
        "anammox nitrogen removal",
        "biofilm wastewater treatment",
        "aerobic granular sludge",
    ],
    "coagulation_flocculation": [
        "coagulation flocculation water treatment",
        "coagulant natural organic matter removal",
        "flocculation mechanism water",
    ],
    "disinfection": [
        "chlorination disinfection drinking water",
        "UV disinfection water",
        "disinfection byproducts formation control",
        "chloramine drinking water",
    ],
    "electrochemical_treatment": [
        "electrocoagulation wastewater",
        "electrochemical oxidation wastewater",
        "capacitive deionization",
        "electrocatalysis water pollutants",
    ],
    "industrial_wastewater": [
        "textile dye wastewater treatment",
        "pharmaceutical wastewater treatment",
        "petrochemical wastewater treatment",
        "landfill leachate treatment",
        "high salinity wastewater treatment",
        "heavy metal wastewater treatment",
    ],
    "sludge_resource_recovery": [
        "sludge dewatering",
        "phosphorus recovery wastewater",
        "sewage sludge anaerobic digestion",
        "sludge biochar",
    ],
    "hard_negative": [
        "PFAS contamination groundwater occurrence",
        "microplastics occurrence rivers",
        "antibiotic resistance environment surveillance",
        "ozonation ecotoxicity bioassay",
        "adsorption soil remediation",
    ],
}

# recent-only queries for freshness (2023-2026), core + breadth highlights
RECENT_QUERIES = [
    ("advanced_oxidation_processes", "persulfate advanced oxidation 2024 water"),
    ("adsorption_ion_exchange", "adsorption emerging contaminants 2024"),
    ("emerging_contaminants", "PFAS destruction treatment 2024"),
    ("membrane_treatment", "membrane fouling mitigation 2024"),
    ("biological_treatment", "anammox mainstream 2024"),
    ("disinfection", "disinfection byproducts 2024"),
    ("industrial_wastewater", "industrial wastewater treatment 2024"),
]

SELECT = (
    "id,doi,title,publication_year,cited_by_count,type,open_access,"
    "primary_location,authorships"
)


def fetch_page(search: str, kind: str, year_from: int | None, per_page: int = 50):
    params = {
        "search": search,
        "select": SELECT,
        "per-page": str(per_page),
        "mailto": MAILTO,
    }
    filters = ["type:article"]
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    params["filter"] = ",".join(filters)
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "phase1-corpus-survey/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def first_author(work: dict) -> str:
    try:
        auth = work["authorships"][0]["author"]["display_name"]
        return auth
    except Exception:
        return ""


def journal_of(work: dict) -> str:
    try:
        src = (work.get("primary_location") or {}).get("source") or {}
        return src.get("display_name") or ""
    except Exception:
        return ""


def collect():
    existing = json.loads((ROOT / "data" / "_inventory_existing.json").read_text(encoding="utf-8"))
    existing_dois = {
        (r["doi"] or "").lower().replace("https://doi.org/", "")
        for r in existing if r.get("doi")
    }
    existing_titles = {norm_title(r.get("title_meta", "")) for r in existing}

    seen_doi: dict[str, dict] = {}
    seen_title: set[str] = set()
    stats = {}

    def add_work(w, topic, query, kind):
        doi = (w.get("doi") or "").lower().replace("https://doi.org/", "")
        title = w.get("title") or ""
        nt = norm_title(title)
        if not doi and not nt:
            return False
        # dedup vs existing corpus
        if doi and doi in existing_dois:
            return False
        if nt and nt in existing_titles:
            return False
        # dedup within candidates
        if doi:
            if doi in seen_doi:
                seen_doi[doi]["queries"].append(f"{kind}:{topic}:{query}")
                return False
            key = doi
        else:
            if nt in seen_title:
                return False
            seen_title.add(nt)
            key = f"no-doi:{nt}"
        rec = {
            "doi": doi,
            "title": title,
            "year": w.get("publication_year"),
            "cited_by_count": w.get("cited_by_count", 0),
            "journal": journal_of(w),
            "first_author": first_author(w),
            "oa_status": (w.get("open_access") or {}).get("oa_status", ""),
            "oa_url": (w.get("open_access") or {}).get("oa_url") or "",
            "topic_cluster": topic,
            "queries": [f"{kind}:{topic}:{query}"],
            "openalex_id": w.get("id", ""),
        }
        seen_doi[key] = rec
        return True

    total_hits = 0
    # Main queries: top-cited classics (fetch by relevance, sort locally later)
    for topic, qs in QUERIES.items():
        for q in qs:
            try:
                d = fetch_page(q, "classics", None)
            except Exception as e:
                print(f"ERR classics {topic}|{q}: {e}")
                continue
            n = 0
            for w in d.get("results", []):
                if add_work(w, topic, q, "classics"):
                    n += 1
            total_hits += n
            stats[f"classics|{topic}|{q}"] = n
            time.sleep(0.15)
    # Recent queries (2023+)
    for topic, q in RECENT_QUERIES:
        try:
            d = fetch_page(q, "recent", 2023)
        except Exception as e:
            print(f"ERR recent {topic}|{q}: {e}")
            continue
        n = 0
        for w in d.get("results", []):
            if add_work(w, topic, q, "recent"):
                n += 1
        total_hits += n
        stats[f"recent|{topic}|{q}"] = n
        time.sleep(0.15)

    out = OUT_DIR / "candidates_raw.json"
    out.write_text(
        json.dumps({"count": len(seen_doi), "works": list(seen_doi.values())},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"candidates={len(seen_doi)} added_hits={total_hits}")
    from collections import Counter
    tc = Counter(r["topic_cluster"] for r in seen_doi.values())
    for k, v in tc.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    collect()
