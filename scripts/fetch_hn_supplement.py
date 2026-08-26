"""Supplement hard-negative candidates: occurrence/fate/measurement studies in
water that are highly similar to water-treatment retrieval queries but do NOT
answer the treatment question. Metadata only, no downloads.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "corpus_expansion"

API = "https://api.openalex.org/works"
SELECT = ("id,doi,title,publication_year,cited_by_count,type,open_access,"
          "primary_location,authorships")

QUERIES = [
    "occurrence of microplastics in wastewater",
    "occurrence of antibiotics in drinking water",
    "PFAS occurrence in groundwater",
    "occurrence disinfection byproducts drinking water",
    "antibiotic resistance genes in rivers",
    "microplastics in freshwater occurrence review",
    "occurrence of pharmaceuticals in surface water",
    "PFAS occurrence wastewater treatment plant effluent",
    "microplastics transport fate river",
    "prevalence PFAS drinking water sources",
]

HN_KEEP = re.compile(
    r"\b(occurrence|distribution|fate|burden|surveillance|detection|monitoring|"
    r"abundance|exposure|ecotoxicity|presence|levels?|prevalence|assessment|"
    r"identification|characteri[sz]ation|aggregation|deposition|accumulation|"
    r"sources?|tracing|quantification|concentration|behavio[u]?r|transport|"
    r"spatial)\b", re.I)
HN_WATER = re.compile(
    r"\b(wastewater|water|aquatic|freshwater|marine|rivers?|lakes?|drinking|"
    r"effluent|sewage|treatment plant|groundwater|surface water)\b", re.I)
HN_OFFTOPIC = re.compile(
    r"\b(soil|terrestrial|sediment-only|epidemiolog|clinical|patients?|food chain|"
    r"crops?|agricultur|air pollut|atmosphere|indoor dust|human health)\b", re.I)
# pure treatment papers (no occurrence keyword) are NOT valid hard negatives
HN_TREATMENT = re.compile(
    r"\b(removal of|degradation of|decomposition of|elimination of|"
    r"treatment of|for water treatment|adsorption of .+ from (aqueous|water)|"
    r"photocatalytic degradation|advanced oxidation of)\b", re.I)


def fetch(search: str, per_page=25):
    params = {"search": search, "select": SELECT, "per-page": str(per_page),
              "filter": "type:article", "mailto": "corpus-survey@local.invalid"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "corpus-survey/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def verdict(title: str):
    if HN_OFFTOPIC.search(title):
        return False, "off_topic"
    if not HN_WATER.search(title):
        return False, "not_water"
    if not HN_KEEP.search(title):
        return False, "not_occurrence_type"
    return True, ""


def first_author(w):
    try:
        return w["authorships"][0]["author"]["display_name"]
    except Exception:
        return ""


def journal_of(w):
    try:
        return ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    except Exception:
        return ""


def main():
    # dedup against everything already in the pipeline
    seen_dois = set()
    import csv as _csv
    for name in ["paper_candidates.csv"]:
        with (EXP / name).open(encoding="utf-8-sig") as f:
            for r in _csv.DictReader(f):
                seen_dois.add(r["doi"].lower())
    inv = json.loads((ROOT / "data" / "_inventory_existing.json").read_text(encoding="utf-8"))
    seen_dois |= {(r.get("doi") or "").lower() for r in inv}

    out, kept = [], 0
    for q in QUERIES:
        try:
            d = fetch(q)
        except Exception as e:
            print(f"ERR {q}: {e}")
            continue
        for w in d.get("results", []):
            doi = (w.get("doi") or "").lower().replace("https://doi.org/", "")
            title = w.get("title") or ""
            if not doi or doi in seen_dois:
                continue
            ok, reason = verdict(title)
            if not ok:
                continue
            seen_dois.add(doi)
            rec = {
                "doi": doi,
                "title": title,
                "year": w.get("publication_year"),
                "cited_by_count": w.get("cited_by_count", 0),
                "journal": journal_of(w),
                "first_author": first_author(w),
                "oa_status": (w.get("open_access") or {}).get("oa_status", ""),
                "oa_url": (w.get("open_access") or {}).get("oa_url") or "",
                "topic_cluster": "hard_negative",
                "queries": [f"hn_supplement:{q}"],
                "openalex_id": w.get("id", ""),
            }
            out.append(rec)
            kept += 1
            if kept >= 60:
                break
        time.sleep(0.15)
        if kept >= 60:
            break

    (EXP / "hn_supplement.json").write_text(
        json.dumps({"count": len(out), "works": out}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"hn_supplement kept={len(out)}")
    for r in out[:40]:
        print(f"  [{r['cited_by_count']:>5}] ({r['year']}) {r['title'][:80]}")


if __name__ == "__main__":
    main()
