"""Final corpus cleanup: dedup + organize the ~360 scanned PDFs.

Reuses data/papers/_scan_cache.json (no PDF re-parsing) and applies ONLY the
minimal dedup rules:

  1. SHA256 identical                    -> duplicate
  2. clear DOI, normalized DOI identical -> duplicate
  3. remaining no-DOI files: normalized title + text fingerprint,
     high-confidence match only          -> duplicate
  4. everything else                     -> unresolved (kept, no more heuristics)

Rejected = unusable files (unreadable, no text, error/login page, corrigendum,
TOC, editorial) plus the files already quarantined in data/papers/rejected/.

Non-destructive: originals are NEVER deleted or moved.  Keepers are COPIED into
final_corpus/, duplicates COPIED into duplicates_quarantine/.  Prior-rejected
files already live in rejected/ and stay there.

Usage:
    python scripts/build_final_corpus.py --plan     # print only, write nothing
    python scripts/build_final_corpus.py --apply    # build folders + write reports
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\10475\AI_PROJECT\literature_rag_eval_code")
DATA = ROOT / "data" / "papers"
CACHE = DATA / "_scan_cache.json"

FINAL = DATA / "final_corpus"
DUP_DIR = DATA / "duplicates_quarantine"
REJ_DIR = DATA / "rejected"

MANIFEST = DATA / "final_paper_manifest.csv"
CLEANUP = DATA / "corpus_cleanup_report.md"
DUP_CSV = DATA / "duplicates_report.csv"
REJ_CSV = DATA / "rejected_report.csv"
UNRES_CSV = DATA / "unresolved_report.csv"

# doc_types that are clearly NOT a usable research paper.
REJECT_DOC_TYPES = {
    "error_or_login_page",
    "no_extractable_text",
    "corrigendum_or_erratum",
    "table_of_contents",
    "editorial",
}
# "supplement" / "supplement_suspect" are deliberately NOT rejected:
# the classifier is a known false-positive source (a 10-page PNAS main article
# fires "supplement_suspect"), so those are treated as normal articles.

TITLE_TEXT_JACCARD = 0.85


# ---------------------------------------------------------------- helpers

def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def titles_match(a: str, b: str) -> bool:
    na, nb = norm_title(a), norm_title(b)
    if len(na) < 15 or len(nb) < 15:
        return False
    if na == nb:
        return True
    if len(na) > len(nb):
        na, nb = nb, na
    return na in nb and len(na) >= 0.6 * len(nb)


def jaccard(a, b) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    u = sa | sb
    return len(sa & sb) / len(u) if u else 0.0


def text_high_conf(r1: dict, r2: dict) -> bool:
    # identical normalized body fingerprint is the strongest signal
    if r1["fingerprint"] and r1["fingerprint"] == r2["fingerprint"]:
        return True
    return jaccard(r1["shingles"], r2["shingles"]) >= TITLE_TEXT_JACCARD


def high_conf(r1: dict, r2: dict) -> bool:
    """High-confidence duplicate: title AND text both agree."""
    return titles_match(r1["title"], r2["title"]) and text_high_conf(r1, r2)


def quality(r: dict):
    return (bool(r["doi"]), int(r["has_references"]), r["text_length"], r["file_name"].lower())


def is_rejected(r: dict) -> bool:
    if r["source_group"] == "papers_rejected_prior":
        return True
    if r["doc_type"] in REJECT_DOC_TYPES:
        return True
    if r["open_error"] or not r["sha256"] or r["images_only"] or r["text_length"] < 200:
        return True
    return False


def sanitize(name: str) -> str:
    n = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name or "").strip(" .")
    return n or "unnamed.pdf"


def unique_name(target_dir: Path, name: str, used: set[str]) -> str:
    base = sanitize(name)
    stem, ext = Path(base).stem, Path(base).suffix or ".pdf"
    cand = f"{stem}{ext}"
    i = 2
    while cand.lower() in used:
        cand = f"{stem}_{i}{ext}"
        i += 1
    used.add(cand.lower())
    return cand


# ---------------------------------------------------------------- main

def load() -> list[dict]:
    return json.loads(CACHE.read_text(encoding="utf-8"))["records"]


def run(records: list[dict], apply: bool) -> dict:
    # ---- reject first (excluded from dedup) ----
    rejected = [r for r in records if is_rejected(r)]
    pool = [r for r in records if not is_rejected(r)]

    # ---- 1. sha256 ----
    by_sha = defaultdict(list)
    for r in pool:
        by_sha[r["sha256"]].append(r)
    sha_survivors = [max(v, key=quality) for v in by_sha.values()]
    sha_dup = len(pool) - len(sha_survivors)

    # ---- 2. DOI ----
    with_doi = [r for r in sha_survivors if r["doi"]]
    no_doi = [r for r in sha_survivors if not r["doi"]]
    by_doi = defaultdict(list)
    for r in with_doi:
        by_doi[r["doi"]].append(r)
    doi_reps = [max(v, key=quality) for v in by_doi.values()]
    doi_dup = len(with_doi) - len(doi_reps)

    # ---- 3. title/text (union-find over doi reps + no-doi files) ----
    nodes = doi_reps + no_doi
    n = len(nodes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = nodes[i], nodes[j]
            # only merge when at least one side has no DOI
            if ri["doi"] and rj["doi"]:
                continue
            if high_conf(ri, rj):
                union(i, j)

    comps = defaultdict(list)
    for i, r in enumerate(nodes):
        comps[find(i)].append(r)

    keepers: list[dict] = []
    titletext_dup: list[dict] = []
    unresolved: list[dict] = []
    for members in comps.values():
        if len(members) == 1:
            keepers.append(members[0])
            if not members[0]["doi"]:
                unresolved.append(members[0])
        else:
            best = max(members, key=quality)
            keepers.append(best)
            for m in members:
                if m is not best:
                    titletext_dup.append(m)

    # resolution tag for manifest
    def resolution(r: dict) -> str:
        if r["doi"]:
            return "unique_doi"
        if any(r is m for m in unresolved):
            return "unresolved"
        return "unique_title_text"

    # duplicates in processing order for the report
    sha_dups = [r for r in pool if r not in sha_survivors]
    doi_dups = [r for r in with_doi if r not in doi_reps]

    total_dup = len(sha_dups) + len(doi_dups) + len(titletext_dup)
    final_count = len(keepers)

    result = {
        "total": len(records),
        "rejected": rejected,
        "pool": len(pool),
        "sha_dup": sha_dups,
        "doi_dup": doi_dups,
        "titletext_dup": titletext_dup,
        "unresolved": unresolved,
        "keepers": keepers,
        "sha_dup_n": len(sha_dups),
        "doi_dup_n": len(doi_dups),
        "titletext_dup_n": len(titletext_dup),
        "unresolved_n": len(unresolved),
        "rejected_n": len(rejected),
        "final_n": final_count,
        "total_dup": total_dup,
        "resolution": resolution,
    }

    if apply:
        build_folders(result)
        write_reports(result)
    return result


def build_folders(r: dict) -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    DUP_DIR.mkdir(parents=True, exist_ok=True)
    REJ_DIR.mkdir(parents=True, exist_ok=True)

    used_final: set[str] = set()
    used_dup: set[str] = set()
    used_rej: set[str] = set()

    for k in r["keepers"]:
        name = unique_name(FINAL, k["file_name"], used_final)
        shutil.copy2(k["path"], FINAL / name)
        k["_final_name"] = name

    for kind, items in (("sha256", r["sha_dup"]), ("doi", r["doi_dup"]),
                        ("title_text", r["titletext_dup"])):
        for d in items:
            name = unique_name(DUP_DIR, d["file_name"], used_dup)
            shutil.copy2(d["path"], DUP_DIR / name)
            d["_dup_name"] = name
            d["_dup_kind"] = kind

    # only copy NEW anomalies; prior-rejected files already live in rejected/
    for x in r["rejected"]:
        if x["source_group"] != "papers_rejected_prior":
            name = unique_name(REJ_DIR, x["file_name"], used_rej)
            shutil.copy2(x["path"], REJ_DIR / name)
            x["_rej_name"] = name


def _w(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_reports(r: dict) -> None:
    # manifest
    _w(MANIFEST, [{
        "final_file": k.get("_final_name", ""),
        "resolution": r["resolution"](k),
        "sha256": k["sha256"],
        "doi": k["doi"],
        "title": k["title"],
        "year": k["year"],
        "first_author": k["first_author"],
        "page_count": k["page_count"],
        "text_length": k["text_length"],
        "source_path": k["path"],
        "source_group": k["source_group"],
    } for k in sorted(r["keepers"], key=lambda x: x.get("_final_name", ""))],
        ["final_file", "resolution", "sha256", "doi", "title", "year",
         "first_author", "page_count", "text_length", "source_path", "source_group"])

    # duplicates
    _w(DUP_CSV, [{
        "dup_kind": d.get("_dup_kind", ""),
        "sha256": d["sha256"],
        "doi": d["doi"],
        "title": d["title"],
        "page_count": d["page_count"],
        "text_length": d["text_length"],
        "source_path": d["path"],
        "source_group": d["source_group"],
        "quarantine_file": d.get("_dup_name", ""),
    } for d in r["sha_dup"] + r["doi_dup"] + r["titletext_dup"]],
        ["dup_kind", "sha256", "doi", "title", "page_count", "text_length",
         "source_path", "source_group", "quarantine_file"])

    # rejected
    _w(REJ_CSV, [{
        "doc_type": x["doc_type"],
        "reject_reason": "prior_rejected" if x["source_group"] == "papers_rejected_prior" else x["doc_type"],
        "doi": x["doi"],
        "title": x["title"],
        "page_count": x["page_count"],
        "text_length": x["text_length"],
        "source_path": x["path"],
        "source_group": x["source_group"],
    } for x in r["rejected"]],
        ["doc_type", "reject_reason", "doi", "title", "page_count",
         "text_length", "source_path", "source_group"])

    # unresolved
    _w(UNRES_CSV, [{
        "sha256": u["sha256"],
        "doi": u["doi"],
        "title": u["title"],
        "page_count": u["page_count"],
        "text_length": u["text_length"],
        "source_path": u["path"],
        "source_group": u["source_group"],
        "note": "no DOI and no high-confidence title/text match",
    } for u in r["unresolved"]],
        ["sha256", "doi", "title", "page_count", "text_length",
         "source_path", "source_group", "note"])

    # cleanup report
    lines = [
        "# Corpus Cleanup Report",
        "",
        "Non-destructive reorganization: originals were never deleted or moved;",
        "files were COPIED into their destination folders.",
        "",
        "## Final counts",
        "",
        f"- Total scanned PDFs: **{r['total']}**",
        f"- Rejected (unusable / prior-quarantined): **{r['rejected_n']}**",
        f"- SHA256 duplicates: **{r['sha_dup_n']}**",
        f"- DOI duplicates: **{r['doi_dup_n']}**",
        f"- Title/text duplicates: **{r['titletext_dup_n']}**",
        f"- Unresolved (kept, no high-confidence dedup): **{r['unresolved_n']}**",
        f"- Final unique valid papers (excl. unresolved): **{r['final_n'] - r['unresolved_n']}**",
        f"- final_corpus actual files (unique valid + unresolved): **{r['final_n']}**",
        "",
        "## Accounting",
        "",
        f"`{r['total']} total = {r['rejected_n']} rejected + {r['sha_dup_n']} sha_dup"
        f" + {r['doi_dup_n']} doi_dup + {r['titletext_dup_n']} title_text_dup"
        f" + {r['final_n']} final_corpus`",
        "",
        f"`final_corpus = {r['final_n'] - r['unresolved_n']} unique valid"
        f" + {r['unresolved_n']} unresolved`",
        "",
        "## Dedup rules applied",
        "",
        "1. SHA256 identical -> duplicate (byte-identical copies).",
        "2. Clear normalized DOI identical -> duplicate.",
        "3. No-DOI files: normalized title + text fingerprint high-confidence",
        "   (title equal/contained AND body fingerprint equal or shingle",
        f"   Jaccard >= {TITLE_TEXT_JACCARD}) -> duplicate.",
        "4. Otherwise -> unresolved (kept, no further heuristics).",
        "",
        "## Reject rule",
        "",
        "Unreadable / no text / error-login page / corrigendum / TOC / editorial,",
        "plus the 4 files already quarantined in data/papers/rejected/.",
        "Note: 'supplement_suspect' is NOT auto-rejected (known false-positive;",
        "a 10-page PNAS main article fires it), so such files are kept as articles.",
        "",
    ]
    CLEANUP.write_text("\n".join(lines), encoding="utf-8")


def print_plan(r: dict) -> None:
    print(f"total scanned          = {r['total']}")
    print(f"rejected               = {r['rejected_n']}")
    print(f"sha256 duplicates      = {r['sha_dup_n']}")
    print(f"doi duplicates         = {r['doi_dup_n']}")
    print(f"title/text duplicates  = {r['titletext_dup_n']}")
    print(f"unresolved             = {r['unresolved_n']}")
    print(f"final unique valid     = {r['final_n'] - r['unresolved_n']}")
    print(f"final_corpus files     = {r['final_n']}")
    print(f"check: {r['total']} == {r['rejected_n']}+{r['sha_dup_n']}+{r['doi_dup_n']}"
          f"+{r['titletext_dup_n']}+{r['final_n']}")
    print("\n--- rejected ---")
    for x in r["rejected"]:
        print(f"  [{x['doc_type']}] {x['file_name']}  ({x['source_group']})")
    print("\n--- title/text duplicates ---")
    for d in r["titletext_dup"]:
        print(f"  {d['file_name']}  title={d['title'][:60]!r}")
    print("\n--- unresolved ---")
    for u in r["unresolved"]:
        print(f"  {u['file_name'][:60]}  title={u['title'][:50]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="copy files and write reports")
    ap.add_argument("--plan", action="store_true", help="print summary only")
    args = ap.parse_args()

    records = load()
    r = run(records, apply=args.apply)
    if args.apply:
        print("Built folders and wrote reports:")
        for p in (MANIFEST, CLEANUP, DUP_CSV, REJ_CSV, UNRES_CSV):
            print(f"  {p}")
        print(f"  final_corpus dir        -> {FINAL}")
        print(f"  duplicates_quarantine   -> {DUP_DIR}")
        print(f"  rejected dir            -> {REJ_DIR}")
    print_plan(r)


if __name__ == "__main__":
    main()
