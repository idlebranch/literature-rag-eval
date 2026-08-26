"""Step 1 of the 4-step plan: bulk legal OA retrieval for the 222-paper list.

Pipeline per DOI:
  1. Unpaywall: all oa_locations (publishedVersion > acceptedVersion > others)
  2. OpenAlex: all locations[].pdf_url as fallback
  3. Download each candidate URL in order until one passes quality checks
Quality checks (existing standards): %PDF magic, PyMuPDF opens, pages >= 2,
extractable text > 500 chars, not supporting-info/corrigendum, title-token
sanity, SHA256 dedup against existing 66 + raw_new.

No paywall bypass, no Sci-Hub, polite rate limits. Resumable via state file.
Outputs: merged paper_manifest.csv + manual_download_remaining.csv (with routing).
"""
from __future__ import annotations
import csv, gzip, hashlib, json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "corpus_expansion"
RAW_NEW = ROOT / "data" / "papers" / "raw_new"
INV = ROOT / "data" / "_inventory_existing.json"
STATE = EXP / "step1_state.json"

EMAIL = "corpus.research@mailinator.com"
UA = "WaterRAG-Corpus/1.0 (research; mailto:%s)" % EMAIL
META_DELAY = 0.35
DL_DELAY = 1.0
RETRIES_429 = 3

BAD_TITLE = re.compile(
    r"\b(supporting information|supplementary|corrigendum|erratum|retraction|"
    r"table of contents|graphical abstract|cover picture|issue information)\b", re.I)


def http(url, timeout=45, max_bytes=None, retries=RETRIES_429):
    last = ""
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read(max_bytes) if max_bytes else r.read()
                return r.status, body
        except Exception as e:
            code = getattr(e, "code", 0)
            last = str(e)[:120]
            if code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            if code == 403 or code == 401 or code == 404:
                return code, last
            return code, last
    return 429, last


def get_json(url, timeout=45):
    st, body = http(url, timeout=timeout)
    if st == 200 and isinstance(body, bytes):
        try:
            return json.loads(body)
        except Exception:
            return None
    return None


def norm_name(title, max_len=120):
    t = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "_", title).strip("_")
    return t[:max_len]


def sha256_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def toks(s):
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return set(s.split()) - {"a", "an", "the", "of", "in", "for", "and", "on",
                             "with", "to", "from", "by", "as", "at", "its", "via"}


def quality_check(body, title):
    """Return (ok, reason, pages)."""
    if body[:5] != b"%PDF-":
        return False, "not_pdf_magic", 0
    try:
        doc = fitz.open(stream=body, filetype="pdf")
    except Exception as e:
        return False, f"pymupdf_open_error:{e}", 0
    pages = doc.page_count
    if pages < 2:
        return False, f"too_few_pages({pages})", pages
    text = "".join(doc[i].get_text() for i in range(min(pages, 4)))
    if len(text) < 500:
        return False, "no_extractable_text", pages
    if BAD_TITLE.search(text[:1500]):
        return False, "supporting_info_or_corrigendum", pages
    ct = toks(title)
    if ct:
        overlap = len(ct & toks(text[:400])) / len(ct)
        if overlap < 0.2:
            return False, f"title_mismatch(overlap={overlap:.2f})", pages
    return True, "", pages


def rebuild_222():
    """222 = main list 240 - downloaded(manifest) - rejected."""
    main = list(csv.DictReader(open(EXP / "papers_to_download.csv", encoding="utf-8-sig")))
    done = set()
    mani_path = EXP / "paper_manifest.csv"
    if mani_path.exists():
        for r in csv.DictReader(open(mani_path, encoding="utf-8-sig")):
            done.add(r["doi"].lower())
    rej_path = EXP / "rejected_papers.csv"
    if rej_path.exists():
        for r in csv.DictReader(open(rej_path, encoding="utf-8-sig")):
            done.add(r["doi"].lower())
    todo = [r for r in main if r["doi"].lower() not in done]
    return todo


def corpus_sha_set():
    shas = set()
    try:
        for r in json.loads(INV.read_text(encoding="utf-8")):
            if r.get("sha256"):
                shas.add(r["sha256"])
    except Exception:
        pass
    for p in RAW_NEW.glob("*.pdf"):
        try:
            shas.add(sha256_file(p))
        except Exception:
            pass
    return shas


def candidate_urls(doi):
    """Merge Unpaywall + OpenAlex locations. Returns (urls, sources_meta)."""
    urls = []
    meta = {"unpaywall_oa": None, "up_locs": 0, "oa_locs": 0}

    up = get_json(f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={urllib.parse.quote(EMAIL)}")
    time.sleep(META_DELAY)
    if up:
        meta["unpaywall_oa"] = up.get("is_oa")
        locs = up.get("oa_locations") or []
        meta["up_locs"] = len(locs)
        def rank(l):
            v = l.get("version") or ""
            return 0 if v == "publishedVersion" else 1 if v == "acceptedVersion" else 2
        for loc in sorted(locs, key=rank):
            for u in [loc.get("url_for_pdf"), loc.get("url")]:
                if u and u not in urls:
                    urls.append(u)

    w = get_json(f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}?mailto={EMAIL}")
    time.sleep(META_DELAY)
    if w:
        locs = w.get("locations") or []
        meta["oa_locs"] = len(locs)
        for loc in locs:
            u = loc.get("pdf_url")
            if u and u not in urls:
                urls.append(u)
    return urls, meta


def main():
    todo = rebuild_222()
    print(f"remaining papers: {len(todo)}", flush=True)

    state = {}
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
    done_dois = set(state.get("done", []))
    results = state.get("results", [])
    todo = [r for r in todo if r["doi"].lower() not in done_dois]
    print(f"after resume filter: {len(todo)} to process", flush=True)

    seen = corpus_sha_set()
    print(f"corpus sha pool: {len(seen)}", flush=True)
    counters = {"ok": 0, "dup": 0, "qc_fail": 0, "no_url": 0, "all_fail": 0}

    for i, row in enumerate(todo, 1):
        doi = row["doi"]
        title = row["title"]
        urls, meta = candidate_urls(doi)
        if not urls:
            counters["no_url"] += 1
            results.append({"doi": doi, "status": "no_oa_url", "meta": meta})
            done_dois.add(doi.lower())
            if i % 20 == 0:
                _save(state, done_dois, results)
                print(f"[{i}/{len(todo)}] {counters}", flush=True)
            continue

        got = False
        for url in urls:
            st, body = http(url, timeout=60)
            if st != 200 or not isinstance(body, bytes):
                continue
            if body[:5] != b"%PDF-":
                continue
            ok, reason, pages = quality_check(body, title)
            if not ok:
                continue
            sha = hashlib.sha256(body).hexdigest()
            if sha in seen:
                counters["dup"] += 1
                results.append({"doi": doi, "status": "duplicate_sha", "meta": meta})
                got = True
                break
            seen.add(sha)
            fname = re.sub(r"\s+", "", f"{row['year']}_{norm_name(title)}.pdf")
            dest = RAW_NEW / fname
            if dest.exists():
                dest = RAW_NEW / f"{dest.stem}_{sha[:8]}.pdf"
            dest.write_bytes(body)
            counters["ok"] += 1
            results.append({
                "doi": doi, "status": "downloaded", "local_path": str(dest.relative_to(ROOT)),
                "sha256": sha, "pages": pages, "via": url,
                "row": {k: row[k] for k in ("candidate_id", "title", "year", "journal",
                                             "topic_cluster", "priority_final")},
            })
            got = True
            break
        if not got:
            counters["all_fail"] += 1
            results.append({"doi": doi, "status": "all_urls_failed", "meta": meta,
                            "urls_tried": len(urls)})
        done_dois.add(doi.lower())
        if i % 20 == 0:
            _save(state, done_dois, results)
            print(f"[{i}/{len(todo)}] {counters}", flush=True)
        time.sleep(DL_DELAY)

    _save(state, done_dois, results)
    write_outputs(results)
    print("\n=== STEP 1 SUMMARY ===", flush=True)
    print(json.dumps(counters, indent=1), flush=True)


def _save(state, done_dois, results):
    state["done"] = list(done_dois)
    state["results"] = results
    STATE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def write_outputs(results):
    # merge downloads into manifest
    mani_path = EXP / "paper_manifest.csv"
    existing_manifest = list(csv.DictReader(open(mani_path, encoding="utf-8-sig"))) if mani_path.exists() else []
    existing_dois = {r["doi"].lower() for r in existing_manifest}
    fields = ["candidate_id", "title", "year", "journal", "doi", "topic_cluster",
              "priority_final", "local_path", "sha256", "pages", "access_status", "download_url"]
    new_rows = []
    for res in results:
        if res["status"] != "downloaded":
            continue
        if res["doi"].lower() in existing_dois:
            continue
        r = res["row"]
        new_rows.append({
            "candidate_id": r["candidate_id"], "title": r["title"], "year": r["year"],
            "journal": r["journal"], "doi": res["doi"], "topic_cluster": r["topic_cluster"],
            "priority_final": r["priority_final"], "local_path": res["local_path"],
            "sha256": res["sha256"], "pages": res["pages"], "access_status": "open_access",
            "download_url": res["via"][:200],
        })
    with mani_path.open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(existing_manifest)
        wr.writerows(new_rows)

    # remaining manual list with routing
    main = {r["doi"].lower(): r for r in csv.DictReader(open(EXP / "papers_to_download.csv", encoding="utf-8-sig"))}
    remaining = []
    for res in results:
        if res["status"] in ("downloaded", "duplicate_sha"):
            continue
        row = main.get(res["doi"].lower())
        if not row:
            continue
        doi = res["doi"]
        if doi.startswith("10.1016/"):
            route, url = "校园网 ScienceDirect 人工下载", f"https://doi.org/{doi}"
        elif doi.startswith(("10.1021/", "10.1002/", "10.1007/", "10.1039/", "10.1111/", "10.1080/")):
            route, url = "校园网出版社页面人工下载", f"https://doi.org/{doi}"
        else:
            route, url = "CASHL/CALIS 文献传递", f"https://doi.org/{doi}"
        remaining.append({
            "candidate_id": row["candidate_id"], "priority_final": row["priority_final"],
            "title": row["title"], "year": row["year"], "journal": row["journal"],
            "doi": doi, "topic_cluster": row["topic_cluster"],
            "failed_status": res["status"], "route_suggestion": route,
            "publisher_url": url, "oa_url": row.get("oa_url", ""),
        })
    with (EXP / "manual_download_remaining.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(remaining[0].keys()) if remaining else
                            ["candidate_id", "priority_final", "title", "year", "journal",
                             "doi", "topic_cluster", "failed_status", "route_suggestion",
                             "publisher_url", "oa_url"])
        wr.writeheader()
        wr.writerows(remaining)
    print(f"manifest now: {len(existing_manifest) + len(new_rows)} rows (new +{len(new_rows)})", flush=True)
    print(f"manual remaining: {len(remaining)} rows", flush=True)


if __name__ == "__main__":
    main()
