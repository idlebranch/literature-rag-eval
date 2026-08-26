"""Phase 2.5: second-pass full-text retrieval over four legal channels.

Channels tried in order per DOI (all go through the existing quality checks):
  1. OpenAlex official hosted content   (content.openalex.org)  -- needs OPENALEX_API_KEY
  2. CORE                               (api.core.ac.uk DOI search)
  3. Unpaywall                          (all oa_locations, publishedVersion first)
  4. OpenAlex locations                 (all locations[].pdf_url)

No paywall bypass, no Sci-Hub, no browser login. Only papers failing ALL four
channels stay in manual_download.csv. Existing 14 downloaded PDFs are kept;
SHA256 dedup against the whole corpus (existing 66 + raw_new).
"""
from __future__ import annotations
import csv, hashlib, json, re, time, urllib.request, urllib.parse, os, sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "corpus_expansion"
RAW_NEW = ROOT / "data" / "papers" / "raw_new"
REJECTED = ROOT / "data" / "papers" / "rejected"
INV = ROOT / "data" / "_inventory_existing.json"

EMAIL = "corpus.research@mailinator.com"
UA = "WaterRAG-Corpus/1.0 (research; mailto:%s)" % EMAIL
META_DELAY = 0.6
DL_DELAY = 1.0
RETRIES_429 = 3

OPENALEX_KEY = os.getenv("OPENALEX_API_KEY", "").strip()
BAD_TITLE = re.compile(
    r"\b(supporting information|supplementary|corrigendum|erratum|retraction|"
    r"table of contents|graphical abstract|cover picture|issue information)\b", re.I)


def norm_name(title, max_len=120):
    t = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "_", title).strip("_")
    return t[:max_len]


def http(url, timeout=40, max_bytes=None, retries=RETRIES_429):
    """GET with 429 backoff. Returns (status, body_bytes_or_errstr)."""
    last = ""
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read(max_bytes) if max_bytes else r.read()
                return r.status, body
        except Exception as e:
            code = getattr(e, "code", 0)
            last = str(e)[:100]
            if code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            return code, last
    return 429, last


def get_json(url, timeout=40):
    st, body = http(url, timeout=timeout)
    if st == 200 and isinstance(body, bytes):
        try:
            return json.loads(body)
        except Exception:
            return None
    return None


def sha256_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def is_pdf(head):
    return head[:5] == b"%PDF-"


def quality_check_bytes(body, doi, title):
    """Return (ok, reason) for an in-memory PDF."""
    if not is_pdf(body[:16]):
        return False, "not_pdf_magic"
    try:
        doc = fitz.open(stream=body, filetype="pdf")
    except Exception as e:
        return False, f"pymupdf_open_error:{e}"
    pages = doc.page_count
    if pages < 2:
        return False, f"too_few_pages({pages})"
    text = "".join(doc[i].get_text() for i in range(min(pages, 4)))
    if len(text) < 500:
        return False, "no_extractable_text"
    if BAD_TITLE.search(text[:1500]):
        return False, "supporting_info_or_corrigendum"
    # title-token sanity on first page
    def toks(s):
        s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
        return set(s.split()) - {"a", "an", "the", "of", "in", "for", "and", "on", "with", "to", "from", "by", "as", "at"}
    ct = toks(title)
    if ct:
        head_toks = toks(text[:400])
        if len(ct & head_toks) / len(ct) < 0.2:
            return False, "title_mismatch"
    return True, ""


# ---------------- channel 1: OpenAlex official content ----------------
def ch1_openalex_content(doi):
    if not OPENALEX_KEY:
        return None, "no_OPENALEX_API_KEY"
    w = get_json(f"https://api.openalex.org/works/doi:{doi}?mailto={EMAIL}")
    if not w:
        return None, "meta_failed"
    cu = (w.get("content_urls") or {}).get("pdf")
    if not cu:
        return None, "no_content_urls"
    sep = "&" if "?" in cu else "?"
    url = f"{cu}{sep}api_key={urllib.parse.quote(OPENALEX_KEY)}"
    st, body = http(url, timeout=90)
    if st == 200 and isinstance(body, bytes):
        return body, None
    return None, f"content_{st}"


# ---------------- channel 2: CORE ----------------
def ch2_core(doi):
    q = urllib.parse.quote(f'doi:"{doi}"')
    d = get_json(f"https://api.core.ac.uk/v3/search/works?q={q}&limit=3")
    time.sleep(META_DELAY)
    if not d:
        return None, "core_meta_failed"
    for res in d.get("results", []):
        for url in [res.get("downloadUrl"), (res.get("links") or [{}])[0].get("url") if res.get("links") else None,
                    *(res.get("sourceFulltextUrls") or [])]:
            if not url:
                continue
            st, body = http(url, timeout=60)
            if st == 200 and isinstance(body, bytes) and is_pdf(body[:16]):
                return body, None
    return None, "core_no_pdf"


# ---------------- channel 3: Unpaywall ----------------
def ch3_unpaywall(doi):
    d = get_json(f"https://api.unpaywall.org/v2/{doi}?email={urllib.parse.quote(EMAIL)}")
    time.sleep(META_DELAY)
    if not d:
        return None, "unpaywall_meta_failed"
    locs = d.get("oa_locations") or []
    # publishedVersion first, then accepted, then any
    def rank(l):
        v = l.get("version") or ""
        return 0 if v == "publishedVersion" else 1 if v == "acceptedVersion" else 2
    for loc in sorted(locs, key=rank):
        url = loc.get("url_for_pdf") or loc.get("url")
        if not url:
            continue
        st, body = http(url, timeout=60)
        if st == 200 and isinstance(body, bytes) and is_pdf(body[:16]):
            return body, None
    return None, "unpaywall_no_pdf"


# ---------------- channel 4: OpenAlex locations ----------------
def ch4_openalex_locations(doi):
    w = get_json(f"https://api.openalex.org/works/doi:{doi}?mailto={EMAIL}")
    time.sleep(META_DELAY)
    if not w:
        return None, "meta_failed"
    pdfs = [loc.get("pdf_url") for loc in w.get("locations") or [] if loc.get("pdf_url")]
    for url in pdfs:
        st, body = http(url, timeout=60)
        if st == 200 and isinstance(body, bytes) and is_pdf(body[:16]):
            return body, None
    return None, "oa_locations_no_pdf"


CHANNELS = [
    ("openalex_content", ch1_openalex_content),
    ("core", ch2_core),
    ("unpaywall", ch3_unpaywall),
    ("openalex_locations", ch4_openalex_locations),
]


def corpus_sha_set():
    """SHA256 of existing 66 + anything already in raw_new, for dedup."""
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


def main():
    manual = list(csv.DictReader(open(EXP / "manual_download.csv", encoding="utf-8-sig")))
    max_fetch = int(os.getenv("MAX_FETCH", "0"))
    if max_fetch > 0:
        manual = manual[:max_fetch]
    print(f"manual to retry: {len(manual)}  | OPENALEX_API_KEY set: {bool(OPENALEX_KEY)}")
    seen = corpus_sha_set()
    print(f"corpus sha pool: {len(seen)}")

    stats = {c: 0 for c, _ in CHANNELS}
    stats["dup_skipped"] = 0
    new_rows = []
    rejected_rows = []
    still_manual = []

    for i, row in enumerate(manual, 1):
        doi = row["doi"]
        title = row["title"]
        body = None
        via = None
        for name, fn in CHANNELS:
            b, err = fn(doi)
            if b:
                ok, reason = quality_check_bytes(b, doi, title)
                if ok:
                    body, via = b, name
                    break
                # failed QC on this channel; try next channel
            time.sleep(0.2)
        if body is None:
            still_manual.append(row)
            if i % 25 == 0:
                print(f"  progress {i}/{len(manual)}")
            continue

        sha = hashlib.sha256(body).hexdigest()
        if sha in seen:
            stats["dup_skipped"] += 1
            still_manual.append(row)
            continue
        seen.add(sha)

        fname = re.sub(r"\s+", "", f"{row['year']}_{norm_name(title)}.pdf")
        dest = RAW_NEW / fname
        dest.write_bytes(body)
        stats[via] += 1
        doc = fitz.open(dest)
        new_rows.append({
            "candidate_id": row["candidate_id"], "title": title, "year": row["year"],
            "journal": row["journal"], "doi": doi, "topic_cluster": row["topic_cluster"],
            "priority_final": row["priority_final"],
            "local_path": str(dest.relative_to(ROOT)), "sha256": sha, "pages": doc.page_count,
            "access_status": "open_access", "download_url": f"via:{via}",
        })
        if i % 25 == 0:
            print(f"  progress {i}/{len(manual)} new={len(new_rows)}")
        time.sleep(DL_DELAY)

    # ---- outputs ----
    mfields = ["candidate_id", "title", "year", "journal", "doi", "topic_cluster",
               "priority_final", "local_path", "sha256", "pages", "access_status", "download_url"]
    # merge into existing manifest
    mani_path = EXP / "paper_manifest.csv"
    existing_manifest = list(csv.DictReader(open(mani_path, encoding="utf-8-sig"))) if mani_path.exists() else []
    with mani_path.open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=mfields)
        wr.writeheader()
        wr.writerows(existing_manifest)
        wr.writerows(new_rows)

    manfields = ["candidate_id", "title", "year", "journal", "doi", "topic_cluster",
                 "priority_final", "access_status", "publisher_url", "oa_url", "note"]
    with (EXP / "manual_download.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=manfields)
        wr.writeheader()
        for r in still_manual:
            r["note"] = "all_four_channels_failed"
            wr.writerow(r)

    print("\n================ PHASE 2.5 RESULT ================")
    for c, _ in CHANNELS:
        print(f"  {c:20s}: {stats[c]}")
    print(f"  {'dup_skipped':20s}: {stats['dup_skipped']}")
    print(f"  new_valid_pdfs       : {len(new_rows)}")
    print(f"  final manual_download: {len(still_manual)}")
    pa = sum(1 for r in still_manual if r["priority_final"] == "A")
    print(f"  Priority A still manual: {pa}")
    print(f"  manifest total now   : {len(existing_manifest) + len(new_rows)}")


if __name__ == "__main__":
    main()
