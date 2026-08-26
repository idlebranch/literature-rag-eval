"""Probe the four legal full-text channels with real DOIs from manual_download.csv.
Determines which channels work WITHOUT api keys, so the main fetcher can be built
correctly. Metadata + tiny content probes only; no bulk downloads.
"""
from __future__ import annotations
import csv, json, time, urllib.request, urllib.parse, sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "data" / "corpus_expansion"
UA = "WaterRAG-Corpus/1.0 (research; mailto:corpus.research@mailinator.com)"

def http(url, timeout=30, max_bytes=None):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(max_bytes) if max_bytes else r.read()
            return r.status, body
    except Exception as e:
        body = e.read(200) if hasattr(e, "read") else b""
        return getattr(e, "code", 0), str(e)[:80] + " | " + body[:60].decode("utf-8", "ignore")

def main():
    rows = list(csv.DictReader(open(EXP / "manual_download.csv", encoding="utf-8-sig")))
    print("manual total:", len(rows))

    # ---- Channel 1: OpenAlex official content (no key) ----
    print("\n=== Channel 1: OpenAlex official content (content.openalex.org, no key) ===")
    for r in rows[:4]:
        doi = r["doi"]
        st, body = http(f"https://api.openalex.org/works/doi:{doi}?mailto=corpus.research@mailinator.com")
        if st != 200:
            print(f"  {doi}: meta {st}"); continue
        w = json.loads(body)
        cu = (w.get("content_urls") or {}).get("pdf")
        if not cu:
            print(f"  {doi}: no content_urls.pdf"); continue
        st2, body2 = http(cu, timeout=60, max_bytes=16)
        magic = body2[:5] if isinstance(body2, bytes) else ""
        print(f"  {doi}: content {st2} magic={magic} url={cu[:70]}")
        time.sleep(0.5)

    # ---- Channel 2: CORE (no key) ----
    print("\n=== Channel 2: CORE (api.core.ac.uk, no key) ===")
    doi = rows[0]["doi"]
    q = urllib.parse.quote(f'doi:"{doi}"')
    st, body = http(f"https://api.core.ac.uk/v3/search/works?q={q}&limit=1")
    print(f"  status: {st} body: {body[:120]}")
    time.sleep(2)
    st, body = http(f"https://api.core.ac.uk/v3/search/works?q={q}&limit=1")
    print(f"  retry-after-delay status: {st}")

    # ---- Channel 3: Unpaywall ----
    print("\n=== Channel 3: Unpaywall (no key, email required) ===")
    cnt = {"oa": 0, "with_pdf": 0, "repo": 0}
    for r in rows[:30]:
        doi = r["doi"]
        st, body = http(f"https://api.unpaywall.org/v2/{doi}?email=corpus.research@mailinator.com")
        if st != 200:
            continue
        u = json.loads(body)
        if u.get("is_oa"):
            cnt["oa"] += 1
            for loc in u.get("oa_locations") or []:
                if loc.get("url_for_pdf"):
                    cnt["with_pdf"] += 1
                    if loc.get("host_type") == "repository":
                        cnt["repo"] += 1
                    break
        time.sleep(0.3)
    print(f"  of first 30: is_oa={cnt['oa']} with_pdf_url={cnt['with_pdf']} repository={cnt['repo']}")

    # ---- Channel 4: OpenAlex locations (all) ----
    print("\n=== Channel 4: OpenAlex all locations pdf_url ===")
    cnt4 = {"has_pdf_loc": 0}
    for r in rows[:30]:
        doi = r["doi"]
        st, body = http(f"https://api.openalex.org/works/doi:{doi}?mailto=corpus.research@mailinator.com")
        if st != 200:
            continue
        w = json.loads(body)
        pdfs = [loc.get("pdf_url") for loc in w.get("locations") or [] if loc.get("pdf_url")]
        if pdfs:
            cnt4["has_pdf_loc"] += 1
        time.sleep(0.3)
    print(f"  of first 30: works with >=1 location pdf_url = {cnt4['has_pdf_loc']}")

if __name__ == "__main__":
    main()
