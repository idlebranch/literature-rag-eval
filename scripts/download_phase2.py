"""Phase 2: download papers_to_download.csv per access policy.

Strategy (no RAG-code changes, no indexing):
  - open_access            -> download oa_url (legal full text)
  - institution_access_*   -> attempt low-concurrency DOI->PDF probe on campus
                              net; keep if a valid PDF is obtained, else queue
                              manual_download.csv
  - manual_required        -> queue manual_download.csv
Never substitute a low-quality OA for an inaccessible high-value paper.

Quality checks per PDF:
  - HTTP content-type / %PDF magic header
  - PyMuPDF opens, page count >= 2, extractable text > 500 chars
  - not HTML/login/error page
  - not abstract-only / supporting-info / corrigendum (heuristic)
  - SHA256 computed for dedup
Outputs:
  data/papers/raw_new/<name>.pdf
  data/papers/rejected/<name>.pdf  + rejected_papers.csv
  manual_download.csv
  paper_manifest.csv
"""
from __future__ import annotations

import csv
import hashlib
import re
import time
import urllib.request
import socket
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "corpus_expansion"
RAW_NEW = ROOT / "data" / "papers" / "raw_new"
REJECTED = ROOT / "data" / "papers" / "rejected"
RAW_NEW.mkdir(parents=True, exist_ok=True)
REJECTED.mkdir(parents=True, exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36 corpus-research/1.0")
TIMEOUT = 60
DELAY_OA = 1.5
DELAY_INST = 3.0
STATE_FILE = EXP / "phase2_state.json"

BAD_TITLE = re.compile(
    r"\b(supporting information|supplementary|corrigendum|erratum|retraction|"
    r"table of contents|graphical abstract|cover picture|issue information)\b", re.I)


def norm_name(title: str, max_len: int = 120) -> str:
    t = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "_", title).strip("_")
    return t[:max_len]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_pdf_bytes(head: bytes) -> bool:
    return head[:5] == b"%PDF-"


def looks_html(head: bytes) -> bool:
    h = head[:500].lower()
    return b"<!doctype html" in h or b"<html" in h


def http_get(url: str) -> tuple[bytes | None, str, int]:
    """Return (body, content_type, status). Follow redirects."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read(), (r.headers.get("Content-Type") or ""), r.status
    except Exception:
        return None, "", 0


def quality_check(path: Path, doi: str, title: str) -> tuple[bool, str]:
    """Return (ok, reason)."""
    try:
        head = path.read_bytes()[:1024]
    except Exception as e:
        return False, f"read_error:{e}"
    if not is_pdf_bytes(head):
        if looks_html(head):
            return False, "html_not_pdf(login/paywall page)"
        return False, "not_pdf_magic"
    try:
        with fitz.open(path) as doc:
            pages = doc.page_count
            if pages < 2:
                return False, f"too_few_pages({pages})"
            text = ""
            for i in range(min(pages, 4)):
                text += doc[i].get_text()
            if len(text) < 500:
                return False, "no_extractable_text(abstract-only/scan)"
            # heuristic: flag supporting-info / corrigendum
            first = text[:1500]
            if BAD_TITLE.search(first):
                return False, "supporting_info_or_corrigendum"
    except Exception as e:
        return False, f"pymupdf_open_error:{e}"
    return True, ""


def resolve_doi_pdf(doi: str) -> str | None:
    """Best-effort: turn a DOI into a direct-ish PDF URL. Returns None if unknown."""
    # Known OA-friendly or direct patterns
    return f"https://doi.org/{doi}"


def openalex_pdf_urls(doi: str) -> list[str]:
    """Fallback: ask OpenAlex for all known PDF locations of this DOI."""
    import json as _json
    import urllib.parse as _up
    url = ("https://api.openalex.org/works/doi:" + _up.quote(doi)
           + "?select=locations&mailto=corpus-survey@local.invalid")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = _json.load(r)
    except Exception:
        return []
    urls = []
    for loc in d.get("locations") or []:
        pu = (loc or {}).get("pdf_url")
        if pu and pu not in urls:
            urls.append(pu)
    return urls


def try_download(row, attempt_inst: bool) -> dict:
    """Return dict(status, path, reason, url_used)."""
    doi = row["doi"]
    title = row["title"]
    fname = f"{row['year']}_{norm_name(title)}.pdf"
    fname = re.sub(r"\s+", "", fname)
    dest = RAW_NEW / fname

    if dest.exists() and dest.stat().st_size > 0:
        return {"status": "already", "path": str(dest), "reason": "", "url_used": ""}

    access = row["access_status"]
    urls = []
    if access == "open_access":
        if row.get("oa_url"):
            urls.append(row["oa_url"])
        urls.extend(openalex_pdf_urls(doi))
    if access == "institution_access_possible" and attempt_inst:
        urls.append(f"https://doi.org/{doi}")

    if not urls:
        return {"status": "manual", "path": "", "reason": "no_direct_url", "url_used": ""}

    for url in urls:
        body, ctype, status = http_get(url)
        if body is None or len(body) < 2048:
            continue
        if not is_pdf_bytes(body[:16]):
            continue  # HTML/paywall -> try next or manual
        dest.write_bytes(body)
        ok, reason = quality_check(dest, doi, title)
        if ok:
            return {"status": "ok", "path": str(dest), "reason": "", "url_used": url}
        else:
            # move to rejected
            rej = REJECTED / fname
            dest.rename(rej)
            return {"status": "rejected", "path": str(rej), "reason": reason, "url_used": url}

    return {"status": "manual", "path": "", "reason": "no_valid_pdf(obtain via campus login/EndNote)", "url_used": ""}


def main():
    with (EXP / "papers_to_download.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"papers to process: {len(rows)}")

    manifest, manual, rejected_rows = [], [], []
    seen_sha = set()
    ok_cnt = manual_cnt = rej_cnt = dup_cnt = 0

    # Pass 1: open access
    oa_rows = [r for r in rows if r["access_status"] == "open_access"]
    inst_rows = [r for r in rows if r["access_status"] == "institution_access_possible"]
    other_rows = [r for r in rows if r["access_status"] not in ("open_access", "institution_access_possible")]

    print(f"\n--- Pass 1: open_access ({len(oa_rows)}) ---")
    for i, r in enumerate(oa_rows, 1):
        res = try_download(r, attempt_inst=False)
        _record(res, r, manifest, manual, rejected_rows, seen_sha)
        if res["status"] == "ok": ok_cnt += 1
        elif res["status"] == "manual": manual_cnt += 1
        elif res["status"] == "rejected": rej_cnt += 1
        elif res["status"] == "duplicate": dup_cnt += 1
        if i % 10 == 0:
            print(f"  OA progress {i}/{len(oa_rows)} ok={ok_cnt} manual={manual_cnt} rej={rej_cnt}")
        time.sleep(DELAY_OA)

    print(f"\n--- Pass 2: institution_access_possible probe ({len(inst_rows)}) ---")
    for i, r in enumerate(inst_rows, 1):
        res = try_download(r, attempt_inst=True)
        _record(res, r, manifest, manual, rejected_rows, seen_sha)
        if res["status"] == "ok": ok_cnt += 1
        elif res["status"] == "manual": manual_cnt += 1
        elif res["status"] == "rejected": rej_cnt += 1
        elif res["status"] == "duplicate": dup_cnt += 1
        if i % 20 == 0:
            print(f"  INST progress {i}/{len(inst_rows)} ok={ok_cnt} manual={manual_cnt} rej={rej_cnt}")
        time.sleep(DELAY_INST)

    print(f"\n--- Pass 3: manual_required ({len(other_rows)}) ---")
    for r in other_rows:
        manual.append(_manual_row(r, "access_requires_manual"))
        manual_cnt += 1

    # ---- write outputs ----
    mfields = ["candidate_id", "title", "year", "journal", "doi", "topic_cluster",
               "priority_final", "local_path", "sha256", "pages", "access_status", "download_url"]
    with (EXP / "paper_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=mfields)
        wr.writeheader()
        wr.writerows(manifest)

    manfields = ["candidate_id", "title", "year", "journal", "doi", "topic_cluster",
                 "priority_final", "access_status", "publisher_url", "oa_url", "note"]
    with (EXP / "manual_download.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=manfields)
        wr.writeheader()
        wr.writerows(manual)

    rejfields = ["candidate_id", "title", "doi", "reason", "local_path"]
    with (EXP / "rejected_papers.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=rejfields)
        wr.writeheader()
        wr.writerows(rejected_rows)

    print("\n================ PHASE 2 SUMMARY ================")
    print(f"downloaded_ok={ok_cnt}  duplicate_skipped={dup_cnt}")
    print(f"manual_download={manual_cnt}  rejected_pdf={rej_cnt}")
    print(f"manifest rows={len(manifest)}")


def _manual_row(r, note):
    return {
        "candidate_id": r["candidate_id"], "title": r["title"], "year": r["year"],
        "journal": r["journal"], "doi": r["doi"], "topic_cluster": r["topic_cluster"],
        "priority_final": r.get("priority_final", r.get("priority", "")),
        "access_status": r["access_status"],
        "publisher_url": r.get("publisher_url", f"https://doi.org/{r['doi']}"),
        "oa_url": r.get("oa_url", ""), "note": note,
    }


def _record(res, r, manifest, manual, rejected_rows, seen_sha):
    if res["status"] == "ok":
        p = Path(res["path"])
        sha = sha256_file(p)
        if sha in seen_sha:
            p.unlink()
            res["status"] = "duplicate"
            return
        seen_sha.add(sha)
        with fitz.open(p) as doc:
            pages = doc.page_count
        manifest.append({
            "candidate_id": r["candidate_id"], "title": r["title"], "year": r["year"],
            "journal": r["journal"], "doi": r["doi"], "topic_cluster": r["topic_cluster"],
            "priority_final": r.get("priority_final", r.get("priority", "")),
            "local_path": str(p.relative_to(ROOT)), "sha256": sha, "pages": pages,
            "access_status": r["access_status"], "download_url": res["url_used"],
        })
    elif res["status"] == "manual":
        manual.append(_manual_row(r, res["reason"]))
    elif res["status"] == "rejected":
        rejected_rows.append({
            "candidate_id": r["candidate_id"], "title": r["title"], "doi": r["doi"],
            "reason": res["reason"], "local_path": res["path"],
        })
    elif res["status"] == "already":
        # count existing good file into manifest
        p = Path(res["path"])
        sha = sha256_file(p)
        if sha in seen_sha:
            res["status"] = "duplicate"
            return
        seen_sha.add(sha)
        with fitz.open(p) as doc:
            pages = doc.page_count
        manifest.append({
            "candidate_id": r["candidate_id"], "title": r["title"], "year": r["year"],
            "journal": r["journal"], "doi": r["doi"], "topic_cluster": r["topic_cluster"],
            "priority_final": r.get("priority_final", r.get("priority", "")),
            "local_path": str(p.relative_to(ROOT)), "sha256": sha, "pages": pages,
            "access_status": r["access_status"], "download_url": "",
        })
        res["status"] = "ok"


if __name__ == "__main__":
    main()
