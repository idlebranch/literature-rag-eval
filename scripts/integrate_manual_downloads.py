"""Integrate manually downloaded PDFs from '文献库' into the project corpus.

Steps:
1. Scan '文献库' for PDFs and ZIPs.
2. Extract ZIPs to '_staging' subfolder (originals untouched).
3. QC all PDFs (magic bytes, PyMuPDF open, page count, text length, HTML/login exclusion).
4. Deduplicate via SHA256 against existing corpus (_inventory_existing.json + papers/raw_new).
5. Match DOIs against manual_download_remaining.csv and paper_manifest.csv.
6. Generate 4 CSV reports (no original files or manifests are modified).
"""
from __future__ import annotations
import csv, hashlib, json, os, re, shutil, zipfile
from pathlib import Path
import sys

# Remove qwen-agent paths from sys.path before importing pymupdf
sys.path = [p for p in sys.path if 'qwen-agent' not in p]

# Suppress PyMuPDF deprecation warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    import pymupdf as fitz
except ImportError:
    import fitz

ROOT = Path(r"C:\Users\10475\AI_PROJECT\literature_rag_eval_code")
SRC = ROOT / "data" / "文献库"
STAGE = SRC / "_staging"
REPORTS = SRC / "_reports"

INV_EXIST = ROOT / "data" / "_inventory_existing.json"
RAW_NEW = ROOT / "data" / "papers" / "raw_new"
MAN_REM = ROOT / "data" / "corpus_expansion" / "manual_download_remaining.csv"
PAP_MAN = ROOT / "data" / "corpus_expansion" / "paper_manifest.csv"

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s&\"<>;\\%]+")

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_doi_and_title(pdf_path: Path):
    doi, title = None, None
    try:
        doc = fitz.open(pdf_path)
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip()
        for field in ["subject", "keywords", "title", "creator"]:
            if not doi and meta.get(field):
                m = DOI_RE.search(meta[field])
                if m: doi = m.group(0).rstrip(".")
        if not doi:
            text = doc[0].get_text()[:3000]
            m = DOI_RE.search(text)
            if m: doi = m.group(0).rstrip(".")
        if not title:
            text = doc[0].get_text()[:500]
            title = text.split("\n")[0].strip()[:150]
        doc.close()
    except Exception as e:
        import sys
        print(f"[DEBUG] Exception in extract_doi_and_title for {pdf_path}: {e}", file=sys.stderr)
    return doi, title

def qc_pdf(pdf_path: Path) -> tuple[bool, str]:
    """Return (is_valid, reason)."""
    try:
        with pdf_path.open("rb") as f:
            head = f.read(1024)
        if head[:5] != b"%PDF-":
            if b"<html" in head.lower() or b"<!doctype" in head.lower():
                return False, "html_page_not_pdf"
            return False, "bad_magic_bytes"
    except Exception as e:
        return False, f"read_error: {e}"

    try:
        doc = fitz.open(pdf_path)
        pages = doc.page_count
        if pages <= 0:
            doc.close()
            return False, "zero_pages"
        text = "".join(doc[i].get_text() for i in range(min(pages, 3)))
        doc.close()
        if len(text.strip()) < 200:
            return False, f"insufficient_text(len={len(text.strip())})"
        low = text.lower()
        if ("please sign in" in low or "access denied" in low or "captcha" in low) and len(text) < 1500:
            return False, "login_or_access_denied_page"
        return True, ""
    except Exception as e:
        return False, f"pymupdf_error: {e}"

def main():
    print(f"Scanning {SRC}...", flush=True)
    STAGE.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    # 1. Extract ZIPs
    zips = [p for p in SRC.iterdir() if p.is_file() and p.suffix.lower() == ".zip"]
    print(f"Found {len(zips)} ZIPs. Extracting to {STAGE}...", flush=True)
    for zp in zips:
        out_dir = STAGE / zp.stem
        out_dir.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                for member in zf.namelist():
                    if member.lower().endswith(".pdf") and not member.startswith("__MACOSX"):
                        # flatten into out_dir
                        fname = Path(member).name
                        with zf.open(member) as src_f, (out_dir / fname).open("wb") as dst_f:
                            shutil.copyfileobj(src_f, dst_f)
        except Exception as e:
            print(f"  [!] Failed to extract {zp.name}: {e}", flush=True)

    # 2. Collect all PDFs (excluding _staging and _reports in the root scan)
    pdfs = []
    for p in SRC.iterdir():
        if p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append((p, "direct"))
    for p in STAGE.rglob("*.pdf"):
        pdfs.append((p, "zip_extracted"))
    print(f"Total PDFs to process: {len(pdfs)}", flush=True)

    # 3. Load existing corpus for dedup
    exist_sha = set()
    exist_doi = set()
    if INV_EXIST.exists():
        for r in json.loads(INV_EXIST.read_text(encoding="utf-8")):
            if r.get("sha256"): exist_sha.add(r["sha256"])
            if r.get("doi"): exist_doi.add(r["doi"].lower())
    if PAP_MAN.exists():
        for r in csv.DictReader(open(PAP_MAN, encoding="utf-8-sig")):
            if r.get("doi"): exist_doi.add(r["doi"].lower())
            if r.get("sha256"): exist_sha.add(r["sha256"])

    raw_new_sha = set()
    if RAW_NEW.exists():
        for p in RAW_NEW.glob("*.pdf"):
            try: raw_new_sha.add(sha256_file(p))
            except Exception: pass

    rem_doi = {}
    if MAN_REM.exists():
        for r in csv.DictReader(open(MAN_REM, encoding="utf-8-sig")):
            if r.get("doi"): rem_doi[r["doi"].lower()] = r

    # 4. Process PDFs
    valid_matched, valid_untracked, duplicates, rejected, unresolved = [], [], [], [], []

    for pdf_path, source_type in pdfs:
        is_valid, reason = qc_pdf(pdf_path)
        if not is_valid:
            rejected.append({"file": pdf_path.name, "source": source_type, "reason": reason, "path": str(pdf_path)})
            continue

        sha = sha256_file(pdf_path)
        doi, title = extract_doi_and_title(pdf_path)
        doi_low = doi.lower() if doi else ""

        rec = {"file": pdf_path.name, "source": source_type, "doi": doi, "title": title, "sha256": sha, "path": str(pdf_path)}

        if sha in exist_sha or sha in raw_new_sha:
            duplicates.append({**rec, "reason": "sha256_match_existing"})
            continue
        if doi_low and doi_low in exist_doi:
            duplicates.append({**rec, "reason": "doi_match_existing_manifest"})
            continue

        if doi_low and doi_low in rem_doi:
            rem_row = rem_doi[doi_low]
            valid_matched.append({**rec, "priority": rem_row.get("priority_final", ""), "topic": rem_row.get("topic_cluster", "")})
        elif doi:
            valid_untracked.append({**rec, "reason": "doi_found_but_not_in_target_lists"})
        else:
            unresolved.append({**rec, "reason": "no_doi_extracted"})

    # 5. Write reports
    def write_csv(path, rows, fields):
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write_csv(REPORTS / "valid_matched_report.csv", valid_matched,
              ["file", "source", "doi", "title", "priority", "topic", "sha256", "path"])
    write_csv(REPORTS / "valid_untracked_report.csv", valid_untracked,
              ["file", "source", "doi", "title", "reason", "sha256", "path"])
    write_csv(REPORTS / "unresolved_report.csv", unresolved,
              ["file", "source", "doi", "title", "reason", "sha256", "path"])
    write_csv(REPORTS / "duplicate_report.csv", duplicates,
              ["file", "source", "doi", "title", "reason", "sha256", "path"])
    write_csv(REPORTS / "rejected_report.csv", rejected,
              ["file", "source", "reason", "path"])

    print(f"\n========== SUMMARY ==========", flush=True)
    print(f"ZIPs extracted:        {len(zips)}", flush=True)
    print(f"Total PDFs processed:  {len(pdfs)}", flush=True)
    print(f"Valid & Matched:       {len(valid_matched)} (ready to merge into manifest)", flush=True)
    print(f"Valid but Untracked:   {len(valid_untracked)} (DOI found, but not in target lists)", flush=True)
    print(f"Unresolved (No DOI):   {len(unresolved)}", flush=True)
    print(f"Duplicates:            {len(duplicates)}", flush=True)
    print(f"Rejected (Bad QC):     {len(rejected)}", flush=True)
    print(f"\nReports saved to {REPORTS}", flush=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnose":
        # Diagnose DOI extraction for first 5 PDFs
        pdfs = []
        for p in SRC.iterdir():
            if p.is_file() and p.suffix.lower() == ".pdf":
                pdfs.append(p)
        for p in STAGE.rglob("*.pdf"):
            pdfs.append(p)
        
        for i, pdf_path in enumerate(pdfs[:5]):
            print(f"\n{'='*60}")
            print(f"PDF {i+1}: {pdf_path.name}")
            print(f"{'='*60}")
            doi, title = extract_doi_and_title(pdf_path)
            print(f"Extracted DOI: {doi}")
            print(f"Extracted title: {title}")
            
            # Also show raw metadata and text
            try:
                doc = fitz.open(pdf_path)
                meta = doc.metadata or {}
                print(f"\nMetadata: {meta}")
                print(f"\nFirst 500 chars of page 0:")
                print(doc[0].get_text()[:500])
                doc.close()
            except Exception as e:
                print(f"Error reading PDF: {e}")
    else:
        main()
