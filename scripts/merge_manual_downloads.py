"""Merge 49 valid papers from manual download into corpus.

This script:
1. Reads valid_matched_report.csv (49 papers)
2. Reads manual_download_remaining.csv to get year, journal, candidate_id
3. Reads PDF to get page count
4. Copies PDF to data/papers/raw_new/
5. Appends to paper_manifest.csv
6. Removes matched entries from manual_download_remaining.csv
"""
import sys
sys.path = [p for p in sys.path if 'qwen-agent' not in p]

import csv
import shutil
import re
from pathlib import Path
import fitz

ROOT = Path(r"C:\Users\10475\AI_PROJECT\literature_rag_eval_code")
REPORTS_DIR = ROOT / "data" / "文献库" / "_reports"
MANIFEST_PATH = ROOT / "data" / "corpus_expansion" / "paper_manifest.csv"
MANUAL_REMAINING_PATH = ROOT / "data" / "corpus_expansion" / "manual_download_remaining.csv"
RAW_NEW_DIR = ROOT / "data" / "papers" / "raw_new"

# Read valid matched report
matched_path = REPORTS_DIR / "valid_matched_report.csv"
with open(matched_path, encoding='utf-8-sig') as f:
    matched = list(csv.DictReader(f))

print(f"Loaded {len(matched)} matched papers from {matched_path.name}")

# Build DOI -> row mapping from manual_download_remaining
with open(MANUAL_REMAINING_PATH, encoding='utf-8-sig') as f:
    manual_remaining = list(csv.DictReader(f))

doi_to_manual = {row['doi'].lower(): row for row in manual_remaining if row.get('doi')}
print(f"Loaded {len(doi_to_manual)} entries from manual_download_remaining.csv")

# Read existing manifest
with open(MANIFEST_PATH, encoding='utf-8-sig') as f:
    manifest = list(csv.DictReader(f))

existing_dois = {row['doi'].lower() for row in manifest}
print(f"Loaded {len(manifest)} entries from paper_manifest.csv")

# Process each matched paper
new_manifest_rows = []
new_candidate_id = len(manifest) + 1
matched_dois = set()

for i, paper in enumerate(matched):
    doi = paper['doi'].lower()
    pdf_path = Path(paper['path'])
    
    if doi in existing_dois:
        print(f"[{i+1}/{len(matched)}] SKIP (already in manifest): {doi}")
        continue
    
    # Get metadata from manual_remaining
    manual_row = doi_to_manual.get(doi)
    if not manual_row:
        print(f"[{i+1}/{len(matched)}] SKIP (not in manual_remaining): {doi}")
        continue
    
    # Read PDF to get page count
    try:
        doc = fitz.open(str(pdf_path))
        pages = doc.page_count
        doc.close()
    except Exception as e:
        print(f"[{i+1}/{len(matched)}] ERROR reading PDF: {e}")
        continue
    
    # Generate candidate_id
    candidate_id = f"C{new_candidate_id:04d}"
    new_candidate_id += 1
    
    # Generate new filename
    safe_title = re.sub(r'[^\w\s-]', '', manual_row['title'])[:80].strip()
    safe_title = re.sub(r'[-\s]+', '_', safe_title)
    new_filename = f"{manual_row['year']}_{safe_title}.pdf"
    new_path = RAW_NEW_DIR / new_filename
    
    # Copy file
    shutil.copy2(pdf_path, new_path)
    
    # Create manifest row
    manifest_row = {
        'candidate_id': candidate_id,
        'title': manual_row['title'],
        'year': manual_row['year'],
        'journal': manual_row['journal'],
        'doi': doi,
        'topic_cluster': manual_row['topic_cluster'],
        'priority_final': manual_row['priority_final'],
        'local_path': str(new_path.relative_to(ROOT)),
        'sha256': paper['sha256'],
        'pages': pages,
        'access_status': 'manual_download',
        'download_url': f"https://doi.org/{doi}"
    }
    
    new_manifest_rows.append(manifest_row)
    matched_dois.add(doi)
    
    print(f"[{i+1}/{len(matched)}] OK: {candidate_id} | {manual_row['title'][:60]}... | {pages}p")

print(f"\nProcessed {len(new_manifest_rows)} papers")

# Append to manifest
if new_manifest_rows:
    with open(MANIFEST_PATH, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=new_manifest_rows[0].keys())
        writer.writerows(new_manifest_rows)
    print(f"Appended {len(new_manifest_rows)} rows to paper_manifest.csv")

# Remove matched entries from manual_remaining
remaining_after = [row for row in manual_remaining if row['doi'].lower() not in matched_dois]

with open(MANUAL_REMAINING_PATH, 'w', newline='', encoding='utf-8-sig') as f:
    if manual_remaining:
        writer = csv.DictWriter(f, fieldnames=manual_remaining[0].keys())
        writer.writeheader()
        writer.writerows(remaining_after)

print(f"Removed {len(matched_dois)} entries from manual_download_remaining.csv")
print(f"Remaining in manual_download_remaining: {len(remaining_after)} entries")

# Summary
print(f"\n=== SUMMARY ===")
print(f"New papers added to manifest: {len(new_manifest_rows)}")
print(f"Total manifest entries now: {len(manifest) + len(new_manifest_rows)}")
print(f"Remaining manual downloads: {len(remaining_after)}")
