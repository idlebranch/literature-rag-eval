"""Diagnose DOI extraction failure from a sample PDF."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.integrate_manual_downloads import extract_doi_and_title, SRC, STAGE

# Find first PDF in staging
sample = next(STAGE.rglob("*.pdf"), None)
if not sample:
    print("No PDFs found in staging")
    sys.exit(1)

print(f"Sample PDF: {sample}")
doi, title = extract_doi_and_title(sample)
print(f"Extracted DOI: {doi}")
print(f"Extracted title: {title}")

# Also check raw text
import fitz
doc = fitz.open(sample)
meta = doc.metadata or {}
print(f"\nMetadata keys: {list(meta.keys())}")
for k, v in meta.items():
    if v and len(str(v)) < 500:
        print(f"  {k}: {v}")
print(f"\nFirst 1000 chars of page 0:")
print(doc[0].get_text()[:1000])
