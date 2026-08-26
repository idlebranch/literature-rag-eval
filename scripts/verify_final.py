"""Final verification: final_corpus is 270 distinct papers (unique sha, unique DOI)."""
import csv
from collections import Counter
from pathlib import Path

DATA = Path(r"C:\Users\10475\AI_PROJECT\literature_rag_eval_code\data\papers")

rows = list(csv.DictReader((DATA / "final_paper_manifest.csv").open(encoding="utf-8-sig")))
print("manifest rows:", len(rows))

shas = [r["sha256"] for r in rows]
print("distinct sha256:", len(set(shas)), "| any empty:", any(not s for s in shas))

dois = [r["doi"] for r in rows if r["doi"]]
print("DOI-bearing keepers:", len(dois), "| distinct DOIs:", len(set(dois)))

dup_dois = [d for d, c in Counter(dois).items() if c > 1]
print("duplicated DOIs among keepers:", dup_dois or "none")

res = Counter(r["resolution"] for r in rows)
print("resolution:", dict(res))

# verify on-disk copies exist and match count
final_files = [p.name for p in (DATA / "final_corpus").glob("*.pdf")]
print("final_corpus on-disk files:", len(final_files))
missing = [r["final_file"] for r in rows if r["final_file"] not in set(final_files)]
print("manifest->disk missing:", missing or "none")
