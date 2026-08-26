"""QA the scan cache before trusting it for dedup decisions."""
import json
import re
from collections import Counter
from pathlib import Path

CACHE = Path(r"C:\Users\10475\AI_PROJECT\literature_rag_eval_code\data\papers\_scan_cache.json")
recs = json.loads(CACHE.read_text(encoding="utf-8"))["records"]

print(f"total={len(recs)}")
print("source_group:", dict(Counter(r["source_group"] for r in recs)))
print("version_type:", dict(Counter(r["version_type"] for r in recs)))
print("doi_source:", dict(Counter(r["doi_source"] for r in recs)))
print("title_source:", dict(Counter(r["title_source"] for r in recs)))
print("author_source:", dict(Counter(r["author_source"] for r in recs)))
print("year_source:", dict(Counter(r["year_source"] for r in recs)))
print("has_references:", Counter(r["has_references"] for r in recs))
print("images_only:", Counter(r["images_only"] for r in recs))

print("\n=== DOI sanity: malformed / suspicious ===")
bad = [r for r in recs if r["doi"] and not re.fullmatch(r"10\.\d{4,9}/\S+", r["doi"])]
print(f"malformed doi: {len(bad)}")
for r in bad[:10]:
    print("  ", r["doi"], "|", r["file_name"][:70])
long_doi = [r for r in recs if len(r["doi"]) > 60]
print(f"suspiciously long doi: {len(long_doi)}")
for r in long_doi[:10]:
    print("  ", r["doi"][:90], "|", r["file_name"][:60])

print("\n=== no DOI (%d) ===" % sum(1 for r in recs if not r["doi"]))
for r in recs:
    if not r["doi"]:
        print(f"  [{r['source_group']}] {r['file_name'][:60]:<62} p={r['page_count']:<4} "
              f"tl={r['text_length']:<7} dt={r['doc_type']:<20} t={(r['title'] or '')[:45]!r}")

print("\n=== no title (%d) ===" % sum(1 for r in recs if not r["title"]))
for r in recs:
    if not r["title"]:
        print(f"  [{r['source_group']}] {r['file_name'][:60]:<62} p={r['page_count']:<4} "
              f"tl={r['text_length']:<7} dt={r['doc_type']}")
        print(f"       head={r['head_text'][:160]!r}")

print("\n=== classified NOT article (%d) ===" % sum(1 for r in recs if r["doc_type"] != "article"))
for r in recs:
    if r["doc_type"] != "article":
        print(f"  {r['doc_type']:<24} p={r['page_count']:<4} tl={r['text_length']:<7} "
              f"[{r['source_group']}] {r['file_name'][:55]}")
        print(f"       doi={r['doi'] or '-'} sig={r['signals'][:3]}")

print("\n=== short/thin articles (page<=4 or text<8000), doc_type=article ===")
thin = [r for r in recs if r["doc_type"] == "article"
        and (r["page_count"] <= 4 or r["text_length"] < 8000)]
print(f"count={len(thin)}")
for r in sorted(thin, key=lambda r: r["text_length"]):
    print(f"  p={r['page_count']:<4} tl={r['text_length']:<7} refs={int(r['has_references'])} "
          f"[{r['source_group']}] {r['file_name'][:52]}")
    print(f"       t={(r['title'] or '')[:80]!r}")

print("\n=== 15 random title/author samples ===")
for r in recs[::24]:
    print(f"  [{r['source_group']}] {r['file_name'][:48]}")
    print(f"       title  = {(r['title'] or '')[:100]!r} ({r['title_source']})")
    print(f"       author = {(r['authors'] or '')[:70]!r} -> {r['first_author']!r} ({r['author_source']})")
    print(f"       year={r['year']}({r['year_source']}) doi={r['doi'] or '-'} jour={(r['journal'] or '')[:45]!r}")
