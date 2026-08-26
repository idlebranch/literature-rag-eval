"""QA pass 2: verify the reclassification and remaining metadata edge cases."""
import json
import re
from collections import Counter
from pathlib import Path

CACHE = Path(r"C:\Users\10475\AI_PROJECT\literature_rag_eval_code\data\papers\_scan_cache.json")
recs = json.loads(CACHE.read_text(encoding="utf-8"))["records"]

print("doc_type:", dict(Counter(r["doc_type"] for r in recs)))
print("title_source:", dict(Counter(r["title_source"] for r in recs)))
print("version_type:", dict(Counter(r["version_type"] for r in recs)))
print("shingles empty:", sum(1 for r in recs if not r["shingles"]))
print("abstract_norm empty:", sum(1 for r in recs if not r["abstract_norm"]))

print("\n=== non-article files ===")
for r in recs:
    if r["doc_type"] != "article":
        print(f"  {r['doc_type']}  p={r['page_count']} tl={r['text_length']} "
              f"[{r['source_group']}] {r['file_name'][:60]}")
        print(f"    doi={r['doi']} markers={r['markers']}")
        print(f"    head={r['head_text'][:300]!r}")

print("\n=== previously-'editorial' file, now re-checked ===")
for r in recs:
    if "Life-Sciences" in r["file_name"]:
        print(f"  {r['file_name']}  dt={r['doc_type']} p={r['page_count']} tl={r['text_length']}")
        print(f"  markers={r['markers']}")
        print(f"  title={r['title']!r}")
        print(f"  head={r['head_text'][:700]!r}")

print("\n=== still no title (%d) ===" % sum(1 for r in recs if not r["title"]))
for r in recs:
    if not r["title"]:
        print(f"  [{r['source_group']}] {r['file_name'][:65]}")
        print(f"    head={r['head_text'][:220]!r}")

print("\n=== CJK-thesis titles now extracted ===")
for r in recs:
    if r["title_source"] == "cjk_thesis_body":
        print(f"  {r['file_name'][:58]:<60} -> {r['title'][:55]!r}")

print("\n=== sanity: files with suspicious PUA/garbled titles ===")
for r in recs:
    if r["title"] and re.search(r"[\ue000-\uf8ff]", r["title"]):
        print(f"  [{r['title_source']}] {r['file_name'][:55]} -> {r['title'][:60]!r}")

print("\n=== thin articles (p<=4 or text<8000) ===")
for r in sorted([r for r in recs if r["doc_type"] == "article"
                 and (r["page_count"] <= 4 or r["text_length"] < 8000)],
                key=lambda r: r["text_length"]):
    print(f"  p={r['page_count']:<3} tl={r['text_length']:<7} refs={int(r['has_references'])} "
          f"tpp={r['markers'].get('text_per_page')} [{r['source_group']}] {r['file_name'][:48]}")
    print(f"    t={(r['title'] or '')[:85]!r} doi={r['doi'] or '-'}")

print("\n=== low text-per-page (possible scanned/incomplete) ===")
for r in sorted(recs, key=lambda r: r["markers"].get("text_per_page", 0))[:12]:
    print(f"  tpp={r['markers'].get('text_per_page'):<8} p={r['page_count']:<4} "
          f"tl={r['text_length']:<8} [{r['source_group']}] {r['file_name'][:52]}")
