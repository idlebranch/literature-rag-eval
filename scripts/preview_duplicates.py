"""Preview duplicate structure using the already-reliable fields (sha256, doi)."""
import json
from collections import Counter, defaultdict
from pathlib import Path

CACHE = Path(r"C:\Users\10475\AI_PROJECT\literature_rag_eval_code\data\papers\_scan_cache.json")
recs = json.loads(CACHE.read_text(encoding="utf-8"))["records"]

print(f"total files = {len(recs)}")

# ---- sha256
by_sha = defaultdict(list)
for r in recs:
    by_sha[r["sha256"]].append(r)
sha_groups = {k: v for k, v in by_sha.items() if len(v) > 1}
print(f"\nunique sha256          = {len(by_sha)}")
print(f"sha256 groups with >1  = {len(sha_groups)}")
print(f"redundant copies       = {sum(len(v) - 1 for v in sha_groups.values())}")
print("group size histogram:", dict(Counter(len(v) for v in by_sha.values())))

print("\n--- cross-source-group overlap matrix (sha256 identical) ---")
pairs = Counter()
for v in sha_groups.values():
    gs = sorted({r["source_group"] for r in v})
    for i in range(len(gs)):
        for j in range(i, len(gs)):
            if i == j and sum(1 for r in v if r["source_group"] == gs[i]) > 1:
                pairs[(gs[i], gs[i])] += 1
            elif i != j:
                pairs[(gs[i], gs[j])] += 1
for (a, b), n in pairs.most_common():
    print(f"  {n:>4}  {a}  <->  {b}")

# ---- doi (on sha-unique representatives)
uniq = [v[0] for v in by_sha.values()]
print(f"\n=== after sha256 collapse: {len(uniq)} distinct files ===")
by_doi = defaultdict(list)
nodoi = []
for r in uniq:
    (by_doi[r["doi"]] if r["doi"] else nodoi).append(r)
doi_groups = {k: v for k, v in by_doi.items() if len(v) > 1}
print(f"distinct DOIs        = {len(by_doi)}")
print(f"DOI groups with >1   = {len(doi_groups)}")
print(f"extra copies via DOI = {sum(len(v) - 1 for v in doi_groups.values())}")
print(f"files with no DOI    = {len(nodoi)}")

print("\n--- DOI groups (different bytes, same DOI) ---")
for d, v in sorted(doi_groups.items(), key=lambda kv: -len(kv[1])):
    print(f"\n  DOI {d}  x{len(v)}")
    for r in v:
        print(f"     p={r['page_count']:<4} tl={r['text_length']:<7} refs={int(r['has_references'])} "
              f"ver={r['version_type']:<20} dt={r['doc_type']:<18} [{r['source_group']}]")
        print(f"       {r['file_name'][:88]}")

print(f"\n--- no-DOI files after sha collapse ({len(nodoi)}) ---")
for r in sorted(nodoi, key=lambda r: r["file_name"]):
    print(f"  p={r['page_count']:<4} tl={r['text_length']:<8} [{r['source_group']:<22}] {r['file_name'][:70]}")

print("\n=== projected unique papers (sha -> doi collapse) ===")
print(f"  {len(doi_groups) and ''}{len(by_doi) - (1 if '' in by_doi else 0) + len(nodoi)}"
      f"  (= {len(by_doi) - (1 if '' in by_doi else 0)} DOI-identified + {len(nodoi)} no-DOI)")
