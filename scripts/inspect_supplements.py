"""Inspect the files flagged as supplement / thin, to fix classifier false positives."""
import json
import re
from pathlib import Path

CACHE = Path(r"C:\Users\10475\AI_PROJECT\literature_rag_eval_code\data\papers\_scan_cache.json")
recs = json.loads(CACHE.read_text(encoding="utf-8"))["records"]

names = ["es0c00712.pdf", "cs4c07556.pdf", "ja4c00429.pdf", "es1c04250.pdf",
         "nl2031713.pdf", "cm5b04457.pdf", "es8b04669.pdf"]
print("################ ACS files flagged supplement ################")
for r in recs:
    if r["file_name"] in names:
        print(f"\n===== {r['file_name']}  p={r['page_count']} tl={r['text_length']} "
              f"dt={r['doc_type']} refs={r['has_references']} doi={r['doi']}")
        print("HEAD[0:700]:", r["head_text"][:700])

print("\n\n################ files whose name suggests real SI ################")
for r in recs:
    if re.search(r"_si|si_|supp|suppl", r["file_name"], re.I):
        print(f"  {r['file_name'][:70]:<72} dt={r['doc_type']} p={r['page_count']}")

print("\n\n################ Distinguishing markers per flagged file ################")
hdr = f"{'file':<58}{'p':>4}{'refs':>5}{'ASSOC':>6}{'AUTHINF':>8}{'ABSTR':>6}{'S1':>5}{'top200SI':>9}"
print(hdr)
flagged = [r for r in recs if r["doc_type"].startswith("supplement")]
for r in flagged:
    h = r["head_text"].lower()
    top200 = "SI" if re.search(r"supporting information|supplementary", h[:200]) else "-"
    print(f"{r['file_name'][:56]:<58}{r['page_count']:>4}{int(r['has_references']):>5}"
          f"{'?':>6}{'?':>8}{('Y' if 'abstract' in h else '-'):>6}"
          f"{'?':>5}{top200:>9}")
    print(f"    head120={r['head_text'][:120]!r}")
