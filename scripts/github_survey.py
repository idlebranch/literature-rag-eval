"""Search GitHub for similar projects (read-only public API)."""
from __future__ import annotations
import json, time, urllib.parse, urllib.request

QUERIES = [
    "paper RAG full text",
    "academic literature RAG knowledge base",
    "paper-qa",
    "ChatPaper",
    "batch download papers DOI",
    "unpaywall bulk download",
    "zotero batch download pdf",
    "water treatment RAG",
]

for q in QUERIES:
    url = ("https://api.github.com/search/repositories?q="
           + urllib.parse.quote(q) + "&sort=stars&order=desc&per_page=5")
    req = urllib.request.Request(url, headers={
        "User-Agent": "literature-survey/1.0",
        "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
    except Exception as e:
        print(f"\n[{q}] ERROR {e}")
        time.sleep(2)
        continue
    print(f"\n[{q}] total={d.get('total_count')}")
    for it in d.get("items", []):
        desc = (it.get("description") or "")[:90]
        print(f"  {it['full_name']} | stars={it['stargazers_count']} | "
              f"updated={it.get('pushed_at','')[:10]} | {desc}")
    time.sleep(2.5)
