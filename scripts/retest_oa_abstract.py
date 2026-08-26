"""Re-test the single OA abstract endpoint that returned status=0, to capture
the real network-level error. One request only. Never prints the API key.
"""
from __future__ import annotations
import json
import urllib.request
from pathlib import Path
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
KEY = dotenv_values(ROOT / ".env").get("ELSEVIER_API_KEY", "")
doi = "10.1016/j.heliyon.2024.e40370"
url = f"https://api.elsevier.com/content/abstract/doi/{doi}"

for attempt in range(3):
    req = urllib.request.Request(url, headers={"X-ELS-APIKey": KEY,
                                               "Accept": "application/json",
                                               "User-Agent": "WaterRAG-Diag/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            print(f"attempt {attempt+1}: HTTP {r.status} | bytes {len(raw)} | ct {r.headers.get('Content-Type')}")
            try:
                j = json.loads(raw)
                core = (j.get("abstracts-retrieval-response") or {}).get("coredata", {})
                print("   title:", (core.get("dc:title") or "")[:60])
                print("   openaccess_flag:", core.get("openaccess"))
            except Exception:
                pass
            break
    except Exception as e:
        body = b""
        try:
            if hasattr(e, "read"):
                body = e.read()
        except Exception:
            pass
        print(f"attempt {attempt+1}: status={getattr(e,'code',0)} | {type(e).__name__}: {str(e)[:110]}")
        if body:
            print("   body:", body[:200].decode("utf-8", "ignore"))
