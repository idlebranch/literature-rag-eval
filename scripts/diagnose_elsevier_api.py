"""Elsevier Article Retrieval API final diagnosis.

One-shot diagnostic: for each of two DOIs (one OA, one subscription) call
the Abstract Retrieval endpoint and the Article Retrieval endpoint with two
Accept headers. Records only: DOI, endpoint, status, content-type, Elsevier
error code/message, response size, body-structure signals, entitlement hints.

Never prints the API key. No bulk download. No RAG-code changes.
"""
from __future__ import annotations
import gzip
import json
import time
import urllib.request
from pathlib import Path
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
KEY = dotenv_values(ROOT / ".env").get("ELSEVIER_API_KEY", "")
if not KEY:
    raise SystemExit("ELSEVIER_API_KEY missing from .env")

CASES = [
    # (label, doi, kind)
    ("A-OA", "10.1016/j.heliyon.2024.e40370", "open_access"),
    ("B-SUB", "10.1016/j.watres.2015.09.045", "subscription"),
]


def request(url: str, accept: str):
    req = urllib.request.Request(
        url, headers={"X-ELS-APIKey": KEY, "Accept": accept,
                      "User-Agent": "WaterRAG-Diag/1.0"}
    )
    meta = {"status": None, "content_type": None, "bytes": 0,
            "error_code": None, "error_message": None,
            "els_headers": {}, "body_signals": {}}
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            raw = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            meta["status"] = r.status
            meta["content_type"] = r.headers.get("Content-Type")
            meta["bytes"] = len(raw)
            for h in ("X-ELS-Req", "X-ELS-Status"):
                if r.headers.get(h):
                    meta["els_headers"][h] = r.headers.get(h)
            return meta, raw
    except Exception as e:
        body = b""
        try:
            if hasattr(e, "read"):
                body = e.read()
                if body[:2] == b"\x1f\x8b":
                    body = gzip.decompress(body)
        except Exception:
            pass
        meta["status"] = getattr(e, "code", 0)
        meta["content_type"] = getattr(e, "headers", {}).get("Content-Type") if hasattr(e, "headers") else None
        meta["bytes"] = len(body)
        meta["error_message"] = str(e)[:120]
        try:
            j = json.loads(body)
            st = ((j.get("service-error") or {}).get("status") or {})
            meta["error_code"] = st.get("statusCode")
            meta["error_message"] = st.get("statusText") or meta["error_message"]
        except Exception:
            pass
        return meta, body


def analyze_xml(body: bytes) -> dict:
    """Detect full-text structure signals in an Elsevier XML payload."""
    text = body.decode("utf-8", "ignore")
    return {
        "has_ce_sections": "<ce:sections" in text,
        "ce_para_count": text.count("<ce:para"),
        "has_abstract": ("<ce:abstract" in text) or ("<abstract" in text),
        "has_coredata_only": "full-text-retrieval-response" not in text,
        "first_title": (text[text.find("<dc:title>") + 10: text.find("</dc:title>")][:80]
                        if "<dc:title>" in text else ""),
    }


def analyze_json(body: bytes) -> dict:
    try:
        d = json.loads(body)
    except Exception:
        return {"parse": False}
    resp = d.get("full-text-retrieval-response")
    if resp is None:
        core = (d.get("abstracts-retrieval-response") or {}).get("coredata", {})
        return {
            "endpoint_kind": "abstract",
            "title": (core.get("dc:title") or "")[:80],
            "has_abstract_text": bool(core.get("dc:description")),
            "openaccess_flag": core.get("openaccess"),
            "entitlement_fields": [k for k in core if "entitl" in k.lower() or "access" in k.lower()],
        }
    core = resp.get("coredata", {})
    orig = resp.get("originalText") or ""
    return {
        "endpoint_kind": "article",
        "title": (core.get("dc:title") or "")[:80],
        "originalText_len": len(orig),
        "has_ce_sections": "<ce:sections" in orig,
        "ce_para_count": orig.count("<ce:para"),
        "entitlement_fields": [k for k in core if "entitl" in k.lower() or "access" in k.lower()],
    }


def main():
    print("API key loaded (length only):", len(KEY))
    for label, doi, kind in CASES:
        print("\n" + "=" * 70)
        print(f"[{label}] doi={doi} kind={kind}")
        # 1) abstract endpoint
        meta, body = request(f"https://api.elsevier.com/content/abstract/doi/{doi}",
                             "application/json")
        sig = analyze_json(body) if meta["status"] == 200 else {}
        print(f"  abstract | status={meta['status']} | ct={meta['content_type']} | bytes={meta['bytes']}")
        if meta["error_code"]:
            print(f"           | elsevier_error={meta['error_code']} | msg={meta['error_message']}")
        for k, v in sig.items():
            print(f"           | {k}: {v}")
        time.sleep(2)

        # 2) article endpoint, text/xml
        meta, body = request(f"https://api.elsevier.com/content/article/doi/{doi}",
                             "text/xml")
        print(f"  article(xml) | status={meta['status']} | ct={meta['content_type']} | bytes={meta['bytes']}")
        if meta["error_code"]:
            print(f"           | elsevier_error={meta['error_code']} | msg={meta['error_message']}")
        if meta["status"] == 200:
            sig = analyze_xml(body)
            for k, v in sig.items():
                print(f"           | {k}: {v}")
        time.sleep(2)

        # 3) article endpoint, application/json
        meta, body = request(f"https://api.elsevier.com/content/article/doi/{doi}",
                             "application/json")
        print(f"  article(json) | status={meta['status']} | ct={meta['content_type']} | bytes={meta['bytes']}")
        if meta["error_code"]:
            print(f"           | elsevier_error={meta['error_code']} | msg={meta['error_message']}")
        if meta["status"] == 200:
            sig = analyze_json(body)
            for k, v in sig.items():
                print(f"           | {k}: {v}")
        time.sleep(2)


if __name__ == "__main__":
    main()
