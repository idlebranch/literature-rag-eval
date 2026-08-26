"""Phase A: read-only forensic scan of every PDF currently in the project.

Extracts, per file: sha256, page count, real text, DOI (metadata + body),
title, authors, year, journal, document-type signals and completeness signals.
Writes everything to a JSON cache so the dedup pass can be iterated cheaply.

Read-only: this script never moves, writes or deletes a PDF.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
import zlib
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(r"C:\Users\10475\AI_PROJECT\literature_rag_eval_code")
DATA = ROOT / "data"
CACHE = DATA / "papers" / "_scan_cache.json"

# (source_group, directory, recurse) -- order defines scan order only.
SOURCES = [
    ("pdfs_root", DATA / "pdfs", False),
    ("pdfs_curated_20260718", DATA / "pdfs" / "curated_20260718", False),
    ("papers_raw_new", DATA / "papers" / "raw_new", True),
    ("papers_rejected_prior", DATA / "papers" / "rejected", True),
    ("wenxianku_root", DATA / "\u6587\u732e\u5e93", False),
    ("wenxianku_nested", DATA / "\u6587\u732e\u5e93" / "\u6587\u732e\u5e93", True),
    ("staging_sciencedirect", DATA / "\u6587\u732e\u5e93" / "_staging", True),
]

# ---------------------------------------------------------------- normalisation

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.I)
# Trailing junk that regularly gets glued onto a DOI by PDF text extraction.
DOI_TRAIL = ".,;:)]}>\u201d\u2019'\"\u3002\uff0c-\u2013\u2014_/"


def normalize_doi(raw: str | None) -> str:
    """lowercase, strip resolver prefixes / 'doi:' / surrounding punctuation."""
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", str(raw)).strip()
    s = s.replace("\u200b", "").replace("\xa0", " ").strip()
    s = re.sub(r"^\s*(?:https?://)?(?:dx\.)?doi\.org/", "", s, flags=re.I)
    s = re.sub(r"^\s*(?:doi|DOI)\s*[:\s]\s*", "", s)
    s = s.strip().strip(DOI_TRAIL)
    s = s.lower()
    # kill anything after whitespace that survived
    s = s.split()[0] if s.split() else ""
    # Elsevier PDFs often append the sentinel below to the footer DOI.
    for stop in ("getrights", "sciencedirect", "http", "\u00a9"):
        i = s.find(stop)
        if i > 0:
            s = s[:i]
    s = s.strip(DOI_TRAIL)
    return s if s.startswith("10.") and "/" in s else ""


def find_dois(text: str) -> list[str]:
    out, seen = [], set()
    for m in DOI_RE.finditer(text or ""):
        d = normalize_doi(m.group(0))
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def norm_text(s: str) -> str:
    """aggressive normalisation for fingerprints: alnum only, lowercased."""
    s = unicodedata.normalize("NFKD", s or "").lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def norm_title(s: str) -> str:
    """normalisation for title matching: alnum + single spaces, lowercased."""
    s = unicodedata.normalize("NFKD", s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- extraction

JUNK_TITLE_RE = re.compile(
    r"^(untitled|microsoft word|doc\d*|document\d*|print|pdf|"
    r"\d{1,3}|[\d\s.\-_]+|.{0,12})$",
    re.I,
)


# PII / ISSN / article-number strings that publishers leave in the Title field.
PII_TITLE_RE = re.compile(
    r"^(?:PII[:\s]|S?\d{4}-\d{3}[\dxX]\s*\(\d{2}\)|\d{4}-\d{3}[\dxX]$|"
    r"S\d{13,}|10\.\d{4,9}/|1-s2\.0-|es\d{6}|doi:)",
    re.I,
)


def clean_meta_title(t: str | None) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t).replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if not t or JUNK_TITLE_RE.match(t):
        return ""
    # filename-looking metadata titles
    if re.search(r"\.(pdf|doc|docx|tex|indd|qxd)$", t, re.I):
        return ""
    if PII_TITLE_RE.match(t):
        return ""
    # a PII/ISSN with no letters at all is never a title
    if not re.search(r"[A-Za-z一-鿿]{3}", t):
        return ""
    if t.count(" ") == 0 and len(t) < 25:
        return ""
    return t


CJK_RE = re.compile(r"[一-鿿]")
THESIS_BOILER_RE = re.compile(
    r"研究生|毕业论文|学位论文|申请|专业|作者姓名|指导教师|学科|摘\s*要|"
    r"分类号|密级|编号|学\s*号|答辩|授位|培养|大学|学院|工程硕士|中图"
)
# Lines that look like a citation, a funding note or a CNKI publication banner.
CJK_NOT_TITLE_RE = re.compile(
    r"\[J\]|\[D\]|\[基金项目\]|基金项目|网络首发|link\.cnki|doi\s*[:：]|"
    r"等\s*[.．]|收稿日期|作者简介|通信作者|第\s*\d+\s*卷|"
    r"^\d|引用格式|万方|中国知网|ISSN|CNKI",
    re.I,
)


def _cjk_title_ok(t: str) -> bool:
    if not t or not (6 <= len(t) <= 90):
        return False
    if CJK_NOT_TITLE_RE.search(t) or THESIS_BOILER_RE.search(t):
        return False
    if len(CJK_RE.findall(t)) < 5:
        return False
    # citation-ish: too many digits/punctuation
    if len(re.findall(r"[\d,;.()\[\]]", t)) > len(t) * 0.3:
        return False
    return True


def title_from_filename_cnki(file_name: str) -> str:
    """CNKI names files `<标题>_<作者>.pdf`; the stem is a usable display title."""
    stem = re.sub(r"\.pdf$", "", file_name, flags=re.I)
    stem = re.sub(r"_[一-鿿]{2,4}$", "", stem)  # trailing _作者
    stem = stem.replace("_", " ").strip()
    return stem if len(stem) >= 6 else ""


def title_from_cjk_thesis(first_text: str, pages_text: list[str]) -> str:
    """Chinese degree theses: pull the declared 论文题目, else the first real CJK line."""
    blob = "\n".join(pages_text[:3]) if pages_text else (first_text or "")
    blob = unicodedata.normalize("NFKC", blob)
    for pat in (r"论文题\s*目\s*[：:]\s*([^\n]{4,120})",
                r"题\s*目\s*[：:]\s*([^\n]{4,120})",
                r"论文名称\s*[：:]\s*([^\n]{4,120})"):
        m = re.search(pat, blob)
        if m:
            t = re.sub(r"\s+", "", m.group(1)).strip("：: ")
            t = re.sub(r"[-]", "", t)
            if _cjk_title_ok(t):
                return t
    # else: longest plausible CJK line in the first page
    best = ""
    for line in (first_text or "").split("\n"):
        line = re.sub(r"[-]", "", unicodedata.normalize("NFKC", line)).strip()
        line = re.sub(r"\s+", "", line)
        if not _cjk_title_ok(line):
            continue
        if len(line) > len(best):
            best = line
    return best


BAD_TITLE_LINE_RE = re.compile(
    r"contents lists available|sciencedirect|www\.|http|"
    r"journal homepage|elsevier|springer|wiley|taylor & francis|"
    r"open access|creative commons|check for updates|"
    r"downloaded (from|via)|all rights reserved|copyright|^\s*\u00a9|"
    r"^\s*cite this|^\s*read online|metrics & more|article recommendations|"
    r"^\s*\d{4}-\d{3}[\dxX]|^\s*PII|^\s*vol\.?\s*\d|^\s*pp\.?\s*\d|"
    r"^\s*supporting information\s*$|received:|accepted:|published:",
    re.I,
)

# Article-type labels and journal-name lines are often the largest font on p1.
SECTION_LABEL_RE = re.compile(
    r"^\s*(?:full length article|research article|review article|review|article|"
    r"letter|letters|communication|short communication|rapid communication|"
    r"editorial|perspective|mini[- ]review|critical review|note|highlight|"
    r"research paper|original (?:article|research|paper)|opinion|commentary|"
    r"feature|technical note|case study|contents|graphical abstract|"
    r"highlights|abstract|keywords)\s*[:.]?\s*$",
    re.I,
)
JOURNAL_NAME_RE = re.compile(
    r"^\s*(?:journal of |international journal|acs |ieee |nature |science |"
    r"proceedings of|npj )|"
    r"(?:journal|science|research|technology|engineering|letters|reviews|"
    r"chemistry|materials|environment(?:al)?|water|catalysis|membrane)\s*$",
    re.I,
)


def _is_journalish(line: str) -> bool:
    """True for a line that is a journal masthead rather than an article title."""
    l = line.strip()
    if len(l.split()) > 12:
        return False
    if JOURNAL_NAME_RE.search(l):
        # a real title starting with "Journal of..." is vanishingly rare
        return True
    low = l.lower()
    return any(h == low or (h in low and len(low) - len(h) < 12) for h in JOURNAL_HINTS)


def title_from_layout(page) -> str:
    """Largest-font contiguous text block near the top of page 1."""
    try:
        d = page.get_text("dict")
    except Exception:
        return ""
    lines = []
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            txt = "".join(sp.get("text", "") for sp in line.get("spans", []))
            txt = re.sub(r"\s+", " ", txt).strip()
            if not txt:
                continue
            size = max((sp.get("size", 0) for sp in line.get("spans", [])), default=0)
            y = line.get("bbox", [0, 0, 0, 0])[1]
            lines.append({"text": txt, "size": round(size, 1), "y": y})
    if not lines:
        return ""
    cand = [l for l in lines if not BAD_TITLE_LINE_RE.search(l["text"])]
    cand = [l for l in cand if not SECTION_LABEL_RE.match(l["text"])]
    cand = [l for l in cand if not _is_journalish(l["text"])]
    cand = [l for l in cand if len(l["text"]) > 8]
    if not cand:
        return ""
    page_h = page.rect.height or 1000
    # Title is normally in the top 55% of the first page.
    top = [l for l in cand if l["y"] < page_h * 0.55] or cand
    max_size = max(l["size"] for l in top)
    # Merge all lines at (near) the max font size, in reading order.
    picked = [l for l in top if l["size"] >= max_size - 0.6]
    picked.sort(key=lambda l: l["y"])
    # keep only lines contiguous in y with the first one
    out = [picked[0]]
    for l in picked[1:]:
        if l["y"] - out[-1]["y"] < max_size * 3.0:
            out.append(l)
    title = " ".join(l["text"] for l in out)
    title = re.sub(r"\s+", " ", title).strip(" .,;:-\u2013\u2014")
    return title if len(title) > 12 else ""


AUTHORISH_RE = re.compile(
    r"[A-Z][a-z]+\s+[A-Z][a-z]+.*(?:,|\band\b|&|\*|\u2020|\u2021)|"
    r"^[A-Z][a-z]+,\s*[A-Z]\.", re.M,
)


def title_from_text(first_text: str) -> str:
    """Fallback: the first substantial non-boilerplate line before the authors.

    Handles the cases layout analysis misses -- e.g. an Elsevier masthead
    rendered larger than the title, or a bare "Review" type label on top.
    """
    raw = [re.sub(r"\s+", " ", l).strip() for l in (first_text or "")[:3000].split("\n")]
    lines = [l for l in raw if l]
    picked: list[str] = []
    for l in lines[:30]:
        if BAD_TITLE_LINE_RE.search(l) or SECTION_LABEL_RE.match(l) or _is_journalish(l):
            if picked:
                break
            continue
        if len(l) < 12 or len(l) > 300:
            if picked:
                break
            continue
        # stop once we reach the author / affiliation block
        if AUTHORISH_RE.search(l) or re.search(
            r"universit|institut|depart|school of|college|academy|laborator|"
            r"\d{5}|@|e-mail", l, re.I
        ):
            if picked:
                break
            continue
        picked.append(l)
        joined = " ".join(picked)
        # a title is normally one to three printed lines
        if len(joined) > 60 and (joined.endswith(".") or len(picked) >= 3):
            break
    title = re.sub(r"\s+", " ", " ".join(picked)).strip(" .,;:-\u2013\u2014")
    return title if len(title) > 15 else ""


YEAR_RE = re.compile(r"(19[6-9]\d|20[0-4]\d)")


def extract_year(first_text: str, meta: dict) -> tuple[str, str]:
    """Return (year, how). Prefers copyright / published lines over file dates."""
    head = first_text[:6000]
    pats = [
        (r"\u00a9\s*(?:copyright\s*)?(19[6-9]\d|20[0-4]\d)", "copyright"),
        (r"copyright\s*\u00a9?\s*(19[6-9]\d|20[0-4]\d)", "copyright"),
        (r"published\D{0,40}?(19[6-9]\d|20[0-4]\d)", "published"),
        (r"accepted\D{0,40}?(19[6-9]\d|20[0-4]\d)", "accepted"),
        (r"received\D{0,40}?(19[6-9]\d|20[0-4]\d)", "received"),
        (r"\b(?:19[6-9]\d|20[0-4]\d)\b(?=\s*[,;]?\s*(?:vol|no\.|pp\.))", "citation"),
    ]
    for pat, how in pats:
        m = re.search(pat, head, re.I)
        if m:
            return m.group(1), how
    for key in ("creationDate", "modDate"):
        v = str(meta.get(key) or "")
        m = re.search(r"(19[6-9]\d|20[0-4]\d)", v)
        if m:
            return m.group(1), f"meta:{key}"
    yrs = YEAR_RE.findall(head)
    if yrs:
        return max(yrs), "first_page_max"
    return "", "none"


JOURNAL_HINTS = [
    "water research", "science of the total environment", "environmental science",
    "chemosphere", "journal of hazardous materials", "desalination",
    "chemical engineering journal", "bioresource technology", "water science",
    "applied catalysis", "separation and purification technology",
    "journal of membrane science", "journal of water process engineering",
    "environment international", "acs es&t", "environmental pollution",
    "critical reviews", "npj clean water", "water reuse",
]


def extract_journal(first_text: str, meta: dict) -> str:
    for key in ("subject", "journal", "publisher"):
        v = (meta.get(key) or "").strip()
        if v and 3 < len(v) < 120 and not v.lower().startswith("10."):
            if not re.match(r"^(doi|http)", v, re.I):
                return re.sub(r"\s+", " ", v)
    low = first_text[:4000].lower()
    for h in JOURNAL_HINTS:
        if h in low:
            i = low.find(h)
            return first_text[i:i + len(h)].strip()
    m = re.search(
        r"journal homepage:\s*www\.[^\s]+|Available online at[^\n]{0,60}", first_text[:4000], re.I
    )
    if m:
        return re.sub(r"\s+", " ", m.group(0))[:120]
    return ""


NAME_TOKEN = r"[A-Z][A-Za-z'\u2019\-\u00c0-\u017f]+"
AUTHOR_LINE_RE = re.compile(
    rf"^(?:{NAME_TOKEN}|[A-Z]\.){{1,4}}\s+{NAME_TOKEN}"
    r"(?:\s*[a-z0-9*,\u2020\u2021\u00a7\u00b6#]{0,6})?"
    r"(?:\s*,\s*.*)?$"
)


def extract_authors(page, first_text: str, meta: dict, title: str) -> tuple[str, str, str]:
    """Return (authors, first_author, how)."""
    ma = (meta.get("author") or "").strip()
    ma = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", ma))
    if ma and 3 < len(ma) < 500 and not re.search(r"\.(pdf|doc)|acrobat|word|latex", ma, re.I):
        return ma, first_author_of(ma), "metadata"

    # Look at the lines between the title and the abstract on page 1.
    lines = [re.sub(r"\s+", " ", l).strip() for l in (first_text or "").split("\n")]
    lines = [l for l in lines if l]
    ntitle = norm_title(title)
    start = 0
    if ntitle:
        for i, l in enumerate(lines[:40]):
            if norm_title(l) and norm_title(l) in ntitle and len(norm_title(l)) > 15:
                start = i + 1
    window = []
    for l in lines[start:start + 14]:
        if re.match(r"^\s*(abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t|keywords|highlights|introduction|1\.)\b", l, re.I):
            break
        window.append(l)
    for l in window:
        if BAD_TITLE_LINE_RE.search(l):
            continue
        if len(l) < 5 or len(l) > 400:
            continue
        # an author line has commas/ands and capitalised name tokens
        if re.search(r"\b(and|&)\b|,", l) or AUTHOR_LINE_RE.match(l):
            names = re.findall(rf"{NAME_TOKEN}", l)
            if len(names) >= 2 and not re.search(
                r"universit|institut|depart|school|colleg|academy|laborator|center|centre|"
                r"china|usa|india|korea|japan|germany|@|\d{4}",
                l, re.I,
            ):
                return l, first_author_of(l), "layout"
    return "", "", "none"


def first_author_of(authors: str) -> str:
    """Best-effort surname of the first author."""
    if not authors:
        return ""
    a = re.split(r"\s*(?:,|;|\band\b|&)\s*", authors.strip())[0]
    a = re.sub(r"[\d*\u2020\u2021\u00a7\u00b6#]+", "", a).strip()
    a = re.sub(r"\s+", " ", a)
    if not a:
        return ""
    toks = [t for t in a.split() if len(t) > 1 or t.endswith(".")]
    if not toks:
        return ""
    # "Smith J." / "Smith, John" -> Smith ; "John Smith" -> Smith
    if len(toks) >= 2 and re.fullmatch(r"(?:[A-Z]\.?){1,3}", toks[-1]):
        surname = toks[0]
    else:
        surname = toks[-1]
    return re.sub(r"[^A-Za-z\u00c0-\u017f\-']", "", surname).lower()


# ---------------------------------------------------------------- doc typing

SUPPLEMENT_PAT = [
    "supporting information", "supplementary material", "supplementary materials",
    "supplementary information", "supplementary data", "supplemental material",
    "electronic supplementary material", "supplementary appendix",
    "supporting infomation", "si appendix",
]
CORRIGENDUM_PAT = ["corrigendum", "erratum", "retraction notice", "retracted:",
                   "publisher's note", "addendum to"]
EDITORIAL_PAT = ["editorial", "guest editorial", "in this issue", "from the editor",
                 "preface", "book review"]
TOC_PAT = ["table of contents", "contents of this issue", "in this issue:"]
ERRORPAGE_PAT = [
    "sign in", "log in to continue", "access denied", "403 forbidden",
    "404 not found", "page not found", "just a moment", "cloudflare",
    "enable javascript", "captcha", "your institution does not have access",
    "purchase pdf", "get access to the full version", "<html", "<!doctype",
]
REFS_PAT = ["references", "bibliography", "literature cited", "\u53c2\u8003\u6587\u732e",
            "references and notes"]


def classify(first_text: str, all_text: str, page_count: int, title: str,
             text_len: int) -> tuple[str, list[str], dict]:
    """Return (doc_type, signals, markers).

    NOTE on supplements: every modern ACS/Elsevier *main* article carries the
    phrase "Supporting Information" in its header widget ("Cite This | ACCESS |
    Metrics & More | Article Recommendations | *si Supporting Information") and
    in its "ASSOCIATED CONTENT" section.  Matching that phrase anywhere is
    therefore useless -- it fires on ~25 real papers in this corpus.  A genuine
    SI file instead *opens* with the phrase and has no abstract/references of
    its own, which is what is tested below.
    """
    sig = []
    head = re.sub(r"\s+", " ", (first_text or "")[:4000]).lower()
    t = (title or "").lower()
    probe = head + " " + t

    m = markers(first_text, all_text, page_count)

    if text_len < 200:
        sig.append("almost_no_text")
    for p in ERRORPAGE_PAT:
        if p in probe:
            sig.append(f"errorpage:{p}")
    for p in SUPPLEMENT_PAT:
        if p in probe:
            sig.append(f"supplement_phrase:{p}")
    for p in CORRIGENDUM_PAT:
        if p in probe:
            sig.append(f"corrigendum:{p}")
    for p in TOC_PAT:
        if p in probe:
            sig.append(f"toc:{p}")
    for p in EDITORIAL_PAT:
        if re.search(rf"(?:^|\W){re.escape(p)}(?:\W|$)", probe):
            sig.append(f"editorial:{p}")
    if m["si_leads"]:
        sig.append("si_leads_page1")
    if m["si_fig_ratio"] >= 0.75 and m["si_fig_hits"] >= 5:
        sig.append(f"si_figure_dominant:{m['si_fig_hits']}/{m['si_fig_hits'] + m['main_fig_hits']}")

    err = [s for s in sig if s.startswith("errorpage")]
    cor = [s for s in sig if s.startswith("corrigendum")]
    toc = [s for s in sig if s.startswith("toc")]
    edi = [s for s in sig if s.startswith("editorial")]

    # ---- ordered strongest-signal-first
    if err and (text_len < 6000 or page_count <= 2):
        return "error_or_login_page", sig, m
    if text_len < 200:
        return "no_extractable_text", sig, m

    # A real SI: leads with the SI banner, and lacks its own abstract.
    # NB: si_fig dominance alone is NOT sufficient -- a PNAS research article
    # can cite 46x "Fig. S#" and zero "Fig. #" and still be the main paper.
    if m["si_leads"] and not m["has_abstract"]:
        return "supplement", sig, m
    if m["si_leads"] and not m["has_refs"]:
        return "supplement_suspect", sig, m

    if cor and page_count <= 4 and not m["has_abstract"]:
        return "corrigendum_or_erratum", sig, m
    if toc and page_count <= 6 and not m["has_abstract"]:
        return "table_of_contents", sig, m
    # "Editorial" printed as the article-type label above the title.
    if m["editorial_label"] and page_count <= 8:
        return "editorial", sig, m
    if edi and page_count <= 4 and not m["has_refs"]:
        return "editorial", sig, m
    if page_count <= 2 and m["has_abstract"] and not m["has_refs"]:
        return "abstract_only", sig, m
    return "article", sig, m


SI_LEAD_RE = re.compile(
    r"^[\s\W]{0,40}(supporting\s+information|supplementary\s+"
    r"(?:material|materials|information|data|appendix)|supplemental\s+material|"
    r"electronic\s+supplementary\s+material|si\s+appendix)\b",
    re.I,
)
SI_LEAD_FOR_RE = re.compile(
    r"^.{0,80}?(supporting\s+information|supplementary\s+"
    r"(?:material|materials|information|data))\s+(?:for|to|of)\b",
    re.I | re.S,
)
ABSTRACT_RE = re.compile(
    r"\b(a\s?b\s?s\s?t\s?r\s?a\s?c\s?t\b|摘\s*要)", re.I
)


def markers(first_text: str, all_text: str, page_count: int) -> dict:
    """Structural markers used both for typing and for best-version scoring."""
    ft = first_text or ""
    at = all_text or ""
    head_raw = re.sub(r"[ \t]+", " ", ft[:400])
    head = re.sub(r"\s+", " ", ft[:6000])
    low = head.lower()
    body = at.lower()

    si_leads = bool(SI_LEAD_RE.match(head_raw.strip())) or bool(SI_LEAD_FOR_RE.match(head_raw))
    si_fig = len(re.findall(r"\b(?:figure|fig\.?|table|scheme|eq\.?)\s*s\s?\d", body, re.I))
    main_fig = len(re.findall(r"\b(?:figure|fig\.?|table|scheme)\s*\d", body, re.I))
    # "Editorial" as the article-type label printed above the title
    ed_label = bool(re.search(
        r"(?:^|\n|\s{2,})(?:guest\s+)?editorial(?:\s*[:\-]|\s*\n|\s{2,})",
        ft[:1200], re.I,
    )) or bool(re.match(r"^\s*(?:guest\s+)?editorial\b", ft.strip(), re.I))
    return {
        "si_leads": si_leads,
        "si_fig_hits": si_fig,
        "main_fig_hits": max(0, main_fig - si_fig),
        "si_fig_ratio": round(si_fig / max(1, si_fig + max(0, main_fig - si_fig)), 3),
        "has_abstract": bool(ABSTRACT_RE.search(low[:3500])),
        "has_assoc_content": ("associated content" in body[:400000]
                              or "author information" in body[:400000]),
        "has_refs": _has_refs(at),
        "editorial_label": ed_label,
        "text_per_page": round(len(at) / max(1, page_count), 1),
    }


def _has_refs(all_text: str) -> bool:
    tail = (all_text or "")[-int(max(2000, len(all_text or "") * 0.35)):].lower()
    return any(p in tail for p in REFS_PAT)


AM_PAT = [
    "accepted manuscript", "author manuscript", "this is a pdf file of an unedited",
    "postprint", "this is the peer reviewed version", "accepted version",
    "author accepted manuscript", "preprint submitted to",
]
PUB_PAT = [
    "contents lists available at sciencedirect", "pubs.acs.org",
    "journal homepage:", "link.springer.com", "onlinelibrary.wiley.com",
    "iwaponline.com", "rsc.li/", "pubs.rsc.org", "mdpi.com",
    "downloaded via", "acs.org/doi", "frontiersin.org", "tandfonline.com",
]


def version_type(first_text: str, all_text: str) -> str:
    head = re.sub(r"\s+", " ", (first_text or "")[:5000]).lower()
    body = re.sub(r"\s+", " ", (all_text or "")[:20000]).lower()
    if any(p in head for p in AM_PAT):
        return "accepted_manuscript"
    if any(p in head for p in PUB_PAT):
        return "publisher_version"
    if any(p in body for p in AM_PAT):
        return "accepted_manuscript"
    if any(p in body for p in PUB_PAT):
        return "publisher_version"
    return "unknown"


# ---------------------------------------------------------------- main scan

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shingle_hashes(nt: str, k: int = 48, step: int = 24, cap: int = 900) -> set[int]:
    """Sample k-gram hashes of the normalised text for Jaccard near-dup scoring.

    crc32 (not builtin hash()) so the values are stable across processes/runs.
    """
    if len(nt) < k:
        return set()
    out = set()
    stride = max(step, (len(nt) - k) // cap + 1)
    for i in range(0, len(nt) - k + 1, stride):
        out.add(zlib.crc32(nt[i:i + k].encode()))
    return out


def extract_abstract(first_text: str) -> str:
    """Text between the abstract heading and the next section heading."""
    ft = re.sub(r"\s+", " ", first_text or "")
    m = re.search(r"\b(?:a\s?b\s?s\s?t\s?r\s?a\s?c\s?t|摘\s*要)\b[:.\s\-]*", ft, re.I)
    if not m:
        return ft[:1600]
    tail = ft[m.end():m.end() + 2600]
    stop = re.search(
        r"\b(keywords?|key\s?words|1\.?\s+introduction|introduction\b|"
        r"graphical abstract|highlights|关键词|引\s*言)\b", tail, re.I
    )
    return tail[: stop.start()] if stop else tail


def scan_one(path: Path, source_group: str) -> dict:
    rec: dict = {
        "path": str(path),
        "rel_path": str(path.relative_to(ROOT)),
        "file_name": path.name,
        "source_group": source_group,
        "file_size": path.stat().st_size,
        "sha256": "",
        "open_ok": False,
        "open_error": "",
        "page_count": 0,
        "is_encrypted": False,
        "text_length": 0,
        "doi": "",
        "doi_source": "",
        "doi_candidates": [],
        "title": "",
        "title_source": "",
        "authors": "",
        "first_author": "",
        "author_source": "",
        "year": "",
        "year_source": "",
        "journal": "",
        "doc_type": "unknown",
        "signals": [],
        "markers": {},
        "version_type": "unknown",
        "has_references": False,
        "fingerprint": "",
        "norm_head": "",
        "norm_all_len": 0,
        "shingles": [],
        "abstract_norm": "",
        "head_text": "",
        "images_only": False,
    }
    try:
        rec["sha256"] = sha256_of(path)
    except Exception as e:  # unreadable on disk
        rec["open_error"] = f"sha256: {e!r}"
        return rec

    doc = None
    try:
        doc = fitz.open(str(path))
        rec["is_encrypted"] = bool(doc.is_encrypted)
        if doc.is_encrypted:
            doc.authenticate("")
        rec["page_count"] = doc.page_count
        meta = {k: (v or "") for k, v in (doc.metadata or {}).items()}

        pages_text: list[str] = []
        for i in range(doc.page_count):
            try:
                pages_text.append(doc.load_page(i).get_text("text") or "")
            except Exception:
                pages_text.append("")
        all_text = "\n".join(pages_text)
        first_text = pages_text[0] if pages_text else ""
        rec["text_length"] = len(all_text)
        rec["open_ok"] = True

        # ---- DOI: metadata + XMP first, then page 1, then last pages, then body
        meta_blob = " ".join(str(v) for v in meta.values())
        try:
            meta_blob += " " + (doc.get_xml_metadata() or "")
        except Exception:
            pass
        chain = [
            ("metadata", meta_blob),
            ("page1", first_text),
            ("last_pages", "\n".join(pages_text[-2:])),
            ("body", all_text[:400000]),
        ]
        seen: list[str] = []
        for src, blob in chain:
            found = find_dois(blob)
            for d in found:
                if d not in seen:
                    seen.append(d)
            if found and not rec["doi"]:
                rec["doi"] = found[0]
                rec["doi_source"] = src
        rec["doi_candidates"] = seen[:12]

        # ---- title
        mt = clean_meta_title(meta.get("title"))
        lt = title_from_layout(doc.load_page(0)) if doc.page_count else ""
        if PII_TITLE_RE.match(lt or "") or not re.search(r"[A-Za-z一-鿿]{3}", lt or ""):
            lt = ""
        if mt and lt and norm_title(mt)[:40] == norm_title(lt)[:40]:
            rec["title"], rec["title_source"] = mt, "metadata+layout"
        elif mt and len(mt) >= 20:
            rec["title"], rec["title_source"] = mt, "metadata"
        elif lt:
            rec["title"], rec["title_source"] = lt, "layout"
        elif mt:
            rec["title"], rec["title_source"] = mt, "metadata_short"
        else:
            rec["title"], rec["title_source"] = "", "none"

        # CJK theses: layout/metadata titles are unreliable, prefer the declared 题目
        cjk_share = len(CJK_RE.findall(first_text[:3000])) / max(1, len(first_text[:3000]))
        if cjk_share > 0.12:
            ct = title_from_cjk_thesis(first_text, pages_text)
            cur = rec["title"] or ""
            cur_bad = (
                not cur
                or len(CJK_RE.findall(cur)) < 4
                or THESIS_BOILER_RE.search(cur)
                or re.search(r"[-]", cur)  # PUA glyphs from broken CNKI fonts
            )
            if ct and (cur_bad or len(ct) > len(cur) * 1.4):
                rec["title"], rec["title_source"] = ct, "cjk_thesis_body"

        rec["year"], rec["year_source"] = extract_year(first_text, meta)
        rec["journal"] = extract_journal(first_text, meta)
        if doc.page_count:
            a, fa, how = extract_authors(doc.load_page(0), first_text, meta, rec["title"])
            rec["authors"], rec["first_author"], rec["author_source"] = a, fa, how

        rec["doc_type"], rec["signals"], mk = classify(
            first_text, all_text, doc.page_count, rec["title"], rec["text_length"]
        )
        rec["markers"] = mk
        rec["version_type"] = version_type(first_text, all_text)
        rec["has_references"] = mk["has_refs"]

        # fingerprint over the real body text (skip page 1 furniture when possible)
        body_for_fp = "".join(pages_text[1:4]) if len(pages_text) > 2 else all_text
        nfp = norm_text(body_for_fp)
        if len(nfp) < 400:
            nfp = norm_text(all_text)
        rec["fingerprint"] = hashlib.sha256(nfp[:4000].encode()).hexdigest() if nfp else ""
        rec["norm_head"] = nfp[:2500]
        rec["norm_all_len"] = len(norm_text(all_text))
        # shingle set for near-duplicate Jaccard, taken over the whole document
        rec["shingles"] = sorted(shingle_hashes(norm_text(all_text)))
        rec["head_text"] = re.sub(r"\s+", " ", first_text)[:1500]
        rec["abstract_norm"] = norm_text(extract_abstract(first_text))[:900]

        if rec["text_length"] < 500 and doc.page_count >= 3:
            rec["images_only"] = True
    except Exception as e:
        rec["open_error"] = f"{type(e).__name__}: {e}"
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
    return rec


def main() -> None:
    files: list[tuple[Path, str]] = []
    seen_paths: set[str] = set()
    for group, d, recurse in SOURCES:
        if not d.exists():
            print(f"[MISSING] {group}: {d}", file=sys.stderr)
            continue
        it = d.rglob("*.pdf") if recurse else d.glob("*.pdf")
        for p in sorted(it):
            key = str(p).lower()
            if key in seen_paths:
                continue
            # a non-recursive group must not swallow a nested group's files
            seen_paths.add(key)
            files.append((p, group))

    print(f"Discovered {len(files)} PDF files across {len(SOURCES)} source groups.")
    records = []
    for i, (p, g) in enumerate(files, 1):
        rec = scan_one(p, g)
        records.append(rec)
        if i % 20 == 0 or i == len(files):
            print(f"  scanned {i}/{len(files)}", flush=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nWrote cache: {CACHE}")
    ok = sum(1 for r in records if r["open_ok"])
    print(f"open_ok={ok}  failed={len(records) - ok}")
    from collections import Counter
    print("doc_type:", dict(Counter(r["doc_type"] for r in records)))
    print("with_doi:", sum(1 for r in records if r["doi"]))
    print("with_title:", sum(1 for r in records if r["title"]))
    print("with_first_author:", sum(1 for r in records if r["first_author"]))
    print("with_year:", sum(1 for r in records if r["year"]))


if __name__ == "__main__":
    main()
