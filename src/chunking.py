"""Chunking: legacy per-page fixed chunker + section-aware page-traceable chunker.

Two modes (selectable via settings.chunking_mode / CHUNKING_MODE, or the
``chunking_mode`` argument to :func:`build_chunks`):

- ``fixed``          -- legacy behavior, unchanged: each page is chunked
                        independently by character window. Kept for Eval V2
                        fixed-vs-section_aware comparison.
- ``section_aware``  -- detects common research section headings (named +
                        numbered) and chunks *within* each section, tracking the
                        source pages per chunk. If no section heading can be
                        found, it falls back to page-aware fixed chunking.

Chunk contract (section_aware):
    {"id": chunk_id, "text": str, "metadata": {
        "paper_id", "title", "doi", "section", "page_start", "page_end",
        "chunk_index", "chunk_id", "source", "chunking_mode"}}
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from src.config import settings

# ---------------------------------------------------------------- index eligibility

# Sections that are kept in the raw chunk manifest but EXCLUDED from the
# Dense/Sparse retrieval index (low signal + title/author/DOI noise).
EXCLUDED_INDEX_SECTIONS = {"references", "acknowledgments"}


def is_indexable(section: str | None) -> bool:
    """True unless ``section`` is an explicitly excluded one.

    Unknown / empty sections default to indexable (never filter aggressively).
    """
    return (section or "").strip().lower() not in EXCLUDED_INDEX_SECTIONS


# ---------------------------------------------------------------- legacy fixed

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple character-based chunking (legacy)."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def build_chunks_fixed(page_records: List[Dict]) -> List[Dict]:
    """Legacy per-page fixed chunker (preserved verbatim behavior)."""
    all_chunks: List[Dict] = []

    for rec in page_records:
        chunks = chunk_text(
            rec["text"],
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        for idx, chunk in enumerate(chunks):
            chunk_id = f'{rec["paper_id"]}_p{rec["page"]}_c{idx}'
            all_chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "paper_id": rec["paper_id"],
                        "source": rec["source"],
                        "page": rec["page"],
                        "chunk_index": idx,
                    },
                }
            )

    return all_chunks


# ---------------------------------------------------------------- section-aware

NAMED_SECTIONS: Dict[str, set] = {
    "abstract": {"abstract", "summary abstract"},
    "introduction": {"introduction", "intro", "introduction and background",
                     "introduction and objectives", "introduction and scope"},
    "background": {"background", "theoretical background", "research background",
                   "background and significance"},
    "methods": {"methods", "method", "materials and methods", "material and methods",
                "materials and method", "methodology", "experimental",
                "experimental section", "experimental methods", "experimental procedures",
                "materials and experimental", "theoretical methods", "computational methods",
                "computational details", "data and methods", "experimental setup"},
    "results": {"results", "result", "results and discussion", "results and discussions",
                "results & discussion", "results and analysis"},
    "discussion": {"discussion", "discussions", "discussion and conclusions",
                   "discussion and conclusion", "discussion and analysis"},
    "conclusion": {"conclusion", "conclusions", "concluding remarks",
                   "summary and conclusions", "summary", "conclusions and outlook",
                   "conclusions and future work"},
    "references": {"references", "reference", "bibliography", "literature cited",
                   "reference list", "references and notes"},
    "supporting_information": {"supporting information", "supporting information available",
                               "supplementary material", "supplementary materials",
                               "supplementary information", "supplementary data",
                               "electronic supplementary material", "si appendix",
                               "supplementary note", "supplementary notes"},
    "acknowledgments": {"acknowledgments", "acknowledgements", "acknowledgment",
                        "acknowledgement", "acknowledgments and funding"},
}

# "1", "1.", "1.1", "2.1.1" + title (title must begin with a letter/CJK char).
NUMBERED_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,2})[\.\)]?\s+([A-Za-z\u4e00-\u9fff][^\n]*)$")

_STRIP_TRAIL = " .,:;-–—"


def _canonical_section(name: str) -> Optional[str]:
    n = re.sub(r"\s+", " ", (name or "").lower().strip(_STRIP_TRAIL)).strip()
    if not n:
        return None
    for canon, aliases in NAMED_SECTIONS.items():
        if n in aliases:
            return canon
    return None


# Sentence verbs / connectors that mark a numbered LIST ITEM or body fragment
# rather than a short section title. Articles and prepositions are deliberately
# NOT included so legit subsections like "The effect of pH" still match.
_SENTENCE_WORDS_RE = re.compile(
    r"\b(must|should|can|could|will|would|may|might|shall|was|were|are|is|been|"
    r"being|am|do|does|did|has|have|had|that|which|these|those|this|several|"
    r"many|such|both|each|however|therefore|thus|although|because|while|when|"
    r"where|whether|so that|such as|due to)\b",
    re.I,
)
# Math / unit artifacts never appear in a section title.
_MATH_RE = re.compile(r"[=×±≤≥·−→←↔↕•†‡°]|m−1|cm−1|s−1|m−3|d−1")
# Transition / connector words never open a section title.
_SENTENCE_START_RE = re.compile(
    r"^(?:only|also|so|then|next|finally|lastly|moreover|furthermore|additionally|"
    r"importantly|notably|interestingly|therefore|thus|hence|however|nevertheless|"
    r"nonetheless|although|though|whereas|meanwhile|but|and|or)\b",
    re.I,
)


def _numbered_title_ok(title: str) -> bool:
    """Cheap guards so reference lines / numbered list items are not headings."""
    if not (3 <= len(title) <= 50):
        return False
    words = title.split()
    if not (1 <= len(words) <= 6):
        return False
    if re.search(r"et\s+al\.?", title, re.I):
        return False
    if re.search(r"\b(?:19|20)\d{2}\b", title):  # bare year (references/dates)
        return False
    if ";" in title:  # reference "year;vol:pages" separators
        return False
    if re.search(r"\bdoi\s*[:.]?\s*10\.|https?://|www\.", title, re.I):
        return False
    if title.count(",") >= 2:
        return False
    if re.search(r"\d+\s*[-–—]\s*\d+\s*[.]?$", title):
        return False
    # internal "word. word" sentence break -> body text, not a title
    if re.search(r"\w\.\s+\w", title):
        return False
    # reference author pattern "Smith, J." at the start of a numbered line
    if re.match(r"^[A-Z][A-Za-z'’-]+,\s*[A-Z]\.", title):
        return False
    if _MATH_RE.search(title):
        return False
    if _SENTENCE_WORDS_RE.search(title):
        return False
    if _SENTENCE_START_RE.match(title):
        return False
    # Latin titles begin with a capital letter; rejects lowercase sentence
    # fragments such as "3.14 is just a number" while CJK titles pass through.
    first = re.match(r"^\s*([A-Za-z])", title)
    if first and not first.group(1).isupper():
        return False
    return True


def match_section_heading(line: str, allow_numbered: bool = True) -> Optional[str]:
    """Return a canonical section label if ``line`` is a section heading, else None.

    ``allow_numbered`` disables numbered-heading detection (used once we are
    inside References, whose numeric entries "1. Author..." are not headings).
    """
    raw = (line or "").strip()
    if not raw:
        return None

    c = _canonical_section(raw)
    if c:
        return c

    if not allow_numbered:
        return None

    m = NUMBERED_RE.match(raw)
    if m:
        title = m.group(2).strip()
        if _numbered_title_ok(title):
            c = _canonical_section(title)
            if c:
                return c
            # generic numbered subsection: require a short multi-word noun phrase
            t = re.sub(r"\s+", " ", title.lower()).strip(_STRIP_TRAIL).strip()
            if t and len(t.split()) >= 2:
                return t[:80]
    return None


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return s or "sec"


def clean_page_text(text: str) -> str:
    """Whitespace-normalise a page and drop blank / pure-page-number lines."""
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in (text or "").split("\n")]
    return "\n".join(ln for ln in lines if ln and not re.fullmatch(r"\d{1,4}", ln))


def _clean_pages(pages: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """Clean every page and remove short running header/footer lines.

    A line is treated as a running header/footer when it appears on >= 60% of
    pages (at least 3) and is shorter than 120 chars.
    """
    cleaned: List[Tuple[int, str]] = []
    for p, t in pages:
        cleaned.append((p, clean_page_text(t)))

    n = len(cleaned)
    threshold = max(3, int(n * 0.6 + 0.999)) if n else 0
    page_count: Counter = Counter()
    for _p, t in cleaned:
        for ln in set(t.split("\n")):
            page_count[ln] += 1
    repeated = {ln for ln, c in page_count.items() if c >= threshold and len(ln) < 120}

    out: List[Tuple[int, str]] = []
    for p, t in cleaned:
        kept = [ln for ln in t.split("\n") if ln not in repeated]
        out.append((p, "\n".join(kept)))
    return out


def _chunk_pieces(pieces: List[Tuple[int, str]], chunk_size: int, overlap: int,
                  min_chunk: int) -> List[Dict]:
    """Chunk an ordered list of (page, text) pieces with page provenance.

    Overlap stays inside this piece-run (i.e. inside one section), and each
    returned chunk records the first/last source page of its characters.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chars: List[str] = []
    pages: List[int] = []
    last_page = 0
    for p, t in pieces:
        t = (t or "").strip()
        if not t:
            continue
        if chars:
            chars.append("\n")
            pages.append(last_page)
        for ch in t:
            chars.append(ch)
            pages.append(p)
        last_page = p

    total = len(chars)
    if total == 0:
        return []

    out: List[Dict] = []
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        text = "".join(chars[start:end]).strip()
        if len(text) >= min_chunk:
            out.append({
                "text": text,
                "page_start": pages[start],
                "page_end": pages[end - 1],
            })
        if end >= total:
            break
        start = end - overlap
    return out


def _detect_runs(pages: List[Tuple[int, str]]) -> List[Tuple[str, List[Tuple[int, str]]]]:
    """Split cleaned pages into ordered (section, [(page, text)]) runs.

    Text before the first recognised heading is labelled ``front_matter``.
    """
    segments: List[Tuple[str, int, str]] = []  # (section, page, text)
    current: Optional[str] = None  # active section persists across pages
    in_references = False
    for page, text in pages:
        buf: List[str] = []
        for ln in text.split("\n"):
            heading = match_section_heading(ln, allow_numbered=not in_references)
            if heading == "references":
                in_references = True
            if heading is not None:
                if buf:
                    segments.append((current or "front_matter", page, " ".join(buf)))
                current = heading
                buf = []
            else:
                buf.append(ln)
        if buf:
            segments.append((current or "front_matter", page, " ".join(buf)))

    runs: List[Tuple[str, List[Tuple[int, str]]]] = []
    for section, page, text in segments:
        if runs and runs[-1][0] == section:
            runs[-1][1].append((page, text))
        else:
            runs.append((section, [(page, text)]))
    return runs


def build_chunks_section_aware(page_records: List[Dict],
                               paper_meta: Optional[Dict] = None) -> List[Dict]:
    """Section-aware + page-traceable chunking over page records.

    ``paper_meta`` maps paper_id -> {"title": ..., "doi": ...}. Missing keys
    default to "". DOI may be empty; that is not an error.
    """
    meta = paper_meta or {}

    # group pages per paper, preserving first-seen order
    by_paper: Dict[str, List[Dict]] = {}
    order: List[str] = []
    for rec in page_records:
        pid = rec["paper_id"]
        if pid not in by_paper:
            by_paper[pid] = []
            order.append(pid)
        by_paper[pid].append(rec)

    all_chunks: List[Dict] = []
    for pid in order:
        recs = sorted(by_paper[pid], key=lambda r: r["page"])
        pages = _clean_pages([(r["page"], r["text"]) for r in recs])
        info = meta.get(pid) or meta.get(recs[0]["source"]) or {}

        runs = _detect_runs(pages)
        has_structure = any(section != "front_matter" for section, _ in runs)

        title = info.get("title", "") or ""
        doi = info.get("doi", "") or ""

        chunk_index = 0
        if not has_structure:
            # fallback: page-aware fixed chunking over the whole cleaned doc
            mode = "fallback_fixed"
            pieces = pages
            for ch in _chunk_pieces(pieces, settings.chunk_size,
                                    settings.chunk_overlap, settings.section_min_chunk):
                cid = f"{pid}__body__p{ch['page_start']}-{ch['page_end']}__i{chunk_index}"
                all_chunks.append(_make_chunk(
                    pid, title, doi, "body", ch["text"], ch["page_start"],
                    ch["page_end"], chunk_index, cid, recs[0]["source"], mode))
                chunk_index += 1
            continue

        mode = "section_aware"
        for section, pieces in runs:
            for ch in _chunk_pieces(pieces, settings.chunk_size,
                                    settings.chunk_overlap, settings.section_min_chunk):
                cid = f"{pid}__{_slug(section)}__p{ch['page_start']}-{ch['page_end']}__i{chunk_index}"
                all_chunks.append(_make_chunk(
                    pid, title, doi, section, ch["text"], ch["page_start"],
                    ch["page_end"], chunk_index, cid, recs[0]["source"], mode))
                chunk_index += 1

    return all_chunks


def _make_chunk(paper_id: str, title: str, doi: str, section: str, text: str,
                page_start: int, page_end: int, chunk_index: int, chunk_id: str,
                source: str, mode: str) -> Dict:
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "paper_id": paper_id,
            "title": title,
            "doi": doi,
            "section": section,
            "page_start": page_start,
            "page_end": page_end,
            "chunk_index": chunk_index,
            "chunk_id": chunk_id,
            "source": source,
            "chunking_mode": mode,
        },
    }


# ---------------------------------------------------------------- dispatcher

def build_chunks(page_records: List[Dict], chunking_mode: Optional[str] = None,
                 paper_meta: Optional[Dict] = None) -> List[Dict]:
    """Build chunks using the configured mode ("fixed" or "section_aware")."""
    mode = chunking_mode or settings.chunking_mode
    if mode == "section_aware":
        return build_chunks_section_aware(page_records, paper_meta=paper_meta)
    return build_chunks_fixed(page_records)
