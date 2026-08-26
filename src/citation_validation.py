"""Deterministic post-generation citation validation; never calls an LLM."""
from __future__ import annotations

import re
from typing import Any


_CITATION_RE = re.compile(r"\[S(\d+)\]", flags=re.IGNORECASE)
_PAGE_RE = re.compile(r"(?:第\s*(\d+)\s*页|\bpage\s*(\d+)\b)", flags=re.IGNORECASE)
_EVIDENCE_CLAIM_RE = re.compile(r"(文献证明|文献显示|研究证明|研究表明|根据文献|证据表明)")
_BIBLIOGRAPHIC_RE = re.compile(
    r"(?:\b[A-Z][A-Za-z-]+\s+et\s+al\.?|[\u4e00-\u9fff]{2,4}等[（(]\d{4}[）)]|《[^》]{2,80}》)"
)


def _sentences(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[。！？!?])|(?<=\n)", text) if part]


def validate_citations(
    answer: str,
    contexts: list[dict[str, Any]],
    *,
    generation_truncated: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Validate and conservatively correct only mappings that are deterministic."""
    allowed_ids = set(range(1, len(contexts) + 1))
    cited_ids = [int(value) for value in _CITATION_RE.findall(answer)]
    invalid_ids = sorted({value for value in cited_ids if value not in allowed_ids})
    corrected = answer
    removed_claims = 0
    unsupported_pages: list[int] = []
    warnings: list[str] = []

    if invalid_ids:
        invalid_set = set(invalid_ids)
        corrected = _CITATION_RE.sub(
            lambda match: "" if int(match.group(1)) in invalid_set else match.group(0).upper(),
            corrected,
        )
        warnings.append("已移除无法映射到检索片段的越界引用。")

    valid_pages = {
        int((context.get("metadata") or {}).get("page"))
        for context in contexts
        if str((context.get("metadata") or {}).get("page", "")).isdigit()
    }
    for match in _PAGE_RE.finditer(corrected):
        page = int(match.group(1) or match.group(2))
        if page not in valid_pages:
            unsupported_pages.append(page)
    if unsupported_pages:
        unsupported_set = set(unsupported_pages)

        def replace_page(match: re.Match[str]) -> str:
            page = int(match.group(1) or match.group(2))
            return "[页码无法核验]" if page in unsupported_set else match.group(0)

        corrected = _PAGE_RE.sub(replace_page, corrected)
        warnings.append("已标记无法映射到本次来源元数据的页码。")

    cleaned_parts: list[str] = []
    for sentence in _sentences(corrected):
        if _EVIDENCE_CLAIM_RE.search(sentence) and not _CITATION_RE.search(sentence):
            removed_claims += 1
            continue
        cleaned_parts.append(sentence)
    if removed_claims:
        corrected = "".join(cleaned_parts).strip()
        warnings.append("已移除声称有文献支持但没有引用的句子。")

    possible_bibliographic_claims = sorted(set(_BIBLIOGRAPHIC_RE.findall(corrected)))
    if possible_bibliographic_claims:
        warnings.append("检测到元数据未必能够核验的作者或标题格式。")

    used_ids = sorted({int(value) for value in _CITATION_RE.findall(corrected) if int(value) in allowed_ids})
    unused_ids = sorted(allowed_ids - set(used_ids))
    no_citation_for_answer = bool(contexts and not used_ids and "没有足够证据" not in corrected)
    if no_citation_for_answer:
        warnings.append("回答没有任何可映射引用。")
    if generation_truncated:
        warnings.append("模型输出达到长度上限，回答可能不完整。")

    hard_failure = bool(no_citation_for_answer or possible_bibliographic_claims or generation_truncated)
    if hard_failure:
        status = "failed"
    elif invalid_ids or unsupported_pages or removed_claims:
        status = "corrected"
    else:
        status = "passed"

    if warnings and status != "passed":
        corrected = corrected.rstrip() + "\n\n> 引用校验：" + " ".join(warnings)

    changed = corrected != answer

    return corrected, {
        "status": status,
        "allowed_source_ids": [f"S{value}" for value in sorted(allowed_ids)],
        "used_source_ids": [f"S{value}" for value in used_ids],
        "unused_source_ids": [f"S{value}" for value in unused_ids],
        "invalid_source_ids": [f"S{value}" for value in invalid_ids],
        "unsupported_pages": sorted(set(unsupported_pages)),
        "claims_removed_without_citation": removed_claims,
        "possible_bibliographic_claim_count": len(possible_bibliographic_claims),
        "generation_truncated": generation_truncated,
        "corrected": changed,
        "warnings": warnings,
    }
