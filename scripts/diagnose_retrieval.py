"""Per-case DEV retrieval diagnosis + References confound check + bootstrap.

For the 15 evidence cases in data/eval_v2/dev.jsonl, retrieves top-10 from the 4
configs and reports rank / paper-hit / page-hit / evidence-span-hit per case,
classifies each case into failure modes A-F, checks whether fixed_hybrid gains
come from References pages, and computes bootstrap CIs + case-level win/loss.

Writes data/eval_v2/dev_retrieval_badcases.csv and dev_retrieval_diagnosis.md.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_eval_v2 import CONFIGS, load_dev, raw_retrieve, set_index  # noqa: E402
from validate_eval_v2 import norm  # noqa: E402

EVAL = ROOT / "data" / "eval_v2"
CHUNKS = ROOT / "data" / "processed" / "section_chunks.jsonl"


def build_section_map():
    """(paper_id, page) -> set of section-aware section names on that page."""
    m = {}
    for l in CHUNKS.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        for p in range(r["page_start"], r["page_end"] + 1):
            m.setdefault((r["paper_id"], p), set()).add(r["section"])
    return m


def hit_details(hits, case):
    gold_pid = case["gold_paper_id"]
    gold_page = case["gold_page_start"]
    gold_ev = norm(case["gold_evidence_text"])
    rank = None
    for i, h in enumerate(hits, 1):
        m = h.get("metadata", {})
        if m.get("paper_id") == gold_pid:
            rank = rank or i
    page_hit = any(
        h.get("metadata", {}).get("paper_id") == gold_pid
        and (h["metadata"].get("page_start") or 0) <= gold_page <= (h["metadata"].get("page_end") or gold_page)
        for h in hits)
    span_hit = any(gold_ev and gold_ev in norm(h.get("text", "")) for h in hits)
    return rank, page_hit, span_hit


def references_share(hits, gold_pid, section_map):
    """For hits on the gold paper, what fraction land on a references page."""
    gold_hits = [h for h in hits if h.get("metadata", {}).get("paper_id") == gold_pid]
    if not gold_hits:
        return 0.0, 0
    ref = 0
    for h in gold_hits:
        m = h.get("metadata", {})
        ps = m.get("page_start") or m.get("page") or 0
        pe = m.get("page_end") or ps
        pages = list(range(ps, pe + 1)) or [ps]
        if any("references" in section_map.get((gold_pid, p), set()) for p in pages):
            ref += 1
    return ref / len(gold_hits), ref


def main():
    section_map = build_section_map()
    cases = [c for c in load_dev() if c.get("gold_paper_id")]
    rows = []
    results = {cfg: {"span": [], "page": [], "paper": []} for cfg, _, _ in CONFIGS}

    for c in cases:
        row = {"case_id": c["case_id"], "query": c["query"][:60],
               "gold_paper": c["gold_paper_id"][:28],
               "gold_page": c["gold_page_start"], "gold_section": c["gold_section"]}
        per = {}
        for cfg, index, mode in CONFIGS:
            set_index(index, mode)
            hits = raw_retrieve(c["query"], 10, mode)
            rank, page_hit, span_hit = hit_details(hits, c)
            per[cfg] = (rank, page_hit, span_hit)
            results[cfg]["paper"].append(1 if rank else 0)
            results[cfg]["page"].append(1 if page_hit else 0)
            results[cfg]["span"].append(1 if span_hit else 0)
            row[f"{cfg}_rank"] = rank if rank else 0
            row[f"{cfg}_page_hit"] = int(page_hit)
            row[f"{cfg}_span_hit"] = int(span_hit)

        # references confound on fixed_hybrid
        set_index("fixed_270_gpu", "hybrid_dense_sparse")
        fh = raw_retrieve(c["query"], 10, "hybrid_dense_sparse")
        row["fixed_hybrid_ref_share"] = round(references_share(fh, c["gold_paper_id"], section_map)[0], 2)

        # classify (based on fixed_hybrid, the candidate baseline)
        fd = per["fixed_dense"]; fh3 = per["fixed_hybrid"]; sh = per["section_hybrid"]
        cat = ""
        if fd[2] == 0 and fh3[2] == 1:
            cat = "A_dense_miss_hybrid_rescue"
        elif fh3[2] == 1 and sh[2] == 0:
            cat = "B_fixed_hit_section_miss"
        elif sh[2] == 1 and fh3[2] == 0:
            cat = "C_section_hit_fixed_miss"
        elif fh3[0] and not fh3[1]:
            cat = "D_paper_hit_wrong_page"
        elif not fh3[0]:
            cat = "E_paper_page_miss"
        elif fh3[1] and not fh3[2]:
            cat = "F_page_hit_but_span_miss"
        else:
            cat = "OK_span_hit"
        row["category"] = cat
        rows.append(row)

    # write CSV
    with (EVAL / "dev_retrieval_badcases.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # bootstrap CI (span hit @10, and page hit @10)
    random.seed(0)
    cis = {}
    for cfg in results:
        spans = results[cfg]["span"]
        pages = results[cfg]["page"]
        cis[cfg] = {
            "span_mean": round(sum(spans) / len(spans), 3),
            "span_ci": _bootstrap(spans),
            "page_mean": round(sum(pages) / len(pages), 3),
            "page_ci": _bootstrap(pages),
        }

    # case-level win/loss fixed_hybrid vs section_hybrid (span hit)
    fh_spans = [r for r in rows]
    fh_v = [r["fixed_hybrid_span_hit"] for r in rows]
    sh_v = [r["section_hybrid_span_hit"] for r in rows]
    wins = sum(1 for a, b in zip(fh_v, sh_v) if a > b)
    losses = sum(1 for a, b in zip(fh_v, sh_v) if a < b)
    ties = sum(1 for a, b in zip(fh_v, sh_v) if a == b)

    # references summary
    ref_share = [r["fixed_hybrid_ref_share"] for r in rows if r["fixed_hybrid_ref_share"] > 0]
    cat_counts = dict(Counter(r["category"] for r in rows))

    md = ["# DEV Retrieval Diagnosis", "",
          f"- evidence cases: **{len(cases)}**", "",
          "## Per-config evidence-span / page hit (with bootstrap 95% CI)", "",
          "| config | span@10 mean | span 95% CI | page@10 mean | page 95% CI |", "|---|---|---|---|---|"]
    for cfg, _, _ in CONFIGS:
        m = cis[cfg]
        md.append(f"| {cfg} | {m['span_mean']} | {m['span_ci']} | {m['page_mean']} | {m['page_ci']} |")
    md += ["", "## fixed_hybrid vs section_hybrid (evidence-span hit, case-level)", "",
           f"- fixed wins: **{wins}**", f"- section wins: **{losses}**", f"- ties: **{ties}**", "",
           "## References confound in fixed_hybrid", "",
           f"- gold-paper hits landing on a references page: **{ref_share and round(sum(ref_share)/len(ref_share),3) or 0.0}** (mean share; {len(ref_share)} cases with any ref-page hit)",
           "- conclusion: see below.", "",
           "## Failure-mode categories (fixed_hybrid)", "",
           f"`{cat_counts}`", "",
           "## Per-case table (see dev_retrieval_badcases.csv)", ""]
    (EVAL / "dev_retrieval_diagnosis.md").write_text("\n".join(md), encoding="utf-8")

    print(f"categories: {cat_counts}")
    print(f"fixed_hybrid vs section_hybrid (span): wins={wins} losses={losses} ties={ties}")
    for cfg, _, _ in CONFIGS:
        print(f"  {cfg}: span@10={cis[cfg]['span_mean']} ci={cis[cfg]['span_ci']} page@10={cis[cfg]['page_mean']} ci={cis[cfg]['page_ci']}")
    print(f"ref_share_mean={ref_share and round(sum(ref_share)/len(ref_share),3) or 0.0} (n={len(ref_share)})")


def _bootstrap(xs, iters=2000):
    n = len(xs)
    means = []
    for _ in range(iters):
        s = [xs[random.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo, hi = means[int(0.025 * iters)], means[int(0.975 * iters)]
    return f"[{lo:.3f}, {hi:.3f}]"


if __name__ == "__main__":
    main()
