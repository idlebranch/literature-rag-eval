"""Bad-case analysis for the 3 fixable ANSWERABLE retrieval misses (DEV).

Dumps Dense top-25 and Sparse top-25 (with gold span present?), and Hybrid top-10.
Classifies each miss: QUERY_MISMATCH / VOCAB_MISMATCH / DENSE_MISS / SPARSE_MISS /
CANDIDATE_K_TOO_SMALL / RRF_RANKING_LOSS / OTHER.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_eval_v2 import load_dev, set_index  # noqa: E402
from src.embedder import embed_query  # noqa: E402
from src.vectorstore import search  # noqa: E402
from validate_eval_v2 import norm  # noqa: E402

TARGETS = {"ev2_a001", "ev2_a050", "ev2_a086"}


def main():
    set_index("section_aware_270_gpu", "hybrid_dense_sparse")
    cases = [c for c in load_dev() if c["case_id"] in TARGETS]

    for c in cases:
        gold = norm(c["gold_evidence_text"])
        emb = embed_query(c["query"])
        dense = search(emb, top_k=25)
        d_span = any(gold in norm(h.get("text", "")) for h in dense)
        d_paper = [h["metadata"]["paper_id"] for h in dense]

        from src.sparse_encoder import encode_query_sparse
        from src.sparse_index import load_index
        from src.fusion import resolve_sparse_hits, rrf_fuse
        qw = encode_query_sparse(c["query"])
        scored = load_index(strict=True).search(qw, top_k=25)
        sparse = resolve_sparse_hits(scored)
        s_span = any(gold in norm(h.get("text", "")) for h in sparse)
        s_paper = [h["metadata"]["paper_id"] for h in sparse]

        hybrid = rrf_fuse(dense, sparse, fusion_k=10)
        h_span = any(gold in norm(h.get("text", "")) for h in hybrid)

        gpid = c["gold_paper_id"]
        d_rank = next((i for i, p in enumerate(d_paper, 1) if p == gpid), 0)
        s_rank = next((i for i, p in enumerate(s_paper, 1) if p == gpid), 0)

        cause = []
        if d_span: cause.append("dense_has_span")
        if s_span: cause.append("sparse_has_span")
        if not d_span and not s_span:
            cause.append("GOLD_SPAN_NOT_IN_TOP25")
        elif d_span and not h_span:
            cause.append("RRF_RANKING_LOSS")
        elif not d_span and s_span and h_span:
            cause.append("SPARSE_RESCUED")

        print(f"\n=== {c['case_id']}: {c['query'][:55]}")
        print(f"  gold: {c['gold_paper_id'][:30]} p{c['gold_page_start']} span={c['gold_evidence_text'][:60]!r}")
        print(f"  dense: paper_rank={d_rank} span_hit={d_span}")
        print(f"  sparse: paper_rank={s_rank} span_hit={s_span}")
        print(f"  hybrid: span_hit={h_span}  cause={'|'.join(cause) or 'OTHER'}")
        # show top dense sources
        tops = []
        for h in dense[:5]:
            m = h["metadata"]
            tops.append(f"{m['paper_id'][:22]}|{m.get('section','')[:14]}")
        print(f"  dense top5: {tops}")
        print(f"  query_norm={norm(c['query'])[:80]}")


if __name__ == "__main__":
    main()
