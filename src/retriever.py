from typing import Dict, List
from src.config import settings
from src.embedder import embed_query
from src.vectorstore import search


def expand_query(query: str) -> str:
    """Rule-based bilingual query expansion for environmental RAG."""
    q = query
    lower_q = query.lower()

    expansions = []

    if "高级氧化" in q or "aop" in lower_q:
        expansions.append(
            "advanced oxidation processes AOPs emerging contaminants micropollutants hydroxyl radical sulfate radical reactive oxygen species"
        )

    if "pms" in lower_q or "pds" in lower_q or "过硫酸" in q or "单过硫酸" in q:
        expansions.append(
            "peroxymonosulfate PMS peroxydisulfate PDS persulfate activation sulfate radical SO4 hydroxyl radical singlet oxygen antibiotics bisphenol A BPA"
        )

    if "臭氧" in q or "ozone" in lower_q or "ozonation" in lower_q:
        expansions.append(
            "ozonation catalytic ozonation ozone O3 hydroxyl radical micropollutants emerging contaminants water treatment"
        )

    if "光催化" in q or "photocatal" in lower_q:
        expansions.append(
            "photocatalysis TiO2 g-C3N4 visible light pharmaceuticals personal care products PPCPs degradation"
        )

    if "fenton" in lower_q or "芬顿" in q:
        expansions.append(
            "Fenton photo-Fenton heterogeneous Fenton iron catalyst hydroxyl radical emerging contaminants wastewater"
        )

    if "pfas" in lower_q or "pfoa" in lower_q or "pfos" in lower_q or "全氟" in q:
        expansions.append(
            "PFAS PFOA PFOS per- and polyfluoroalkyl substances removal treatment adsorption membrane ion exchange activated carbon destruction limitations full-scale application"
        )

    if "新污染物" in q:
        expansions.append(
            "emerging contaminants micropollutants PPCPs pharmaceuticals endocrine disrupting compounds PFAS antibiotics water treatment"
        )

    if expansions:
        return query + "\n" + "\n".join(expansions)

    return query


def diversify_hits(hits: List[Dict], top_k: int, max_per_source: int = 2) -> List[Dict]:
    """Avoid all top results coming from the same PDF."""
    selected = []
    source_count = {}

    for hit in hits:
        source = hit["metadata"]["source"]
        count = source_count.get(source, 0)

        if count < max_per_source:
            selected.append(hit)
            source_count[source] = count + 1

        if len(selected) >= top_k:
            break

    return selected


def retrieve(query: str, top_k: int | None = None) -> List[Dict]:
    if top_k is None:
        top_k = settings.top_k

    expanded_query = expand_query(query)
    q_emb = embed_query(expanded_query)

    raw_hits = search(q_emb, top_k=max(top_k * 4, 20))
    hits = diversify_hits(raw_hits, top_k=top_k, max_per_source=2)

    return hits


def format_context(hits: List[Dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        blocks.append(
            f"[S{i}] source={meta['source']} page={meta['page']} "
            f"chunk={meta['chunk_index']}\n{hit['text']}"
        )
    return "\n\n".join(blocks)
