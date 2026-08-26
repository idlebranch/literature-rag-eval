"""Build the Eval V2 dataset from real PDF-derived chunk evidence.

Gold evidence is NEVER LLM-written. Each ANSWERABLE / PARTIAL / FALSE_PREMISE /
CONFLICTING case carries a ``gold_anchor`` (a short verbatim phrase) plus
``gold_paper_id``/``gold_page_start``; the exact evidence span is located
programmatically inside data/processed/section_chunks.jsonl (which is verbatim
PyMuPDF text). A case whose anchor cannot be located is rejected.

Outputs:
    data/eval_v2/eval_v2.jsonl
    data/eval_v2/dev.jsonl
    data/eval_v2/test.jsonl
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed" / "section_chunks.jsonl"
OUT_DIR = ROOT / "data" / "eval_v2"

# answerability_class -> allowed expected_action (single canonical value)
CLASS_ACTION = {
    "ANSWERABLE": "answer",
    "AMBIGUOUS": "clarify",
    "NO_EVIDENCE": "refuse",
    "PARTIAL_EVIDENCE": "partial_answer",
    "FALSE_PREMISE": "correct_premise",
    "CONFLICTING_EVIDENCE": "present_conflict",
}

NO_GOLD_CLASSES = {"AMBIGUOUS", "NO_EVIDENCE"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    for a, b in (("\u2212", "-"), ("\u2013", "-"), ("\u2014", "-"), ("\u03bc", "u")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).casefold()


def _sentence_window(text: str, start: int, end: int) -> str:
    """Expand [start,end) to the enclosing sentence(s), trimmed."""
    bounds = [m.end() for m in re.finditer(r"[.。!?！？;；]", text)]
    s = max([b for b in bounds if b <= start] or [0])
    e = min([b for b in bounds if b >= end] or [len(text)])
    return re.sub(r"\s+", " ", text[s:e]).strip(" .;:,-")


def locate_evidence(rows, case):
    """Locate the evidence span for an anchor across the whole corpus.

    The paper_id is populated from the matched chunk (never hand-entered), so it
    is guaranteed to be a real corpus paper. ``gold_page_start`` is used only as
    a disambiguation hint, not as a hard key.
    """
    anchor = case["gold_anchor"]
    page = case.get("gold_page_start")
    hits = []
    for r in rows:
        m = re.search(re.escape(anchor), r["text"], re.IGNORECASE)
        if m:
            hits.append((r, m))
    if not hits:
        return None, None
    if page is not None:
        for r, m in hits:
            if r["page_start"] <= page <= r["page_end"]:
                return _sentence_window(r["text"], m.start(), m.end()), r
    r, m = hits[0]
    return _sentence_window(r["text"], m.start(), m.end()), r


def build() -> list[dict]:
    rows = [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_pid = {}
    for r in rows:
        by_pid.setdefault(r["paper_id"], r)

    out, rejected = [], []
    for c in CASES:
        c = dict(c)
        c.setdefault("split", "test")
        c.setdefault("query_language", "zh")
        c.setdefault("expected_action", CLASS_ACTION[c["answerability_class"]])
        c.setdefault("gold_title", "")
        c.setdefault("gold_doi", "")
        c.setdefault("gold_page_end", c.get("gold_page_start"))
        c.setdefault("gold_section", "")
        c.setdefault("gold_evidence_text", "")
        c.setdefault("answer_key", "")
        c.setdefault("negative_rationale", "")
        c.setdefault("notes", "")
        c.setdefault("paired_with", "")

        if c["answerability_class"] in NO_GOLD_CLASSES:
            c["gold_paper_id"] = ""
            c["gold_page_start"] = None
            c["gold_page_end"] = None
            c["gold_evidence_text"] = ""
        elif "gold_anchor" in c:
            span, rec = locate_evidence(rows, c)
            if span is None:
                rejected.append((c["case_id"], c["gold_anchor"]))
                continue
            c["gold_evidence_text"] = span
            c["gold_paper_id"] = rec["paper_id"]
            c["gold_title"] = rec["title"]
            c["gold_doi"] = rec["doi"]
            c["gold_page_start"] = rec["page_start"]
            c["gold_page_end"] = rec["page_end"]
            c["gold_section"] = rec["section"]
            c.pop("gold_anchor", None)
        out.append(c)

    return out, rejected


def main() -> None:
    cases, rejected = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval_v2.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    (OUT_DIR / "dev.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases if c["split"] == "dev") + "\n",
        encoding="utf-8")
    (OUT_DIR / "test.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases if c["split"] == "test") + "\n",
        encoding="utf-8")
    print(f"built {len(cases)} cases (dev={sum(1 for c in cases if c['split']=='dev')}, "
          f"test={sum(1 for c in cases if c['split']=='test')})")
    if rejected:
        print(f"REJECTED {len(rejected)} cases (anchor not found):")
        for cid, anchor in rejected:
            print(f"  - {cid}: {anchor!r}")


# --------------------------------------------------------------------------
# Curated cases. gold_anchor is a verbatim phrase located in the corpus text.
# --------------------------------------------------------------------------

CASES = [
    # ---- ANSWERABLE (adsorption / ion exchange) ----
    dict(case_id="ev2_a001", split="dev", query_language="zh", query_type="parametric",
         answerability_class="ANSWERABLE",
         query="根据文献，用 H3PO4 活化的活性炭对铜的吸附容量是多少？",
         gold_paper_id="1-s2.0-S138589471401153X-main", gold_page_start=6,
         gold_anchor="46.3 mg/g for copper", answer_key="46.3 mg/g（比表面积 118.3 m2/g）",
         negative_rationale="不能编造其他数值；必须来自该篇活性炭文献。"),
    dict(case_id="ev2_a002", split="test", query_language="en", query_type="parametric",
         answerability_class="ANSWERABLE",
         query="What adsorption capacity for phosphate did the La(III)-loaded SOW gel reach?",
         gold_paper_id="2019_Performance_and_prospects_of_differ", gold_page_start=12,
         gold_anchor="13.63 mg-P", answer_key="13.63 mg-P/g",
         negative_rationale="不要与 78.1 mg-P/g 混淆（那是柱实验 300 mg-P/L 的值）。"),
    dict(case_id="ev2_a003", split="test", query_language="en", query_type="comparison",
         answerability_class="ANSWERABLE",
         query="How does sludge-derived adsorbent tetracycline capacity compare to commercial activated carbon?",
         gold_paper_id="2020_Occurrence_fate_and_risk_assessment", gold_page_start=10,
         gold_anchor="512\u2013672 mg/g",
         answer_key="sludge-derived 512-672 mg/g > commercial AC 65-471 mg/g",
         negative_rationale="必须给出两组数值的对比，不能只说 sludge 更好。"),

    # ---- ANSWERABLE (AOP / PMS / PDS / ozone / Fenton) ----
    dict(case_id="ev2_a010", split="dev", query_language="zh", query_type="parametric",
         answerability_class="ANSWERABLE",
         query="文献中 Fenton 反应的典型最佳 pH 是多少？",
         gold_paper_id="1-s2.0-S0926337314007176-main", gold_page_start=24,
         gold_anchor="optimum pH is typically near 3.0", answer_key="约 pH 3.0（接近 2.8）",
         negative_rationale="不能答成 pH 7 或其他值。"),
    dict(case_id="ev2_a011", split="test", query_language="en", query_type="factual",
         answerability_class="ANSWERABLE",
         query="In the presence of 25 ppm PMS and 0.1 mg/L Co2+, what E. coli removal was observed?",
         gold_paper_id="1-s2.0-S1385894716314759-main", gold_page_start=3,
         gold_anchor="99.99% removal of E. coli", answer_key="99.99% removal within 1 h",
         negative_rationale="条件必须对应：25 ppm PMS + 0.1 mg/L Co2+。"),
    dict(case_id="ev2_a012", split="test", query_language="en", query_type="factual",
         answerability_class="ANSWERABLE",
         query="At what pH is PMS stability minimal according to the literature?",
         gold_paper_id="1-s2.0-S1385894716314759-main", gold_page_start=3,
         gold_anchor="Minimum stability of PMS", answer_key="pH 9（pH<6 和 pH=12 相对稳定）",
         negative_rationale="不要答成 pH<6 或 pH=12 为最不稳定。"),
    dict(case_id="ev2_a013", split="dev", query_language="en", query_type="comparison",
         answerability_class="ANSWERABLE",
         query="How does photo-electro-Fenton degradation efficiency compare to electric Fenton for tetracycline?",
         gold_paper_id="2020_Occurrence_fate_and_risk_assessment", gold_page_start=11,
         gold_anchor="photo-electro-Fenton (98.5%)",
         answer_key="photo-electro-Fenton 98.5% > electric Fenton 87.7% > UV 13.5%",
         negative_rationale="要按降序给出三个工艺的数值。"),
    dict(case_id="ev2_a014", split="test", query_language="en", query_type="comparison",
         answerability_class="ANSWERABLE",
         query="Which achieved higher degradation: E-peroxone, electrolysis, or ozonation?",
         gold_paper_id="2020_Occurrence_fate_and_risk_assessment", gold_page_start=11,
         gold_anchor="E-peroxone (99%)",
         answer_key="E-peroxone 99% > ozonation 96% > electrolysis 82.9%",
         negative_rationale="顺序与数值必须正确。"),

    # ---- ANSWERABLE (membrane) ----
    dict(case_id="ev2_a020", split="test", query_language="en", query_type="parametric",
         answerability_class="ANSWERABLE",
         query="In Lee et al. (2001), what fraction of filtration resistance was the membrane itself?",
         gold_paper_id="1-s2.0-S0043135408006581-main", gold_page_start=4,
         gold_anchor="membrane resistance (12%)", answer_key="12%（其余为 cake 等阻力）",
         negative_rationale="只答 12%，不要混淆 cake resistance。"),
    dict(case_id="ev2_a021", split="dev", query_language="zh", query_type="factual",
         answerability_class="ANSWERABLE",
         query="文献中 MFR 的最佳投加量是多少（mg/mg MLSS）？",
         gold_paper_id="1-s2.0-S0043135408006581-main", gold_page_start=13,
         gold_anchor="0.025 mg/mg MLSS", answer_key="0.025 mg/mg MLSS",
         negative_rationale="超出最佳剂量会释放胞外可溶物。"),

    # ---- ANSWERABLE (biological) ----
    dict(case_id="ev2_a030", split="test", query_language="en", query_type="parametric",
         answerability_class="ANSWERABLE",
         query="What nitrogen removal rate did the low-strength anammox reactor reach at 16 C?",
         gold_paper_id="1-s2.0-S0960852415014777-main", gold_page_start=3,
         gold_anchor="2.28 kg N/m3/d", answer_key="2.28 kg N/m3/d（16 C 低浓度废水）",
         negative_rationale="不要答成常温或高浓度下的其他速率。"),
    dict(case_id="ev2_a031", split="test", query_language="zh", query_type="parametric",
         answerability_class="ANSWERABLE",
         query="文献报道氨化作用的最佳 pH 和温度范围是多少？",
         gold_paper_id="2006_Removal_of_nutrients_in_various_typ", gold_page_start=4,
         gold_anchor="optimal pH is between 6.5 and 8.5", answer_key="pH 6.5-8.5，温度 40-60 C",
         negative_rationale="pH 与温度两个量都要给出。"),
    dict(case_id="ev2_a032", split="dev", query_language="zh", query_type="parametric",
         answerability_class="ANSWERABLE",
         query="HNAD 工艺在 C/N=7.5 时总氮去除率是多少？",
         gold_paper_id="2021_A_review_of_research_progress_of_he", gold_page_start=6,
         gold_anchor="94.43% were achieved at C/N of 7.5", answer_key="总氮去除率 94.43%（HNAD 效率 94.21%）",
         negative_rationale="C/N=7.5 对应 94.43%，不要张冠李戴。"),

    # ---- ANSWERABLE (disinfection) ----
    dict(case_id="ev2_a040", split="test", query_language="en", query_type="parametric",
         answerability_class="ANSWERABLE",
         query="What log reduction of bacterial cells did ozonation achieve in the flow-cytometry study?",
         gold_paper_id="2007_Flow_cytometric_total_bacterial_cel", gold_page_start=4,
         gold_anchor="3 log reduction", answer_key="3 log reduction",
         negative_rationale="是臭氧化步骤的 3 log，不是后续 UF。"),

    # ---- ANSWERABLE (electrochemical / coagulation) ----
    dict(case_id="ev2_a050", split="dev", query_language="zh", query_type="parametric",
         answerability_class="ANSWERABLE",
         query="电絮凝去除 Acid Green 50 的最佳 pH 是多少？",
         gold_paper_id="1-s2.0-S0926337314007176-main", gold_page_start=8,
         gold_anchor="optimum pH of 6.9", answer_key="pH 6.9（电絮凝；EO 则无 pH 效应）",
         negative_rationale="不要与 EO 混淆（EO 无 pH 效应）。"),
    dict(case_id="ev2_a051", split="test", query_language="en", query_type="factual",
         answerability_class="ANSWERABLE",
         query="How effective was electrocoagulation at removing Cr from drinking water?",
         gold_paper_id="Chromium-removal-from-drinking-water-by-", gold_page_start=2,
         gold_anchor="reduction of Cr (over 98%)",
         answer_key="over 98%（初始 Cr 浓度最高 500 mg/L）",
         negative_rationale="不要答成 100%。"),
    dict(case_id="ev2_a052", split="test", query_language="en", query_type="factual",
         answerability_class="ANSWERABLE",
         query="How much oil and grease did electrocoagulation remove from olive mill wastewater?",
         gold_paper_id="2020_Application_of_coagulationflocculat", gold_page_start=11,
         gold_anchor="96% oil and grease", answer_key="96%",
         negative_rationale="是电絮凝处理橄榄油厂废水的除油率。"),

    # ---- ANSWERABLE (emerging contaminants) ----
    dict(case_id="ev2_a060", split="dev", query_language="zh", query_type="parametric",
         answerability_class="ANSWERABLE",
         query="该污水处理厂对微塑料的总去除率是多少？",
         gold_paper_id="2019_Transfer_and_fate_of_microplastics_", gold_page_start=3,
         gold_anchor="removal rate of 64.4%", answer_key="64.4%（进水 79.9 n/L → 出水 28.4 n/L）",
         negative_rationale="不要答成更高或更低值。"),

    # ---- ANSWERABLE (resource recovery) ----
    dict(case_id="ev2_a070", split="test", query_language="en", query_type="parametric",
         answerability_class="ANSWERABLE",
         query="What was the maximum phosphate removal for the bio-derived material at 10 mg/L?",
         gold_paper_id="2019_Performance_and_prospects_of_differ", gold_page_start=13,
         gold_anchor="optimal pH range for phosphate removal", answer_key="98.7%（10 mg/L）；25 mg/L 为 92.3%",
         negative_rationale="10 mg/L→98.7%，25 mg/L→92.3%，不能互换。"),

    # ---- AMBIGUOUS ----
    dict(case_id="ev2_b001", split="dev", query_language="zh", query_type="comparison",
         answerability_class="AMBIGUOUS",
         query="哪种高级氧化工艺最好？",
         negative_rationale="缺污染物、指标、工况，无法唯一判定，应澄清。"),
    dict(case_id="ev2_b002", split="test", query_language="en", query_type="comparison",
         answerability_class="AMBIGUOUS",
         query="Which membrane is better?",
         negative_rationale="无应用场景、目标污染物与指标。"),
    dict(case_id="ev2_b003", split="test", query_language="zh", query_type="factual",
         answerability_class="AMBIGUOUS",
         query="活性炭能不能再生？",
         negative_rationale="未指定炭种、吸附质与再生方式，多解。"),
    dict(case_id="ev2_b004", split="dev", query_language="en", query_type="factual",
         answerability_class="AMBIGUOUS",
         query="Is ozone effective?",
         negative_rationale="未指定污染物、水质与目标。"),
    dict(case_id="ev2_b005", split="test", query_language="zh", query_type="parametric",
         answerability_class="AMBIGUOUS",
         query="絮凝剂加多少合适？",
         negative_rationale="未指定水质、絮凝剂种类与目标。"),

    # ---- NO_EVIDENCE (out-of-scope) ----
    dict(case_id="ev2_c001", split="dev", query_language="zh", query_type="factual",
         answerability_class="NO_EVIDENCE",
         query="2024 年诺贝尔物理学奖颁给了谁？",
         negative_rationale="与水处理语料无关，应拒答/声明无证据。"),
    dict(case_id="ev2_c002", split="test", query_language="en", query_type="factual",
         answerability_class="NO_EVIDENCE",
         query="What is the capital of France?",
         negative_rationale="out-of-scope。"),
    dict(case_id="ev2_c003", split="test", query_language="en", query_type="factual",
         answerability_class="NO_EVIDENCE",
         query="How to cook pasta al dente?",
         negative_rationale="out-of-scope。"),

    # ---- NO_EVIDENCE (in-domain unsupported) ----
    dict(case_id="ev2_c010", split="dev", query_language="en", query_type="parametric",
         answerability_class="NO_EVIDENCE",
         query="What is the exact half-life of PFOS in human serum reported in this corpus?",
         negative_rationale="语料为水处理，不包含毒代动力学血清半衰期数值。"),
    dict(case_id="ev2_c011", split="test", query_language="zh", query_type="parametric",
         answerability_class="NO_EVIDENCE",
         query="处理市政污水的 AOP 电耗成本精确是多少元/立方米？",
         negative_rationale="语料无该统一精确数值，应声明证据不足。"),
    dict(case_id="ev2_c012", split="test", query_language="en", query_type="causal",
         answerability_class="NO_EVIDENCE",
         query="Does adding salt to activated carbon always double its adsorption capacity?",
         negative_rationale="无该因果结论证据。"),

    # ---- PARTIAL_EVIDENCE ----
    dict(case_id="ev2_d001", split="dev", query_language="zh", query_type="synthesis",
         answerability_class="PARTIAL_EVIDENCE",
         query="请给出本语料覆盖的所有 AOP 工艺的完整经济成本对比。",
         gold_paper_id="1-s2.0-S0926337314007176-main", gold_page_start=24,
         gold_anchor="optimum pH is typically near 3.0",
         answer_key="语料只有局部成本/条件证据，只能给出部分对比并注明局限。",
         negative_rationale="语料不足以支撑全工艺成本对比，应部分回答+局限说明。"),
    dict(case_id="ev2_d002", split="test", query_language="en", query_type="synthesis",
         answerability_class="PARTIAL_EVIDENCE",
         query="Provide a complete quantitative comparison of all membrane fouling control methods.",
         gold_paper_id="1-s2.0-S0043135408006581-main", gold_page_start=4,
         gold_anchor="membrane resistance (12%)",
         answer_key="只能给出部分数值（如阻力占比），无法完整量化所有方法。",
         negative_rationale="语料为综述，缺少统一量化。"),

    # ---- FALSE_PREMISE (paired with answerable) ----
    dict(case_id="ev2_e001", split="dev", query_language="zh", query_type="factual",
         answerability_class="FALSE_PREMISE", paired_with="ev2_a050",
         query="电絮凝去除 Acid Green 50 的最佳 pH 是不是 10？",
         gold_paper_id="1-s2.0-S0926337314007176-main", gold_page_start=8,
         gold_anchor="optimum pH of 6.9",
         answer_key="否，最佳 pH 为 6.9。",
         negative_rationale="错误前提（pH=10），应纠正为 6.9。"),
    dict(case_id="ev2_e002", split="test", query_language="zh", query_type="factual",
         answerability_class="FALSE_PREMISE", paired_with="ev2_a010",
         query="Fenton 反应的最佳 pH 是不是 7？",
         gold_paper_id="1-s2.0-S0926337314007176-main", gold_page_start=24,
         gold_anchor="optimum pH is typically near 3.0",
         answer_key="否，约 pH 3.0。",
         negative_rationale="错误前提（pH=7），应纠正。"),
    dict(case_id="ev2_e003", split="test", query_language="en", query_type="factual",
         answerability_class="FALSE_PREMISE", paired_with="ev2_a060",
         query="Does this WWTP remove 95% of microplastics?",
         gold_paper_id="2019_Transfer_and_fate_of_microplastics_", gold_page_start=3,
         gold_anchor="removal rate of 64.4%",
         answer_key="否，实际为 64.4%。",
         negative_rationale="错误前提（95%），应纠正为 64.4%。"),
    dict(case_id="ev2_e004", split="dev", query_language="en", query_type="factual",
         answerability_class="FALSE_PREMISE", paired_with="ev2_a013",
         query="Does electric Fenton outperform photo-electro-Fenton for tetracycline?",
         gold_paper_id="2020_Occurrence_fate_and_risk_assessment", gold_page_start=11,
         gold_anchor="photo-electro-Fenton (98.5%)",
         answer_key="否，photo-electro-Fenton 98.5% > electric Fenton 87.7%。",
         negative_rationale="错误前提（顺序反了），应纠正。"),

    # ---- CONFLICTING_EVIDENCE (Atrazine: pH-dependent results) ----
    dict(case_id="ev2_f001", split="test", query_language="en", query_type="comparison",
         answerability_class="CONFLICTING_EVIDENCE",
         query="Does Co2+/PMS remove atrazine fully?",
         gold_paper_id="1-s2.0-S1385894716314759-main", gold_page_start=5,
         gold_anchor="Fe2+ Atrazine PMS = 1 mM",
         answer_key="结果因条件冲突：Fe2+/PMS pH3.0 15min <50%；Co2+/PMS pH7 45min 100%。需分别呈现。",
         negative_rationale="同一污染物不同条件给出冲突结果，不能给唯一结论。"),

    # ---- ANSWERABLE (round 2) ----
    dict(case_id="ev2_a080", split="test", query_language="en", query_type="factual",
         answerability_class="ANSWERABLE",
         query="When the activated carbon surface area increased from 121.3 to 346.5 m2/g, what were the adsorption capacities?",
         gold_anchor="27.7 and 35.1 mg/g", gold_page_start=7,
         answer_key="27.7 → 35.1 mg/g",
         negative_rationale="不能只说容量随比表面积升高，要给出两个数值。"),
    dict(case_id="ev2_a081", split="dev", query_language="en", query_type="factual",
         answerability_class="ANSWERABLE",
         query="In what fraction of surface water samples was PFOA detected?",
         gold_anchor="PFOA was detected in 98%", gold_page_start=7,
         answer_key="98% 地表水样；地下水 36%",
         negative_rationale="区分地表水 98% 与地下水 36%。"),
    dict(case_id="ev2_a082", split="test", query_language="en", query_type="parametric",
         answerability_class="ANSWERABLE",
         query="What methyl orange degradation did the bifunctional fabric cotton achieve?",
         gold_anchor="96% methyl orange", gold_page_start=14,
         answer_key="96%（1 sun, 3 h）",
         negative_rationale="给出条件：1 sun / 3 h。"),
    dict(case_id="ev2_a083", split="dev", query_language="en", query_type="comparison",
         answerability_class="ANSWERABLE",
         query="What was the flux recovery ratio of PVDF membranes without TiO2 NPs after UV cleaning?",
         gold_anchor="was only 73%", gold_page_start=22,
         answer_key="仅 73%（含 TiO2 的膜可 100% 恢复）",
         negative_rationale="区分 73% 与含 TiO2 的完全恢复。"),
    dict(case_id="ev2_a084", split="test", query_language="zh", query_type="factual",
         answerability_class="ANSWERABLE",
         query="温度升高时，絮凝剂的最佳投加量如何变化？",
         gold_anchor="decreases as the temperature increases", gold_page_start=6,
         answer_key="在合适范围内随温度升高而降低",
         negative_rationale="给出方向：降低。"),
    dict(case_id="ev2_a085", split="test", query_language="en", query_type="factual",
         answerability_class="ANSWERABLE",
         query="What is the nitrogen content of graphitic carbon nitride (g-C3N4)?",
         gold_anchor="high nitrogen content (around 70% wt)", gold_page_start=14,
         answer_key="约 70 wt%",
         negative_rationale="仅 g-C3N4 氮含量约 70%，不要延伸到其他材料。"),
    dict(case_id="ev2_a086", split="dev", query_language="en", query_type="factual",
         answerability_class="ANSWERABLE",
         query="In what fraction of samples were PFAS detected overall?",
         gold_anchor="PFAS were detected in 67%", gold_page_start=9,
         answer_key="67%（PFOA 最常检出，约 55%）",
         negative_rationale="总体 67%，PFOA 55%。"),
    dict(case_id="ev2_a087", split="test", query_language="en", query_type="factual",
         answerability_class="ANSWERABLE",
         query="Can TiO2 enhance solar disinfection of fecal coliforms?",
         gold_anchor="complete inactivation of fecal coliforms", gold_page_start=11,
         answer_key="是，TiO2 可显著促进粪大肠菌群的完全灭活",
         negative_rationale="强调太阳能消毒场景。"),

    # ---- AMBIGUOUS (round 2) ----
    dict(case_id="ev2_b006", split="test", query_language="zh", query_type="parametric",
         answerability_class="AMBIGUOUS",
         query="臭氧应该投加多少？",
         negative_rationale="未指定水质、目标污染物与去除率。"),
    dict(case_id="ev2_b007", split="dev", query_language="zh", query_type="comparison",
         answerability_class="AMBIGUOUS",
         query="哪种膜材料最好？",
         negative_rationale="未指定应用、进水水质与评价指标。"),
    dict(case_id="ev2_b008", split="test", query_language="en", query_type="factual",
         answerability_class="AMBIGUOUS",
         query="Is photocatalysis better than ozonation?",
         negative_rationale="未指定污染物、条件与指标。"),

    # ---- NO_EVIDENCE (round 2) ----
    dict(case_id="ev2_c013", split="test", query_language="en", query_type="factual",
         answerability_class="NO_EVIDENCE",
         query="How do I configure a home Wi-Fi router?",
         negative_rationale="out-of-scope。"),
    dict(case_id="ev2_c014", split="dev", query_language="zh", query_type="parametric",
         answerability_class="NO_EVIDENCE",
         query="该语料中 PFAS 的毒理学致癌斜率因子是多少？",
         negative_rationale="语料为水处理，不含毒理学斜率因子数值。"),
    dict(case_id="ev2_c015", split="test", query_language="zh", query_type="parametric",
         answerability_class="NO_EVIDENCE",
         query="臭氧-活性炭联用对某特定农药的去除率精确是多少？",
         negative_rationale="语料无该特定农药联用精确数值。"),

    # ---- PARTIAL_EVIDENCE (round 2) ----
    dict(case_id="ev2_d003", split="test", query_language="zh", query_type="synthesis",
         answerability_class="PARTIAL_EVIDENCE",
         query="给出语料中所有 PFAS 去除技术的完整成本-效益分析。",
         gold_anchor="PFAS were detected in 67%", gold_page_start=9,
         answer_key="只能给出部分技术/部分数据，无法完整成本效益分析。",
         negative_rationale="语料缺系统化成本数据。"),
    dict(case_id="ev2_d004", split="dev", query_language="zh", query_type="synthesis",
         answerability_class="PARTIAL_EVIDENCE",
         query="给出语料中所有膜蒸馏配置的通量与热效率完整对比。",
         gold_anchor="membrane resistance (12%)", gold_page_start=4,
         answer_key="只能部分对比，无法覆盖所有配置。",
         negative_rationale="语料为部分综述，缺统一量化。"),

    # ---- FALSE_PREMISE (round 2) ----
    dict(case_id="ev2_e005", split="test", query_language="zh", query_type="factual",
         answerability_class="FALSE_PREMISE", paired_with="ev2_a050",
         query="电絮凝去除 Acid Green 50 的最佳 pH 是不是 3.0？",
         gold_anchor="optimum pH of 6.9", gold_page_start=8,
         answer_key="否，最佳 pH 为 6.9。",
         negative_rationale="错误前提（3.0），应纠正为 6.9。"),
    dict(case_id="ev2_e006", split="dev", query_language="zh", query_type="factual",
         answerability_class="FALSE_PREMISE", paired_with="ev2_a012",
         query="PMS 在 pH=9 时是不是最稳定？",
         gold_anchor="Minimum stability of PMS", gold_page_start=3,
         answer_key="否，pH=9 时稳定性最低。",
         negative_rationale="错误前提（最稳定），应纠正为最不稳定。"),
    dict(case_id="ev2_e007", split="test", query_language="en", query_type="factual",
         answerability_class="FALSE_PREMISE", paired_with="ev2_a081",
         query="Was PFOA detected in 100% of surface water samples?",
         gold_anchor="PFOA was detected in 98%", gold_page_start=7,
         answer_key="否，为 98%。",
         negative_rationale="错误前提（100%），应纠正为 98%。"),
]

if __name__ == "__main__":
    main()
