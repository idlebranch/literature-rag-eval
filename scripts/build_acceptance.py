"""Build the Final Acceptance Set (fresh cases, never-used gold paper_ids).

Gold evidence is located verbatim in section_chunks.jsonl (never LLM-written),
preferring paper_ids that were NOT used as Eval V2 gold. Writes
data/acceptance/acceptance.jsonl and freezes a SHA-256 hash.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "processed" / "section_chunks.jsonl"
OUT = ROOT / "data" / "acceptance"

CLASS_ACTION = {
    "ANSWERABLE": "answer",
    "AMBIGUOUS": "clarify",
    "NO_EVIDENCE": "refuse",
    "PARTIAL_EVIDENCE": "partial_answer",
    "FALSE_PREMISE": "correct_premise",
    "CONDITIONALLY_DIVERGENT": "answer",  # present sources+conditions, no forced merge
}
NO_GOLD = {"AMBIGUOUS", "NO_EVIDENCE"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    for a, b in (("\u2212", "-"), ("\u2013", "-"), ("\u2014", "-"), ("\u03bc", "u")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).casefold()


def _window(text, start, end):
    bounds = [m.end() for m in re.finditer(r"[.。!?！?;；]", text)]
    s = max([b for b in bounds if b <= start] or [0])
    e = min([b for b in bounds if b >= end] or [len(text)])
    return re.sub(r"\s+", " ", text[s:e]).strip(" .;:,-")


def locate(rows, anchor, page=None):
    for r in rows:
        if page is not None and not (r["page_start"] <= page <= r["page_end"]):
            continue
        m = re.search(re.escape(anchor), r["text"], re.IGNORECASE)
        if m:
            return _window(r["text"], m.start(), m.end()), r
    return None, None


def build():
    rows = [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()]
    out, rejected = [], []
    for c in CASES:
        c = dict(c)
        c.setdefault("query_language", "zh")
        c.setdefault("expected_action", CLASS_ACTION[c["answerability_class"]])
        c.setdefault("gold_title", ""); c.setdefault("gold_doi", "")
        c.setdefault("gold_section", ""); c.setdefault("gold_evidence_text", "")
        c.setdefault("answer_key", ""); c.setdefault("negative_rationale", "")
        c.setdefault("notes", "")
        if c["answerability_class"] in NO_GOLD:
            c["gold_paper_id"] = ""; c["gold_page_start"] = None; c["gold_page_end"] = None
            out.append(c); continue
        span, rec = locate(rows, c["gold_anchor"], c.get("gold_page_start"))
        if span is None:
            rejected.append((c["case_id"], c["gold_anchor"])); continue
        c["gold_evidence_text"] = span
        c["gold_paper_id"] = rec["paper_id"]
        c["gold_title"] = rec["title"]; c["gold_doi"] = rec["doi"]
        c["gold_page_start"] = rec["page_start"]; c["gold_page_end"] = rec["page_end"]
        c["gold_section"] = rec["section"]
        c.pop("gold_anchor", None)
        out.append(c)
    return out, rejected


def main():
    cases, rejected = build()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n"
    (OUT / "acceptance.jsonl").write_text(payload, encoding="utf-8")
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (OUT / "acceptance.hash").write_text(h, encoding="utf-8")
    print(f"built {len(cases)} cases, hash={h}")
    if rejected:
        print("REJECTED:", rejected)
        raise SystemExit(1)


CASES = [
    # ---- ANSWERABLE (never-used papers) ----
    dict(case_id="acc_a001", query_language="zh", answerability_class="ANSWERABLE",
         query="磁性水热炭在 pH 9、投加量 1.2 g/L 时对亚甲基蓝的去除率是多少？",
         gold_anchor="71.74%和36.02%",
         answer_key="MB 去除率 71.74%（COD 去除率 36.02%）"),
    dict(case_id="acc_a002", query_language="zh", answerability_class="ANSWERABLE",
         query="CHTC 对亚甲基蓝的实际平衡吸附量是多少？",
         gold_anchor="34.64mg/g",
         answer_key="34.64 mg/g（MHTC 为 46.64 mg/g）"),
    dict(case_id="acc_a003", query_language="zh", answerability_class="ANSWERABLE",
         query="热活化过硫酸盐静态再生氧氟沙星饱和活性炭的再生率是多少？",
         gold_anchor="再生率为56.19%",
         answer_key="56.19%（KPS 10 mmol/L、H2O2 25%、pH 7、60℃、2h）"),
    dict(case_id="acc_a004", query_language="zh", answerability_class="ANSWERABLE",
         query="臭氧微纳气泡再生活性炭的制水成本比臭氧活性炭工艺降低了多少？",
         gold_anchor="78.92%~81.26%",
         answer_key="78.92%~81.26%"),
    dict(case_id="acc_a005", query_language="en", answerability_class="ANSWERABLE",
         query="What ammonium concentration does municipal wastewater typically contain?",
         gold_anchor="around 100 mg/L of ammonium",
         answer_key="约 100 mg/L"),
    dict(case_id="acc_a006", query_language="en", answerability_class="ANSWERABLE",
         query="What fraction of Earth's water is suitable for direct human use and consumption?",
         gold_anchor="around 2.5%",
         answer_key="约 2.5%"),
    dict(case_id="acc_a007", query_language="en", answerability_class="ANSWERABLE",
         query="What salinity range defines low-salinity brackish water RO (BWRO)?",
         gold_anchor="500 and 2500 mg/L",
         answer_key="500–2500 mg/L"),
    dict(case_id="acc_a008", query_language="zh", answerability_class="ANSWERABLE",
         query="粉末活性炭吸附-氧化联用工艺对低浓度嗅味物质能保持多高的去除效果？",
         gold_anchor="保持稳定去除效果",
         answer_key="＞80%"),
    dict(case_id="acc_a009", query_language="zh", answerability_class="ANSWERABLE",
         query="热活化过硫酸盐再生氧氟沙星饱和活性炭时 KPS 的最佳投加量是多少？",
         gold_anchor="KPS 的投加量为10mmol/L",
         answer_key="10 mmol/L"),
    dict(case_id="acc_a010", query_language="en", answerability_class="ANSWERABLE",
         query="In the oilfield produced water review, how much oil is produced per day from conventional onshore and offshore fields?",
         gold_anchor="300 million barrels per day",
         answer_key="约 300 million barrels/day"),
    dict(case_id="acc_a011", query_language="zh", answerability_class="ANSWERABLE",
         query="煤质活性炭经过 7 次吸附-再生循环后对莠去津、双酚A、磺胺甲恶唑的初始去除率还能达到多少？",
         gold_anchor="初始去除率仍然高于60%",
         answer_key="仍然高于 60%"),
    dict(case_id="acc_a012", query_language="zh", answerability_class="ANSWERABLE",
         query="臭氧微纳气泡间歇再生运行中，活性炭出水的高锰酸盐指数在什么范围？",
         gold_anchor="高锰酸盐指数在0.5 mg/L~0.9 mg/L",
         answer_key="0.5 mg/L~0.9 mg/L"),
    dict(case_id="acc_a013", query_language="zh", answerability_class="ANSWERABLE",
         query="静态吸附 48 小时后经臭氧微纳气泡再生 4 小时，活性炭对莠去津、双酚A、磺胺甲恶唑的去除效果如何？",
         gold_anchor="去除效果均能恢复到",
         answer_key="均能恢复到与原始活性炭相当的水平"),
    dict(case_id="acc_a014", query_language="zh", answerability_class="ANSWERABLE",
         query="椰壳活性炭经过 3 次吸附-再生循环后对双酚A、磺胺甲恶唑的初始去除率是多少？",
         gold_anchor="仅有50%",
         answer_key="仅有 50%"),

    # ---- AMBIGUOUS ----
    dict(case_id="acc_b001", query_language="zh", answerability_class="AMBIGUOUS",
         query="活性炭再生哪种方式最好？"),
    dict(case_id="acc_b002", query_language="en", answerability_class="AMBIGUOUS",
         query="Which catalyst is the best?"),
    dict(case_id="acc_b003", query_language="zh", answerability_class="AMBIGUOUS",
         query="这个工艺的效果如何？"),
    dict(case_id="acc_b004", query_language="en", answerability_class="AMBIGUOUS",
         query="Is membrane treatment effective?"),

    # ---- NO_EVIDENCE ----
    dict(case_id="acc_c001", query_language="zh", answerability_class="NO_EVIDENCE",
         query="2024 年世界杯冠军是谁？"),
    dict(case_id="acc_c002", query_language="en", answerability_class="NO_EVIDENCE",
         query="What is the melting point of gold?"),
    dict(case_id="acc_c003", query_language="zh", answerability_class="NO_EVIDENCE",
         query="该语料中 PFAS 的人体致癌斜率因子精确值是多少？"),
    dict(case_id="acc_c004", query_language="en", answerability_class="NO_EVIDENCE",
         query="What is the exact cost per cubic meter of every AOP in this corpus?"),
    dict(case_id="acc_c005", query_language="zh", answerability_class="NO_EVIDENCE",
         query="某特定新型农药在臭氧+活性炭下的去除率精确是多少？"),

    # ---- PARTIAL_EVIDENCE ----
    dict(case_id="acc_d001", query_language="zh", answerability_class="PARTIAL_EVIDENCE",
         query="给出本语料所有活性炭再生方法的完整成本对比。",
         gold_anchor="78.92%~81.26%",
         answer_key="只能给出部分：臭氧微纳气泡再生制水成本降低 78.92%~81.26%，无法完整对比所有再生方法"),
    dict(case_id="acc_d002", query_language="zh", answerability_class="PARTIAL_EVIDENCE",
         query="给出语料中所有吸附剂对磺胺甲恶唑的完整吸附容量对比。",
         gold_anchor="126.149 mg/g",
         answer_key="只能给出部分：活性炭 126.149 > 碳纳米管 73.691 > 生物活性炭 41.211 mg/g，无法覆盖所有吸附剂"),
    dict(case_id="acc_d003", query_language="en", answerability_class="PARTIAL_EVIDENCE",
         query="Give a complete quantitative comparison of every photocatalyst in this corpus.",
         gold_anchor="Perovskite Type ABO3",
         answer_key="只能给出部分：语料含钙钛矿 ABO3 光催化综述，但无法量化对比所有光催化剂"),

    # ---- FALSE_PREMISE ----
    dict(case_id="acc_e001", query_language="zh", answerability_class="FALSE_PREMISE",
         query="磁性水热炭在 pH 9 时对亚甲基蓝的去除率是不是 95%？",
         gold_anchor="71.74%和36.02%",
         answer_key="否，是 71.74%"),
    dict(case_id="acc_e002", query_language="zh", answerability_class="FALSE_PREMISE",
         query="热活化过硫酸盐再生氧氟沙星饱和活性炭的再生率是不是 90%？",
         gold_anchor="再生率为56.19%",
         answer_key="否，是 56.19%"),
    dict(case_id="acc_e003", query_language="en", answerability_class="FALSE_PREMISE",
         query="Does municipal wastewater contain 500 mg/L of ammonium?",
         gold_anchor="around 100 mg/L of ammonium",
         answer_key="否，约 100 mg/L"),
    dict(case_id="acc_e004", query_language="en", answerability_class="FALSE_PREMISE",
         query="Is 50% of Earth's water suitable for direct human use?",
         gold_anchor="around 2.5%",
         answer_key="否，约 2.5%"),

    # ---- CONDITIONALLY_DIVERGENT (answer + present sources/conditions) ----
    dict(case_id="acc_f001", query_language="zh", answerability_class="CONDITIONALLY_DIVERGENT",
         query="经过多次吸附-再生循环后，煤质和椰壳活性炭对莠去津的去除率有什么差异？",
         gold_anchor="初始去除率仅有40%",
         answer_key="条件依赖：煤质活性炭 7 次循环后 >60%；椰壳活性炭 3 次循环后仅 40%；应分别呈现材料与条件",
         notes="conditionally divergent: coal vs coconut activated carbon -> different atrazine removal"),
    dict(case_id="acc_f002", query_language="zh", answerability_class="CONDITIONALLY_DIVERGENT",
         query="CHTC 和 MHTC 对亚甲基蓝的吸附能力谁更强？",
         gold_anchor="34.64mg/g、46.64",
         answer_key="MHTC 更强（46.64 vs 34.64 mg/g），且 MHTC 投加量比 CHTC 省 25%"),
]

if __name__ == "__main__":
    main()
