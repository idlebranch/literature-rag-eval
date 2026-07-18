"""Produce the three deliverables from a single source of truth (EVALUATIONS):

1. outputs/answer_judge_report_claude.md  (full per-question report + summary)
2. outputs/answer_judge_summary.csv       (compact CSV for downstream analysis)
3. outputs/badcase_analysis.md            (error_type-grouped badcase write-up)
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
SRC_MD = OUT_DIR / "answer_eval_manual.md"
SRC_CSV = OUT_DIR / "answer_eval_manual.csv"  # for question text
REPORT_MD = OUT_DIR / "answer_judge_report_claude.md"
SUMMARY_CSV = OUT_DIR / "answer_judge_summary.csv"
BADCASE_MD = OUT_DIR / "badcase_analysis.md"

JUDGE_NAME = "claude-opus-4-7"

# ---------------------------------------------------------------------------
# Evaluations (single source of truth)
# ---------------------------------------------------------------------------
EVALUATIONS: dict[str, dict] = {
    "q001": {
        "faithfulness_score": 4,
        "completeness_score": 3,
        "citation_score": 4,
        "overall_score": 4,
        "error_type": "incomplete_answer",
        "judge_reason": "覆盖·OH/SO₄·⁻/Cl·三类自由基与矿化机理且引用合理，但遗漏了非自由基（直接电子转移）路径和¹O₂、O₂·⁻等活性物种，也未点出主导机制随氧化剂/催化剂/pH/基质变化。",
    },
    "q002": {
        "faithfulness_score": 5,
        "completeness_score": 4,
        "citation_score": 4,
        "overall_score": 4,
        "error_type": "incomplete_answer",
        "judge_reason": "区分了PMS/PDS主导路径并列出SO₄·⁻、¹O₂、•OH等主要物种，同时坦诚说明材料未专门覆盖抗生素场景，但漏掉了O₂·⁻和直接电子转移路径。",
    },
    "q003": {
        "faithfulness_score": 5,
        "completeness_score": 4,
        "citation_score": 5,
        "overall_score": 4,
        "error_type": "incomplete_answer",
        "judge_reason": "清晰对比了臭氧两条路径与催化臭氧的强化机制并引用规范，但缺少催化臭氧本身的局限（催化剂稳定性、副产物控制、基质影响）这一对比维度。",
    },
    "q004": {
        "faithfulness_score": 3,
        "completeness_score": 1,
        "citation_score": 1,
        "overall_score": 1,
        "error_type": "generation_error",
        "judge_reason": "检索源中Photodegradation of PPCPs和Photocatalytic membranes均高度相关（distance低于0.63），但模型直接拒答，是典型的生成端过度保守而非检索失败。",
    },
    "q005": {
        "faithfulness_score": 4,
        "completeness_score": 2,
        "citation_score": 3,
        "overall_score": 2,
        "error_type": "incomplete_answer",
        "judge_reason": "仅指出催化剂金属种类差异，未充分挖掘Heterogeneous Fenton catalysts综述中关于pH适用范围、铁泥、非均相催化剂回收性等典型差异。",
    },
    "q006": {
        "faithfulness_score": 5,
        "completeness_score": 4,
        "citation_score": 5,
        "overall_score": 4,
        "error_type": "incomplete_answer",
        "judge_reason": "成本、规模化、副产物、技术成熟度等工程层面挑战覆盖详细且引用清晰，但缺少PFAS自身C-F键稳定性导致传统方法效果有限以及富集液处置问题。",
    },
    "q007": {
        "faithfulness_score": 4,
        "completeness_score": 2,
        "citation_score": 4,
        "overall_score": 2,
        "error_type": "insufficient_context",
        "judge_reason": "AOP部分详尽且引用规范，但吸附法部分完全缺失，是检索端未召回吸附法专题文献所致，导致对比类问题只回答了一半。",
    },
    "q008": {
        "faithfulness_score": 5,
        "completeness_score": 4,
        "citation_score": 5,
        "overall_score": 4,
        "error_type": "incomplete_answer",
        "judge_reason": "中间产物毒性升高的因果链阐述清晰且引用精准，但未提出综合矿化率、转化产物识别和生物毒性测试这一评估方法论建议。",
    },
    "q009": {
        "faithfulness_score": 5,
        "completeness_score": 3,
        "citation_score": 4,
        "overall_score": 3,
        "error_type": "incomplete_answer",
        "judge_reason": "重点落在PMS/PDS体系差异和活性物种种类，对自由基vs非自由基路径在反应速率、选择性、抗基质淬灭能力等核心反应特性差异的说明不够明确。",
    },
    "q010": {
        "faithfulness_score": 5,
        "completeness_score": 4,
        "citation_score": 5,
        "overall_score": 4,
        "error_type": "incomplete_answer",
        "judge_reason": "DOM竞争、pH连锁效应和溴酸盐副反应三大基质效应解释清晰且引用规范，但碳酸盐/碳酸氢盐淬灭·OH、氯离子、悬浮物等其他基质成分未覆盖。",
    },
    "q011": {
        "faithfulness_score": 5,
        "completeness_score": 4,
        "citation_score": 5,
        "overall_score": 4,
        "error_type": "incomplete_answer",
        "judge_reason": "覆盖了PFRs、缺陷与石墨化结构、N掺杂和金属复合等活化机制并引用精准，但未明确点出表面电子转移这一典型非自由基机制。",
    },
    "q012": {
        "faithfulness_score": 4,
        "completeness_score": 2,
        "citation_score": 3,
        "overall_score": 2,
        "error_type": "generation_error",
        "judge_reason": "检索源包含AOP综述与Catalytic ozonation等相关文献，但模型仅给出零星推断而未系统提炼能耗、放大、催化剂稳定性、副产物控制等典型工程挑战。",
    },
    "q013": {
        "faithfulness_score": 5,
        "completeness_score": 4,
        "citation_score": 4,
        "overall_score": 4,
        "error_type": "incomplete_answer",
        "judge_reason": "光利用、稳定性、矿化、规模化等局限覆盖较全且引用合理，但电子-空穴复合和真实水体基质干扰这两个机理性局限未明确提及。",
    },
    "q014": {
        "faithfulness_score": 5,
        "completeness_score": 5,
        "citation_score": 5,
        "overall_score": 5,
        "error_type": "none",
        "judge_reason": "精准命中·OH惰性与C-F键稳定性两个核心原因，引用规范，还补充了SO₄·⁻基非常规AOP的替代路径，可直接使用。",
    },
    "q015": {
        "faithfulness_score": 4,
        "completeness_score": 2,
        "citation_score": 4,
        "overall_score": 2,
        "error_type": "retrieval_error",
        "judge_reason": "光催化与PMS对比清晰且引用合理，但臭氧适用场景完全缺失，三者比较只完成2/3，是检索端未召回臭氧专题文献所致。",
    },
}

ERROR_TYPES = [
    "retrieval_error",
    "generation_error",
    "citation_error",
    "incomplete_answer",
    "insufficient_context",
    "none",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_questions_from_csv() -> dict[str, str]:
    """Pull question text per id from the source CSV."""
    qmap: dict[str, str] = {}
    with SRC_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qmap[row["id"]] = (row.get("question") or "").strip()
    return qmap


def build_judge_section(qid: str) -> str:
    ev = EVALUATIONS[qid]
    return (
        "### Judge Evaluation\n"
        "\n"
        f"_Judge: {JUDGE_NAME}_\n"
        "\n"
        "| 维度 | 评分 (1-5) |\n"
        "| --- | --- |\n"
        f"| faithfulness_score | {ev['faithfulness_score']} |\n"
        f"| completeness_score | {ev['completeness_score']} |\n"
        f"| citation_score | {ev['citation_score']} |\n"
        f"| overall_score | **{ev['overall_score']}** |\n"
        "\n"
        f"- **error_type**: `{ev['error_type']}`\n"
        f"- **judge_reason**: {ev['judge_reason']}\n"
        "\n"
    )


QID_RE = re.compile(r"^## (q\d{3})\b", re.MULTILINE)


def write_report_md() -> None:
    text = SRC_MD.read_text(encoding="utf-8")
    matches = list(QID_RE.finditer(text))
    if not matches:
        raise SystemExit("No question sections found in source markdown.")

    parts: list[str] = [text[: matches[0].start()]]
    for i, m in enumerate(matches):
        qid = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        sep_idx = block.rfind("\n---")
        if sep_idx == -1:
            new_block = block.rstrip() + "\n\n" + build_judge_section(qid) + "\n"
        else:
            before_sep = block[:sep_idx].rstrip()
            after_sep = block[sep_idx:]
            new_block = (
                before_sep
                + "\n\n"
                + build_judge_section(qid)
                + after_sep
                + ("\n" if not after_sep.endswith("\n") else "")
            )
        parts.append(new_block)

    f_scores = [e["faithfulness_score"] for e in EVALUATIONS.values()]
    c_scores = [e["completeness_score"] for e in EVALUATIONS.values()]
    ci_scores = [e["citation_score"] for e in EVALUATIONS.values()]
    o_scores = [e["overall_score"] for e in EVALUATIONS.values()]
    avg = lambda xs: round(mean(xs), 2)

    badcases = sorted(
        [(qid, ev) for qid, ev in EVALUATIONS.items() if ev["overall_score"] < 4],
        key=lambda kv: (kv[1]["overall_score"], kv[0]),
    )

    err_count: dict[str, int] = defaultdict(int)
    for ev in EVALUATIONS.values():
        err_count[ev["error_type"]] += 1

    summary = [
        "",
        "## Summary",
        "",
        f"- **样本数**：{len(EVALUATIONS)}",
        f"- **Judge**：{JUDGE_NAME}",
        "",
        "### 平均分",
        "",
        "| 维度 | 平均分 (1-5) |",
        "| --- | --- |",
        f"| faithfulness_score | {avg(f_scores)} |",
        f"| completeness_score | {avg(c_scores)} |",
        f"| citation_score | {avg(ci_scores)} |",
        f"| overall_score | **{avg(o_scores)}** |",
        "",
        "### error_type 分布",
        "",
        "| error_type | 计数 |",
        "| --- | --- |",
    ]
    for k in sorted(err_count):
        summary.append(f"| {k} | {err_count[k]} |")
    summary.append("")

    summary += [
        f"### Badcases（overall_score < 4，共 {len(badcases)} 条）",
        "",
        "| qid | overall | faithfulness | completeness | citation | error_type | 一句话原因 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for qid, ev in badcases:
        summary.append(
            f"| {qid} | {ev['overall_score']} | {ev['faithfulness_score']} "
            f"| {ev['completeness_score']} | {ev['citation_score']} "
            f"| `{ev['error_type']}` | {ev['judge_reason']} |"
        )
    summary.append("")

    parts.append("\n".join(summary))
    REPORT_MD.write_text("".join(parts), encoding="utf-8")


def write_summary_csv() -> None:
    qmap = load_questions_from_csv()
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "question",
                "faithfulness_score",
                "completeness_score",
                "citation_score",
                "overall_score",
                "error_type",
                "judge_reason",
            ]
        )
        for qid in sorted(EVALUATIONS.keys()):
            ev = EVALUATIONS[qid]
            writer.writerow(
                [
                    qid,
                    qmap.get(qid, ""),
                    ev["faithfulness_score"],
                    ev["completeness_score"],
                    ev["citation_score"],
                    ev["overall_score"],
                    ev["error_type"],
                    ev["judge_reason"],
                ]
            )


def write_badcase_analysis() -> None:
    qmap = load_questions_from_csv()

    # All qids per error_type (badcase = overall < 4)
    badcases_by_type: dict[str, list[str]] = defaultdict(list)
    all_by_type: dict[str, list[str]] = defaultdict(list)
    for qid, ev in EVALUATIONS.items():
        all_by_type[ev["error_type"]].append(qid)
        if ev["overall_score"] < 4:
            badcases_by_type[ev["error_type"]].append(qid)
    for v in badcases_by_type.values():
        v.sort()
    for v in all_by_type.values():
        v.sort()

    total_bad = sum(len(v) for v in badcases_by_type.values())

    lines: list[str] = []
    lines += [
        "# Badcase 分析报告",
        "",
        f"Judge：**{JUDGE_NAME}** ｜ 样本数：15 ｜ Badcase 判定：`overall_score < 4`",
        "",
        f"本批共 **{total_bad}** 个 badcase，分布如下：",
        "",
        "| error_type | badcase 数 | 涉及 qid | 全样本总数（含非 badcase） |",
        "| --- | --- | --- | --- |",
    ]
    for et in ERROR_TYPES:
        bc = badcases_by_type.get(et, [])
        allc = all_by_type.get(et, [])
        lines.append(
            f"| `{et}` | {len(bc)} | {', '.join(bc) if bc else '—'} | {len(allc)} |"
        )
    lines += ["", "---", ""]

    def fmt_case(qid: str) -> str:
        ev = EVALUATIONS[qid]
        q = qmap.get(qid, "")
        return (
            f"#### {qid}：{q}\n"
            f"- 评分：overall={ev['overall_score']} ｜ faithfulness={ev['faithfulness_score']} "
            f"｜ completeness={ev['completeness_score']} ｜ citation={ev['citation_score']}\n"
            f"- judge_reason：{ev['judge_reason']}\n"
        )

    # ---------- retrieval_error ----------
    lines += [
        "## 1. retrieval_error（检索端问题）",
        "",
        "**定义**：检索器未召回回答所需的关键文档/页面，生成端\"巧妇难为无米之炊\"。问题不在生成端，而在向量检索的召回率或多样性。",
        "",
        "### Badcase",
        "",
    ]
    bc = badcases_by_type.get("retrieval_error", [])
    if bc:
        for qid in bc:
            lines.append(fmt_case(qid))
            lines.append("")
    else:
        lines.append("_本批无 badcase。_")
        lines.append("")
    lines += [
        "### 根因分析",
        "- 多对象比较类问题（A vs B vs C），单次 query 容易把某一对象挤出 top-k。",
        "- 用户 query 的嵌入更靠近高频技术（PMS、光催化）的文献簇，低频对象（如本批中的\"臭氧\"专题）相似度被压低。",
        "",
        "### 优化建议",
        "1. **Query 分解**：对比较类问题先 rewrite 成 N 个子 query（每个对象单独查），合并 top-N。",
        "2. **Per-aspect 配额检索**：识别 query 里的并列对象，对每个对象保留固定配额（如各 4 条）。",
        "3. **混合检索**：向量召回 + BM25/关键字过滤，强制保证关键术语（\"臭氧\"/\"ozonation\"）出现在结果文件名或前几页。",
        "",
        "---",
        "",
    ]

    # ---------- generation_error ----------
    lines += [
        "## 2. generation_error（生成端问题）",
        "",
        "**定义**：检索源已提供足够信息，但生成端因过度保守、不善归纳综述或拒答倾向过强而未充分作答。检索 OK，问题在生成。",
        "",
        "### Badcase",
        "",
    ]
    bc = badcases_by_type.get("generation_error", [])
    if bc:
        for qid in bc:
            lines.append(fmt_case(qid))
            lines.append("")
    else:
        lines.append("_本批无 badcase。_")
        lines.append("")
    lines += [
        "### 根因分析",
        "- 生成 prompt 里\"不确定就说不知道\"的安全约束过强，对概括类问题倾向判为\"上下文不足\"。",
        "- 模型不擅长把综述（review paper）里的多段散落讨论提炼成结构化要点清单。",
        "- chunk 切分可能过短，模型看到的是片段而非完整章节，难以归纳工程化层面的系统结论。",
        "",
        "### 优化建议",
        "1. **调整 prompt**：明确\"当 ≥3 条 source 的 distance < 0.65 时，禁止直接拒答；至少给出基于源的 3-5 条结构化要点\"。",
        "2. **同文档相邻 chunk 聚合**：rerank 后按 doc 合并连续 chunk 再入上下文，方便模型抓住整体脉络。",
        "3. **few-shot 示例**：在 prompt 中加 1-2 个\"从综述源里提炼工程挑战 / 机理要点\"的示例。",
        "4. **针对 q004 这类完全拒答**：可加 self-check 后处理，命中\"完全拒答 + 高相关度检索\"时强制重试。",
        "",
        "---",
        "",
    ]

    # ---------- citation_error ----------
    lines += [
        "## 3. citation_error（引用错误）",
        "",
        "**定义**：引用与所支持的结论不匹配（张冠李戴），或引用了 [Sx] 范围外、甚至不存在的编号。",
        "",
        "### Badcase",
        "",
    ]
    bc = badcases_by_type.get("citation_error", [])
    if bc:
        for qid in bc:
            lines.append(fmt_case(qid))
            lines.append("")
    else:
        lines += [
            "_本批 15 题中未出现典型 citation_error，citation_score 平均 4.07。_",
            "",
            "_潜在风险：q013 答案里出现「引用文献55专门论述...」这种推测式表达，属于把 source 内部的二级引用编号当成自己引用源，是轻微的引用编造苗头。_",
            "",
        ]
    lines += [
        "### 根因分析（针对潜在风险）",
        "- 模型偶尔会引用源文档内部出现的二级文献编号，把它误当作自己的 [Sx]。",
        "- 长答案中后段引用与前段证据可能错位。",
        "",
        "### 优化建议",
        '1. **prompt 约束**：明确"只能引用本次检索给出的 [S1]…[Sn]，不要引用 source 内部出现的文献编号"。',
        "2. **后处理校验**：扫描答案中所有 [Sx]，确认 x 在本次 retrieved 范围内；超出范围则标红或截断。",
        "3. **细粒度归因**：rerank 后给每条 source 标注关键句，引导模型把结论挂到具体句而非整篇 source。",
        "",
        "---",
        "",
    ]

    # ---------- incomplete_answer ----------
    lines += [
        "## 4. incomplete_answer（回答不完整）",
        "",
        "**定义**：方向正确、内容真实，但相对 ideal answer 漏掉了若干关键要点（机理、影响因素、对比维度等）。是本批最高发的错误类型。",
        "",
        "### Badcase（overall < 4）",
        "",
    ]
    bc = badcases_by_type.get("incomplete_answer", [])
    if bc:
        for qid in bc:
            lines.append(fmt_case(qid))
            lines.append("")
    else:
        lines.append("_本批无 badcase。_")
        lines.append("")
    non_bad = [
        qid
        for qid in all_by_type.get("incomplete_answer", [])
        if EVALUATIONS[qid]["overall_score"] >= 4
    ]
    if non_bad:
        lines += [
            f"### 非 badcase 但仍属轻微遗漏（overall=4，共 {len(non_bad)} 条）",
            "",
            f"{', '.join(non_bad)} —— 这些回答整体可用，但每个都漏了 1-2 个 ideal answer 中的要点（如 ¹O₂、O₂·⁻、直接电子转移、电子-空穴复合、综合毒性评估方法论等）。",
            "",
        ]
    lines += [
        "### 根因分析",
        "- 模型偏好\"基于检索源做保守归纳\"，对于 ideal answer 中需要\"领域常识 + 综述抽取\"组合的要点，只覆盖直接命中部分。",
        "- Ideal answer 通常要求穷举要点；RAG top-k 检索时，罕见要点散落在 top-5~10 容易被剪枝。",
        "- 综述源往往把多个要点压缩在一段，模型容易只抓一两个最显眼的。",
        "",
        "### 优化建议",
        "1. **扩大检索 + rerank**：先粗排 top-20，rerank 后取 top-8 进入上下文，提高罕见要点的曝光率。",
        '2. **结构化 prompt**：对 mechanism / comparison 类问题，提示模型"请覆盖：主要活性物种、反应路径、影响因素、典型条件、对比维度"等参考清单。',
        "3. **多 query 检索**：对一个问题自动生成 2-3 个变体 query（聚焦不同子要点），合并 retrieved sources。",
        "4. **回答自查**：在生成后让模型对照\"参考清单\"做一次 self-check，主动补全遗漏要点。",
        "",
        "---",
        "",
    ]

    # ---------- insufficient_context ----------
    lines += [
        "## 5. insufficient_context（上下文确实不足）",
        "",
        "**定义**：与 generation_error 的区别——这里是检索源中**确实**缺少回答某部分所需材料，模型坦诚拒答是合理的，但检索端/语料端需要补救。",
        "",
        "### Badcase",
        "",
    ]
    bc = badcases_by_type.get("insufficient_context", [])
    if bc:
        for qid in bc:
            lines.append(fmt_case(qid))
            lines.append("")
    else:
        lines.append("_本批无 badcase。_")
        lines.append("")
    lines += [
        "### 根因分析",
        "- 知识库本身可能就缺少某一子主题的专题文献（如本批中专门讨论吸附法处理 ECs 的综述）。",
        "- 比较类问题中，向量召回会偏向出现频次更高的技术簇，使弱势对象被完全排除。",
        "",
        "### 优化建议",
        "1. **扩充语料**：核对 data/ 目录是否缺少相关专题（吸附法、臭氧专题等），补充 1-3 篇综述。",
        "2. **Per-aspect 检索**（与 retrieval_error 共用）：把\"A vs B\"类 query 拆成独立检索，强制每方都召回若干文献。",
        "3. **召回侧 boost**：query 包含明确并列对象时，用 metadata filter 或 sub-query 对每个对象都查一次。",
        "4. **拒答 fallback**：当某一子部分 retrieved 为空时，把\"该部分检索不足\"明确写入答案末尾的\"局限说明\"，避免对比类问题被误判为\"答了一半\"。",
        "",
        "---",
        "",
    ]

    # ---------- Priority table ----------
    lines += [
        "## 优化优先级总览",
        "",
        "| 优先级 | 工程动作 | 直接收益（badcase） |",
        "| --- | --- | --- |",
        "| P0 | 调整生成 prompt：检索源 ≥3 条且 distance<0.65 时禁止直接拒答 | q004, q012 |",
        "| P0 | 比较类问题：query 分解 + per-aspect 配额检索 | q007, q015 |",
        "| P1 | 扩大粗排 k 后 rerank 取 top-8；同文档相邻 chunk 聚合 | 所有 incomplete_answer（q001/q002/q005/q008/q009/q010/q011/q013） |",
        "| P1 | mechanism / comparison 类问题加结构化清单提示 + self-check | q005, q009 及多数 incomplete_answer |",
        "| P2 | 补充吸附法 / 臭氧专题文献到知识库 | q007（根治）、q015（部分缓解） |",
        "| P2 | 答案后处理：校验 [Sx] 在 retrieved 范围内 + 禁用 source 内部二级编号 | 预防 citation_error |",
        "",
    ]

    BADCASE_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    write_report_md()
    write_summary_csv()
    write_badcase_analysis()
    print(f"Wrote {REPORT_MD.name}, {SUMMARY_CSV.name}, {BADCASE_MD.name}")


if __name__ == "__main__":
    main()
