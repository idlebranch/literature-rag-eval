"""Badcase analysis: group by error_type and propose fixes."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.eval.schema import EvalRun, QuestionResult


ERROR_GUIDE: Dict[str, dict] = {
    "retrieval_error": {
        "definition": "检索器未召回回答所需的关键文档/页面，生成端\"巧妇难为无米之炊\"。",
        "fixes": [
            "对比较类问题做 query rewrite，把 \"A vs B vs C\" 拆成多个子 query 分别检索后合并。",
            "Per-aspect 配额检索：识别 query 中的并列对象，对每个对象保留固定召回配额。",
            "向量召回 + BM25/关键字混合检索，保证关键术语出现在前几页结果。",
        ],
    },
    "generation_error": {
        "definition": "检索源已提供足够信息，但生成端因过度保守、不善归纳综述或拒答倾向过强而未充分作答。",
        "fixes": [
            "调 prompt：检索源 ≥3 条且 distance<0.65 时禁止直接拒答，至少给出 3-5 条结构化要点。",
            "同文档相邻 chunk 聚合后再入上下文，便于模型抓住综述章节脉络。",
            "在 prompt 加 few-shot 示例：从综述里提炼工程挑战 / 机理要点。",
        ],
    },
    "citation_error": {
        "definition": "引用与所支持的结论不匹配（张冠李戴），或引用了 [Sx] 范围外的编号。",
        "fixes": [
            "Prompt 约束：只能引用本次 context 中实际给出的 [Sx]，禁用 source 内部的二级文献编号。",
            "后处理校验：扫描答案中的 [Sx]，确认 x 在 retrieved 范围内。",
            "Rerank 后给每条 source 标注关键句，引导模型把结论挂到具体句。",
        ],
    },
    "incomplete_answer": {
        "definition": "方向正确、内容真实，但相对 ideal answer 漏掉了若干关键要点。",
        "fixes": [
            "扩大粗排 top-20 + rerank 取 top-8，提高罕见要点的曝光率。",
            "对 mechanism / comparison 类问题，prompt 提示参考清单（活性物种、路径、影响因素、对比维度等）。",
            "多 query 检索：对一个问题自动生成 2-3 个变体 query。",
            "答案 self-check：让模型对照参考清单做一次补全。",
        ],
    },
    "insufficient_context": {
        "definition": "检索源中确实缺少回答某部分所需材料，模型拒答合理但检索/语料端需要补救。",
        "fixes": [
            "扩充语料：核对 data/ 目录是否缺少相关专题，补充 1-3 篇综述。",
            "Per-aspect 检索：把对比类 query 拆成独立检索，强制每方都召回若干文献。",
            "拒答 fallback：缺料的对象仍保留小节标题并标注 \"该部分检索不足\"。",
        ],
    },
    "hallucination": {
        "definition": "生成的事实在检索源中找不到对应，或与源中陈述矛盾。",
        "fixes": [
            "Prompt 强约束：只能基于 context 回答，禁止引入领域常识。",
            "Faithfulness 自检：生成后让模型逐句标注证据 [Sx]，无证据的句子标 (无来源) 或删除。",
            "降低 temperature 至 0；增加 chunk 上下文长度。",
        ],
    },
    "script_error": {
        "definition": "脚本运行错误（API key 失效、网络异常等），不属于模型/检索问题。",
        "fixes": [
            "运行前调用 src.eval.health 做 preflight 检查。",
            "失败时单点重试 + 指数退避，而不是 sleep 固定时间。",
            "把错误样本单独标记，避免污染分数统计。",
        ],
    },
}


def write_badcase_report(
    run: EvalRun, path: Path, badcase_threshold: int = 4
) -> Path:
    by_type_bad: Dict[str, List[QuestionResult]] = defaultdict(list)
    by_type_all: Dict[str, List[QuestionResult]] = defaultdict(list)
    for r in run.results:
        if not r.judge:
            continue
        by_type_all[r.judge.error_type].append(r)
        if r.judge.overall < badcase_threshold:
            by_type_bad[r.judge.error_type].append(r)

    total_bad = sum(len(v) for v in by_type_bad.values())

    lines: List[str] = []
    lines.append(f"# Badcase Analysis · {run.run_id}")
    lines.append("")
    lines.append(
        f"Judge: **{run.config.judge_model or '-'}** ｜ 样本数: {len(run.results)} "
        f"｜ Badcase 判定: `overall < {badcase_threshold}`"
    )
    lines.append("")
    lines.append(f"本批共 **{total_bad}** 个 badcase，分布如下：")
    lines.append("")
    lines.append("| error_type | badcase 数 | 涉及 qid | 全样本总数 |")
    lines.append("| --- | --- | --- | --- |")
    for et in ERROR_GUIDE.keys():
        bad = sorted(r.qid for r in by_type_bad.get(et, []))
        allc = by_type_all.get(et, [])
        lines.append(
            f"| `{et}` | {len(bad)} | {', '.join(bad) if bad else '—'} | {len(allc)} |"
        )
    # also include 'none' which exists in scores but not in ERROR_GUIDE
    none_count = len(by_type_all.get("none", []))
    if none_count:
        lines.append(f"| `none` | 0 | — | {none_count} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    for idx, (et, guide) in enumerate(ERROR_GUIDE.items(), start=1):
        lines.append(f"## {idx}. {et}")
        lines.append("")
        lines.append(f"**定义**：{guide['definition']}")
        lines.append("")
        bad = by_type_bad.get(et, [])
        lines.append("### Badcase")
        lines.append("")
        if bad:
            for r in sorted(bad, key=lambda x: x.qid):
                j = r.judge
                lines.append(f"#### {r.qid}：{r.question}")
                lines.append(
                    f"- 评分：overall={j.overall} ｜ faithfulness={j.faithfulness} "
                    f"｜ completeness={j.completeness} ｜ citation={j.citation}"
                )
                lines.append(f"- reason：{j.reason}")
                lines.append("")
        else:
            lines.append("_本批无 badcase。_")
            lines.append("")
        lines.append("### 优化建议")
        for i, fix in enumerate(guide["fixes"], start=1):
            lines.append(f"{i}. {fix}")
        lines.append("")
        lines.append("---")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
