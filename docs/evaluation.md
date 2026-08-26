# 评测说明

## Final Acceptance 方法

v1.0.0 在一套冻结、fresh 的 32-case acceptance set 上评测一次。其 gold-paper IDs 与
Eval V2 paper-disjoint。所有带证据的 case 均在冻结的本地 PDF 中重新定位，验证结果为
**0 errors**。

- 类别分布：ANSWERABLE 14、AMBIGUOUS 4、NO_EVIDENCE 5、PARTIAL_EVIDENCE 3、
  FALSE_PREMISE 4、CONDITIONALLY_DIVERGENT 2。
- Fresh gold-paper IDs：9。
- 冻结 acceptance 输入 SHA-256：
  `a1cc901ad8cda834325f073837385e2daf4d488a28e8a9ebc77fd134a2c2d9e5`。
- 冻结 runtime：`section_aware_270_gpu` + BGE-M3 Dense/Sparse + RRF
  （`section_hybrid`），搭配最终的 Answerability 与 Citation Validation pipeline。

## 预先固定的发布门槛与结果

| 指标 | 门槛 | 结果 | 状态 |
| --- | ---: | ---: | --- |
| Recall@10 | ≥ 70.0% | 91.3% | pass |
| PageHit@10 | ≥ 55.0% | 91.3% | pass |
| EvidenceSpanHit@10 | ≥ 50.0% | 87.0% | pass |
| Action Accuracy | ≥ 60.0% | 75.0% | pass |
| ANSWERABLE Accuracy | ≥ 55.0% | 71.4% | pass |
| NO_EVIDENCE Recall | ≥ 80.0% | 100% | pass |

Verdict：**FINAL_ACCEPT_WITH_LIMITATIONS**。

## Citation 与 Unsupported Claim 的解读

Automatic Citation Support 为 61.9%，automatic Unsupported Claim metric 为
42.86%。这些是保守 validator 的输出，**不是 hallucination rate**。它们会受到
quick-mode 1,200-token truncation、bibliographic-claim detection 与 citation
normalization 的影响。

evidence-level audit 为 10 SUPPORTED / 1 flagged UNSUPPORTED。唯一 flag 为
`34.64 mg/g` 与 `34.64mg/g` 的空白 normalization artifact。发布版本不宣称 zero
hallucination，也不宣称完美 citation support。

## 范围与结论解释

Eval V2 的 held-out 结果在发布前已被消费。之后的 V2 运行仅是 regression/postmortem
材料，不构成新的泛化结论。发布证据以这套 final acceptance set 为准；该仓库不公开论文
正文或 gold evidence。

conflict-routing postmortem 是一次工程决策：keyword-based auto-routing 得到
CONFLICT_OFF 6 wins / 1 loss，因此删除了不可靠 automatic classifier。条件差异由回答层
结合 provenance 呈现，而不是被强制压缩成二元 conflict action。

参阅[项目概览](../README.md)与明确的[系统局限性](../LIMITATIONS.md)。
