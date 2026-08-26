# 系统范围与局限性

本文说明水处理科研文献 RAG 系统有意设定的边界；它们是当前冻结版本的能力范围，
而非未记录的缺陷。

1. **Corpus**：冻结的 270 篇主 PDF（`data/papers/final_corpus`）。
2. **覆盖范围**：可机器提取文本的 PDF 正文；没有 text layer 的扫描页或图片页不在覆盖范围。
3. **Supplementary files**：每篇论文独立的 Supplementary Information 不纳入索引。
4. **Tabular supplementary data**：不解析 XLSX/CSV supplementary datasets。
5. **PDF tables**：文本提取可能丢失行列结构，导致表格中的精确数值证据无法可靠恢复。
6. **Figure-only evidence**：当前 text-RAG 不覆盖这类证据；没有 OCR、image model 或 visual grounding。
7. **References section**：为避免 citation leakage，References 被有意排除出检索索引；引用句本身不等于正文证据。
8. **Automatic conflict classification**：已移除。旧的 keyword-based increase/decrease heuristic 会在
   held-out 数据上系统性误判普通 ANSWERABLE 文本；可靠识别需要 claim/condition alignment，
   不在冻结范围内。
9. **Condition-dependent / source-disagreement**：由回答层处理。generation prompt 要求按来源、指标与
   实验条件分别呈现，不能强行给出单一结论；不再使用预分类的 conflict action。
10. **Eval V2 held-out behavior TEST 已消费**：post-freeze V2 数字仅用于 regression/postmortem，
    不再被表述为 held-out generalization result。

RAG pipeline：270 PDFs → Section-aware ingestion → BGE-M3 Dense + Sparse + RRF
(`section_hybrid`) → evidence-aware Answerability（ANSWER / CLARIFY / REFUSE /
PARTIAL_ANSWER / CORRECT_PREMISE）→ traceable citations。

发布方法与预固定 acceptance gates 见 [docs/evaluation.md](docs/evaluation.md)。
