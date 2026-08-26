# Release Smoke Test

这是冻结 v1.0.0 演示的发布核对清单；它不会重建索引，也不会运行 acceptance benchmark。

## Runtime identity

启动 API 与 Streamlit UI 后，`/health` 应报告：

- `application_version`：`1.0.0`
- `build_id`：`v1.0.0-final`
- knowledge base：270 PDFs，路径为 `data/papers/final_corpus`
- dense collection：`section_aware_270_gpu`，17,028 chunks
- sparse index：`ready`，17,028 chunks
- retrieval：`section_hybrid`、BGE-M3 Dense + Sparse、RRF、Section-aware

## 两项 live checks

1. 提交一个可直接回答的 MB 吸附容量问题。确认回答包含有文献支持的 `34.64 mg/g`、
   至少一个有效 `[Sx]` citation，以及可展开的 paper/page/section provenance。source ordinal
   会随检索 context 排序变化，但必须能映射回页面展示的 context card。
2. 提交 “What year did the French Revolution begin?”。确认返回 evidence-insufficiency，且不
   编造 citation。

## 截图核对

仓库中的图片均从最终 runtime 的真实 UI 直接捕获：

- `docs/assets/demo-answer.png`：可见 question、answer 与 citation。
- `docs/assets/demo-evidence.png`：展开的 source card 展示 paper、page、section 与 chunk provenance。
- `docs/assets/demo-refusal.png`：条件不足的问题收到 clarification/refusal，而不是无依据回答。

不要使用 mock data、image generation 或手工伪造的 UI 截图。
