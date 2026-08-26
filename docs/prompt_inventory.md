# Prompt 与模型调用清单

本清单以最终实际发送给模型的 `messages` 为准，而不是只搜索含有 “prompt” 字样的字符串。正式回答 Prompt 统一位于 `src/prompts.py`，版本为 `rag_answer_prompt_v2`。

| 场景 | 文件与入口 | 触发时机 | 普通 RAG 路径 | 额外 LLM 调用 | 原问题 | 当前处理 |
| --- | --- | --- | --- | --- | --- | --- |
| 普通 RAG 回答 | `src/prompts.py` 的 `build_answer_system_prompt` / `build_answer_user_prompt`；`src/rag_chain.py` 调用 | `/chat` 或 `/chat/stream` 且检索证据足够 | 是 | 仅一次主要生成 | 模板单一、证据边界和冲突/无答案规则不够集中 | 已集中、版本化，并按快速/详细模式控制篇幅 |
| 直接 LLM | `src/prompts.py` 的 `DIRECT_LLM_SYSTEM_PROMPT`；`api_server.py::llm_chat` | 用户显式开启对照或直接调用 `/llm/chat` | 否 | 是，一次 | 原链路直接发送用户内容，约束分散 | 已增加独立系统约束，并与 RAG 明确隔离 |
| Query Rewrite | `src/retriever.py::rewrite_query` | 普通检索前的双语规则扩展 | 是 | 否 | 容易被误认为隐藏模型调用 | 保持规则实现；Trace 记录改写文本与阶段耗时 |
| LLM Judge | `src/prompts.py` 的 `build_judge_messages`；`src/eval/runner.py`、`src/eval_answer_judge.py` | 显式执行评测命令 | 否 | 是，一次/样本 | 旧模板重复，缺少 correctness 与 evidence relevance | 已统一并升级为六项评分，记录 Judge/回答 Prompt 版本 |
| 四维/扩展评测 | 与 LLM Judge 共用同一组 messages | 显式 Judge 运行 | 否 | 不额外增加第二个 Judge | 旧说明只列 faithfulness、completeness、citation、overall | 在同一次 Judge 中增加 correctness、evidence relevance |
| 七类错误归因 | `src/eval/badcase.py` | 显式 badcase/export | 否 | 否 | 无模型 Prompt；可能被误认为自动归因 | 保持确定性规则，只在评测路径运行 |
| fallback / 防注入 / 模糊问题 | `src/rag_chain.py` 的 guardrail 与 evidence 判断 | 命中安全、歧义或证据不足条件 | 是 | 否；通常为 0 次 | 无答案、精确数值和歧义场景边界不足 | 已扩展确定性规则，不增加修复模型调用 |
| 文档上传/摘要/索引 | `src/ingest.py`、PDF/Embedding 流程 | 人工受控建库 | 否 | 否 | 当前正式产品没有上传摘要 LLM | 不新增 Prompt，不改冻结的 270-PDF 语料或索引 |

## 普通问答的最终调用链

```text
用户问题
  → 规则型 Query Rewrite（0 次 LLM）
  → BGE-M3 Query Embedding
  → Chroma 检索
  → 去重、来源多样化、Token Budget
  → rag_answer_prompt_v2
  → 一次 DeepSeek 主要生成
  → 确定性引用校验（0 次 LLM）
```

Judge、直接 LLM、七类错误归因和答案修复均不会被普通问答暗中触发。证据不足、提示注入、伪造引用请求和需要补充条件的歧义问题会在主要生成前返回，因此 LLM 调用次数为 0。

## Prompt 版本与可复现性

- 回答 Prompt：`rag_answer_prompt_v2`。
- Judge Prompt：`rag_judge_prompt_v2`。
- 快速与详细回答共享事实边界，分别生成不同 `prompt_hash`，便于 A/B 区分。
- API 结果、Trace、评测 run config 和 `/health` 均记录相应版本。
- Prompt 文件不包含 API Key、Authorization Header、`.env` 内容或内部思维链要求。
