# RAG 准确性、证据一致性与 8010 进程治理报告

报告日期：2026-08-07。项目：`literature-rag-eval-code`。本轮未更换 BGE-M3、ChromaDB 或 DeepSeek 配置，未修改 66 个 PDF，未重建 `chroma_db`，默认 Top-K 保持为 5。

## 1. 当前 Prompt 文件和调用位置

完整逐项清单见 [`prompt_inventory.md`](prompt_inventory.md)。正式模板集中在 `src/prompts.py`：普通 RAG 由 `src/rag_chain.py` 调用，直接 LLM 由 `api_server.py` 的显式 `/llm/chat` 调用，Judge 仅由 `src/eval/runner.py` 与兼容入口 `src/eval_answer_judge.py` 调用。Query Rewrite、fallback、引用校验和七类错误归因均为确定性代码，不调用模型。

## 2. 原 Prompt 的主要问题

- 普通回答是一个固定、偏刚性的长模板，没有明确区分快速与详细回答。
- 事实边界、冲突证据、无答案、精确数字、文档内指令不可信和引用映射规则没有统一管理。
- 直接 LLM 没有独立的集中系统约束。
- Judge 模板存在重复实现，评分维度未明确覆盖 correctness 与 evidence relevance。
- Prompt 缺少稳定版本和内容哈希，运行结果难以区分 A/B 版本。

## 3. 新 Prompt 的核心变化

- 所有正式 Prompt 集中到 `src/prompts.py`，回答与 Judge 分别版本化。
- 只允许依据检索片段作答；不得编造作者、标题、页码、实验条件、数字或结论。
- 明确区分无证据与证据冲突；冲突时并列来源，不强行给唯一结论。
- 文档内容被标记为不可信数据，片段中的指令、角色或示例不得覆盖系统规则。
- 关键结论必须使用真实 `[Sx]`；来源列表只能引用本次实际检索片段。
- 不泄露系统 Prompt、内部策略、密钥或配置，不要求展示思维链。
- 相同证据约束下提供快速与详细两种深度。

## 4. Prompt 版本

- 回答：`rag_answer_prompt_v2`。
- Judge：`rag_judge_prompt_v2`。
- 当前应用：`2.0.0`。
- 当前构建：`20260807-rag-accuracy-v2`。
- 每种回答模式有独立 `prompt_hash`；版本与哈希写入 API 结果、Trace 和评测配置。

## 5. 快速与详细模式

| 项目 | 快速（默认） | 详细 |
| --- | --- | --- |
| 目标 | 直接结论、3–5 个主要点、必要限制 | 多来源综合、差异条件、机制与研究边界 |
| 事实/引用规则 | 与详细模式完全相同 | 与快速模式完全相同 |
| `max_tokens` | 1,200 | 2,200 |
| UI/API 值 | `quick` | `detailed` |

两种模式只改变深度和篇幅，不放宽证据要求。页面明确显示当前模式，原非流式 `/chat` 仍保留。

## 6. 引用校验实现方式

`src/citation_validation.py` 在生成后执行确定性检查，不增加 LLM 调用：

- `[Sx]` 必须落在本次 context 范围内；越界编号会被移除并标为 corrected。
- 页码必须出现在检索 metadata 中；无法核验的页码会被替换为显式警告。
- 声称“文献/研究证明”却没有引用的句子会被移除。
- 可疑作者/标题格式、完全无可映射引用、长度截断被标为 failed。
- 返回 used/unused/invalid source IDs、修正状态和警告，并写入 Trace。
- 流式阶段暂不展示 `[Sx]`，最终事件用校验后的完整回答覆盖临时文本，避免提前展示伪造引用。

## 7. 无答案、冲突证据和模糊问题

- 安全/注入、伪造引用、明显歧义由规则前置处理，返回说明且不调用 LLM。
- 最佳距离超过阈值，或问题要求精确数值而片段没有数值/单位证据时，返回“当前知识库中没有足够证据回答这个问题”。
- 冲突检测要求同一指标在不同来源中出现相反方向，减少“提高/降低”泛词造成的假冲突；Prompt 要求列出差异来源与可能条件，不输出唯一强结论。
- 简单边界由规则处理，没有新增自动问题分类模型。

## 8. 是否增加额外 LLM 调用

没有。普通成功问答仍是一次主要 DeepSeek 生成；确定性 fallback 为 0 次。Query Rewrite、引用校验、错误归因均为 0 次。直接 LLM 与 Judge 只有在用户显式开启对照或评测时才执行，并会在 Trace/页面计数中体现。

## 9. 修改文件

本轮准确性与进程治理的核心文件：

- `src/prompts.py`、`src/citation_validation.py`、`src/rag_chain.py`、`src/retriever.py`、`src/llm_client.py`。
- `api_server.py`、`app.py`。
- `src/tracing/schema.py`、`src/tracing/instrumentation.py`。
- `src/eval/runner.py`、`src/eval/schema.py`、`src/eval/io.py`、`src/eval/export.py`、`src/eval/cli.py`、`src/eval_answer_judge.py`。
- `src/build_info.py`、`src/status.py`、`build_manifest.json`、`launcher.pyw`、`.env.example`。
- `tests/test_prompt_and_citations.py`、`tests/test_launcher_process_safety.py`、`scripts/benchmark_answer_modes.py`。
- 本报告、Prompt 清单和 README。

此前同一工作区内的启动器、性能、缓存、预热与 API 契约改动仍保留；本轮没有修改 `data/pdfs` 或 `chroma_db`。

## 10. 自动测试

最终自动测试：`71 passed, 1 warning in 3.32s`。原 52 项全部继续通过，并新增 19 项覆盖：明确问题、跨 chunk 综合、库内无答案、精确数值无证据、冲突证据、模糊问题、提示注入、伪造引用请求、快速/详细模式、引用越界、流式最终映射、外部端口占用、PID 文件缺失、旧 build、停止后端口释放等。唯一 warning 为既有 Starlette/httpx 兼容弃用提示，不影响结果。

## 11. 两次真实 DeepSeek 测试

按约束只执行两次，问题均为“PFAS 水处理工程化面临哪些主要限制？”，Top-K=5、真实 SSE 流式接口、同一索引；无重试。脚本不打印/保存答案正文、文档内容或密钥。

| 指标 | 快速 | 详细 |
| --- | ---: | ---: |
| 回答字符数 | 375 | 1,386 |
| context 数 | 5 | 5 |
| 使用的有效来源 | 2 | 3 |
| 越界引用 | 0 | 0 |
| 引用校验 | passed | passed |
| LLM 调用 / 重试 | 1 / 0 | 1 / 0 |
| Prompt tokens（API） | 1,740 | 1,752 |
| 服务端 TTFT | 13.841 s | 29.846 s |
| 客户端首字 | 13.944 s | 29.850 s |
| LLM 完整生成 | 20.880 s | 60.786 s |
| 服务端总耗时 | 20.957 s | 60.789 s |

快速模式篇幅明显更短；详细模式引用更多来源且展开更充分。两次结果均通过确定性引用映射，无越界或警告。两次样本只能验证接口、模式差异、引用结构和实际延迟，不能据此宣称语义准确率或 SLA 已稳定；长期质量仍应由固定 ground truth + 独立 Judge/人工复核评估。

## 12. 8010 原占用进程

- API 根进程 PID `21704`，监听子进程 PID `25944`。
- 命令行为项目 `.venv\\Scripts\\python.exe -m uvicorn api_server:app --host 127.0.0.1 --port 8010`。
- 启动器日志记录开始时间 `2026-08-07 01:14:43 +08:00`。
- UI 根进程 PID `36244`，监听子进程 PID `15464`，命令行为同一项目虚拟环境中的 Streamlit 8501。
- PID 文件、可执行路径、命令行、父子进程树和项目工作目录均确认属于 `literature_rag_eval_code`；没有以“端口是 8010”为唯一依据。

## 13. 旧进程清理结果

旧 UI/API 进程树分别于 `02:43:05` 安全停止，8010/8501 均经轮询确认释放。一次预热期间的 Hugging Face 网络探测导致启动超时，启动器按事务回滚了当次新进程，没有留下监听；随后将 Production 默认设为 BGE-M3 本地缓存加载，未更换模型。最终旧构建再次被识别并替换，未终止任何无关 Python/Uvicorn/Streamlit 进程。

## 14. 当前 8010 身份

最终 `/health`：

- `project_id=literature-rag-eval-code`
- `application_version=2.0.0`
- `build_id=20260807-rag-accuracy-v2`
- `prompt_version=rag_answer_prompt_v2`
- `top_k_default=5`
- `prewarmed=true`
- `document_count=66`
- `chunk_count=7337`
- API `process_id=796`，启动时间 `2026-08-07T02:49:40+08:00`

现场复核的根进程为 API PID `10052`、UI PID `6624`，均来自当前项目 `.venv`；Production 命令没有 `--reload`。

## 15. 启动器是否能识别旧版本

可以。健康身份分为 `current`、`old_project`、`legacy_project`、`foreign`、`unavailable`。只有 project/application/build/prompt 身份符合预期才显示 ready；旧构建会显示“检测到旧版本服务”，经归属验证后停止并在 8010 释放后启动当前构建。外部服务或无法验证的监听者只报端口冲突，不自动终止。

## 16. 孤儿进程风险

已显著收敛：PID 文件分别记录 API/UI 的 PID、创建时间、工作目录、命令参数、build/prompt；停止使用 Windows 进程树并轮询端口；PID 文件缺失时必须从监听 PID 沿父进程链独立验证项目命令和虚拟环境。启动失败执行回滚。现场最终只有一棵 API 和一棵 UI 树。仍需注意操作系统在异常断电或用户用任务管理器强杀根进程时可能绕过清理，因此下次启动仍会重新执行身份核验。

## 17. 推荐的面试演示设置

- 使用桌面 **Literature RAG** 启动器；展示 build、Prompt、Embedding、Chroma、66 文档、7,337 chunks 和 `prewarmed=true`。
- 默认 Top-K=5、快速回答、流式开启、对照模式关闭、Tracing 默认关闭。
- 用 PFAS 工程限制问题展示真实阶段进度、快速首答、最终 `[Sx]` 和折叠证据；需要体现综合深度时切到详细回答再运行。
- 展开性能详情说明网络 TTFT 与本地检索分开，且本次 LLM calls=1。
- 展示无答案/伪造引用请求的 0 次 LLM fallback，再按需打开独立评测看板；不要在普通问答中自动运行 Judge。
- 若现场需展示 Trace，临时开启并在演示后关闭；不要展示 `.env`、API Key、原始完整文档或大段日志。

## 18. 混合检索与重排消融实验（2026-08）

本轮在不改动现有链路的前提下，新增三种显式检索模式（`RETRIEVAL_MODE`：`dense_only` / `hybrid_dense_sparse` / `hybrid_reranker`），并用同一评测集做消融。所有数据为实测（`scripts/ablation_retrieval.py`，结果存 `outputs/retrieval_ablation.json`）。

实现要点：

- Sparse 路径复用官方 FlagEmbedding 的 BGE-M3 lexical weights（非自行复现），加载本地快照目录，`sparse_linear.pt` 头文件齐全；对 transformers 4.46.3 与 FlagEmbedding 的 `dtype` 参数不兼容做了一处单关键字转换垫片。
- 稀疏倒排索引为轻量本地实现（`sparse_index/index.npz` + `manifest.json`，约 5MB，864,813 postings / 24,325 terms），与 Chroma collection 版本绑定，语料变化后拒绝加载并要求重建。
- 融合用 RRF（k=60），重排用 `BAAI/bge-reranker-v2-m3`（CUDA）。任一前置缺失时显式报错，不静默回退到 dense。

实测结果（`groundtruth.example.jsonl`，15 题）：

| Variant | Recall@5 | Recall@10 | MRR@10 | Hit@10 | 平均延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.544 | 0.628 | 0.744 | 1.000 | ~282ms* |
| Hybrid | 0.528 | 0.594 | 0.810 | 0.933 | ~183ms* |
| Hybrid + Reranker | 0.500 | 0.556 | 0.593 | 0.867 | ~7580ms |

*平均延迟含首题的模型/索引冷启动（dense 首题约 3.9s、hybrid 首题约 2.0s）；稳态单题延迟：dense 约 20ms，hybrid 约 50ms，rerank 阶段约 7.5s（25 个候选、fp32、512 长度）。

结论：

1. **Hybrid 值得保留为可选模式**：唯一稳定的正向信号是 MRR 0.744 → 0.810（正确片段排位提前，如 q011 从 rank3 升到 rank1、q015 从 hit@5 miss 变为 rank2 命中）；recall 与 hit@10 的差异方向相反且幅度小，15 题样本不足以判定优劣，不做过拟合调参。
2. **Reranker 当前配置为净损失，不建议启用**：全指标下降且延迟增加约 26 倍。结构性原因是 `final_k=8` 的重排结果再经过 `max_per_source=2` 的来源去重，多来源问题（expected_sources≥3）的最优上下文被去重砍掉（q006 最终只剩 2 个片段）；且融合候选仅 25 条，上游漏检（如 q008 的 toxicity 文献）重排无法补救。个别题上重排确有修复作用（q013 把 rank8 的正确文献提到 rank3），说明问题在配置与候选量，不在模型本身。
3. 评测集金标准是文档名模式匹配而非片段级标注，recall 上限受 expected_sources 数量影响，解释结果时需注意该口径限制。

下一阶段待办（本轮不做）：评测集扩容并加片段级金标准；section-aware chunking（q013/q015 的漏检与字符级切片跨 section 有关）；重排与来源去重的交互修正后重测。

## 结论

本轮完成了 Prompt 中央化与版本化、快速/详细模式、无答案/冲突/歧义边界、确定性引用校验、Judge 隔离与升级，以及 8010 的版本身份和安全进程树治理。没有通过增加模型调用换取质量，也没有重建索引或牺牲引用真实性。当前实际主要等待仍来自外部 DeepSeek TTFT/生成时间，不能由本地 Prompt/检索代码消除。
