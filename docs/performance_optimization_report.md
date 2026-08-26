# 历史性能诊断与优化报告（pre-freeze）

> 本文记录的是冻结前的 66-PDF Demo，不是 v1.0.0 的运行配置、性能基线或发布指标。
> 最终 270-PDF `section_hybrid` release 请以 `README.md` 和
> `docs/evaluation.md` 为准。

测试日期：2026-08-07（Asia/Shanghai）

## 结论

确认的主要瓶颈是外部 DeepSeek 推理，而不是本地检索。优化前热启动本地链路平均约 67ms，而外部 LLM 完整生成平均 34.30s；冷启动另有约 7.06–7.33s 的 BGE-M3 首次加载。原实现还会在每次请求重新创建 Chroma client/collection 与 OpenAI client，后者平均增加约 356ms。

已完成：资源单例与启动预热、连接池复用、显式有限重试、逐阶段计时、Query/检索缓存、完整 chunk 上下文预算、相邻重复去重、默认 Top-K=5、真实 SSE 流式输出、健康状态和性能 UI。未更换 ChromaDB、BGE-M3 或知识库，未重建索引，未缓存最终回答，未删除引用。

## 测试方法

- 基线：5 次相同问题、5 次不同问题、3 次相同问题流式探针，共 13 次真实 DeepSeek 调用，Top-K=8。
- 优化后：7 次相同问题（含 Top-K=4/5/8 与 3 次流式）、5 次不同问题，共 12 次真实 DeepSeek 调用。
- 测试问题包括“PFAS 水处理工程化面临哪些主要限制？”及 5 个不同的水处理问题。
- 所有阶段使用 `perf_counter`；结果只保存计时、token 和引用指标，不保存 API Key、答案正文或文档内容。
- P95 采用样本内线性插值。外部模型样本量较小，数值用于诊断和演示设置，不作为 SLA。

## 优化前后阶段统计

单位均为 ms。前后工作负载不完全相同（基线主要 Top-K=8，优化后主要 Top-K=5），因此 LLM 总耗时差异只报告观察值，不推断为稳定网络收益。

| 阶段 | 优化前平均 / P50 / P95 | 优化后平均 / P50 / P95 |
| --- | ---: | ---: |
| 请求解析 | 0.012 / 0.011 / 0.018 | 0.013 / 0.011 / 0.021 |
| 规则 Query Rewrite | 0.006 / 0.005 / 0.009 | 0.006 / 0.005 / 0.010 |
| Query Embedding（含冷启动/缓存） | 603.15 / 64.93 / 2881.61 | 32.88 / 21.40 / 82.33 |
| Chroma 检索 | 16.96 / 5.49 / 66.40 | 1.93 / 2.53 / 3.31 |
| 去重/过滤 | 0.006 / 0.005 / 0.007 | 0.105 / 0.010 / 0.384 |
| Prompt 构建 | 0.013 / 0.012 / 0.018 | 0.012 / 0.010 / 0.021 |
| LLM client 创建/准备 | 355.63 / 343.54 / 468.43 | 0.003 / 0.002 / 0.004 |
| 流式请求建立（n=3） | 157.78 / 159.56 / 164.72 | 103.81 / 107.47 / 114.62 |
| LLM TTFT（n=3） | 18260.07 / 17535.17 / 21808.13 | 28559.70 / 32519.31 / 34657.69 |
| LLM 完整生成 | 34303.36 / 33654.19 / 51005.56 | 32341.00 / 27845.16 / 47088.93 |
| 引用整理 | 0.024 / 0.016 / 0.062 | 0.005 / 0.001 / 0.016 |
| 总耗时 | 35279.19 / 34066.04 / 53005.88 | 32376.82 / 27870.70 / 47134.27 |

TTFT 优化后样本反而更慢，说明它受外部模型排队/推理波动主导。连接建立下降是真实可重复的本地客户端复用收益，但不能保证外部服务的首字时间。

## 冷启动与热启动

| 场景 | 结果 |
| --- | --- |
| 优化前冷启动首问（Top-K=8） | BGE 7059.71ms；Chroma 155.11ms；LLM client 445.44ms；LLM 41999.98ms；总计 49660.44ms |
| 优化后启动预热 | 7329.10ms；Embedding ready；Chroma ready；66 PDFs；7,337 chunks；不调用 LLM |
| 预热后首个 Top-K=5 本地检索 | Embedding 69.60ms；Chroma 2.73ms；最终 5 chunks |
| 相同 Query + Top-K=5 第二次 | Retrieval cache hit；Embedding 0ms；Chroma 0ms；仍返回同一组完整 chunks |
| 不同问题 Top-K=5（5 次） | 本地处理平均 66.71ms；Embedding 63.62ms；Chroma 2.82ms |

冷启动成本没有被伪装或消失，而是被前移到启动器可见的预热阶段，避免用户提交第一问后才等待模型加载。

## Top-K 对照

同一 PFAS 工程限制问题的真实调用结果：

| Top-K | 样本 | Prompt Tokens | 引用编号有效性 | 已检索 chunk 引用覆盖 | 五维工程限制覆盖 | 观察到的总耗时 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 1,596 | 100% | 50.0% | 100% | 46.84s |
| 5 | 5 | 1,859 | 100% | 平均 48.0% | 100% | 平均 33.31s，P50 27.41s |
| 8 | 1 | 2,684 | 100% | 62.5% | 100% | 27.84s |

默认设为 Top-K=5。与 8 相比，Prompt Tokens 减少 30.7%，本题要求的五个工程维度保持全覆盖，引用编号有效性保持 100%；相比 4，多保留一个证据片段，降低证据过窄风险。单次外部耗时不随 Prompt 单调变化，Top-K=4 的 46.84s 正好说明网络/推理波动远大于本地检索差异。

“引用覆盖”是回答实际引用的检索 chunk 比例，不等同于答案真实性分数；更多 Top-K 会机械性改变该分母。此次没有在普通问答路径运行 LLM Judge，质量检查使用结构覆盖、引用编号有效性和现有人工测试标准。

## 调用链与模型调用

普通问答现在严格为：规则扩展 → 一次 Query Embedding → 一次 Chroma 检索 → 去重/预算 → 一次主 LLM → 引用整理。

- 普通成功问答：1 次 LLM。
- guardrail、需澄清或证据不足 fallback：0 次 LLM。
- 对照模式：显式增加 1 次直接 LLM，并在页面调用数中显示。
- Judge、四维评测、七类归因：仅评测命令/看板触发。
- Query Rewrite：现有规则函数，0 次 LLM。
- 12 次优化后真实测试共 12 次 LLM，重试总数 0；每次都在性能数据中报告调用和重试次数。

## 缓存、连接与引用保护

- BGE-M3/tokenizer、Chroma client/collection、OpenAI-compatible client 在进程内复用。
- SDK 隐式重试关闭；仅连接错误、超时、429、408/409 和 5xx 可重试，最大 1 次，无限重试被禁止。
- Query Embedding cache：规范化扩展 Query，最大 128 项。
- Retrieval cache：Query + Top-K + collection version，最大 128 项；`chroma.sqlite3` 的 mtime/size 或 collection 名变化自动失效。
- 最终 LLM 回答不缓存。
- 同一 PDF 最多 2 chunks；相邻 chunk 仅在 3-gram Jaccard ≥ 0.90 时去重。
- 上下文默认预算 3,000 estimated tokens，只纳入完整 chunk；不截断句子。文件名、页码、chunk_id 始终保留。
- 12 个优化后答案的引用编号有效性均为 100%。

## 流式输出与健康状态

DeepSeek/OpenAI-compatible 接口支持可靠 streaming，已新增 `/chat/stream` SSE：Streamlit 依次显示理解、向量化、检索、找到片段、生成、整理引用；首个真实 token 到达即显示。引用只在 final 事件返回。浏览器断开时关闭后端生成器和 HTTP stream；原 `/chat` 保留。

三次优化后流式请求建立平均 103.81ms；TTFT 平均 28.56s；完整生成平均 39.90s。页面改善的是“有 token 后立即显示”与阶段透明度，不能消除外部模型在首字前的等待。

健康状态已真实验证：`prewarmed=true`、Embedding `ready`、Chroma `ready`、66 PDFs、7,337 chunks。启动器等待预热完成后再打开页面；不扫描/重建索引，也不调用 LLM 预热。

## 测试结果

- `pytest -q`：52 passed，1 个既有 Starlette `httpx`/`httpx2` deprecation warning。
- 新增性能回归：Chroma/LLM 单例、SDK 隐式重试关闭、版本化缓存、相邻去重不截断、普通问答单 LLM、真实流事件先于 final 引用。
- 真实本地预热：通过；66 PDFs / 7,337 chunks。
- 真实模型：基线 13 次 + 优化后 12 次；优化后全部引用编号有效。
- 没有修改 `data/pdfs` 或 `chroma_db`，没有重建索引。

## 文件改动（本次性能部分）

- `src/embedder.py`、`src/vectorstore.py`、`src/llm_client.py`：线程安全复用、连接池、重试、stream。
- `src/retriever.py`：计时、缓存、版本失效、去重、文档配额、上下文预算。
- `src/rag_chain.py`：逐阶段性能、单调用约束、非流/流式链。
- `src/warmup.py`、`src/status.py`：本地预热与健康状态。
- `api_server.py`、`app.py`、`launcher.pyw`：SSE、进度、性能摘要、预热可视化。
- `src/tracing/schema.py`、`src/tracing/instrumentation.py`：性能字段、调用/重试信息、文档预览限长。
- `src/config.py`、`.env.example`、本机 `.env`：Top-K=5、预算、缓存和 timeout 配置。
- `tests/test_performance_runtime.py`：性能回归测试。
- `README.md`、本报告：使用和验收说明。

项目还包含此前 Production Demo/启动器任务产生的其他未提交改动，详见最终 `git status`；本次没有覆盖或回滚这些改动。

## 无法由本地代码消除的部分

DeepSeek 的队列、跨网传输与推理速度仍占总耗时绝大多数。观测 TTFT 为约 15–35s、完整生成约 21–58s，且小 Prompt 也可能更慢。若要继续降低这部分，只能在获得确认后调整外部模型/服务等级、限制回答长度或更换网络路径；本次均未执行。

## 面试演示建议

1. 先用启动器停止旧进程再启动，等待 Embedding/Chroma `ready` 和“预热完成”。
2. 默认 Top-K=5、Tracing 关闭、对照模式关闭；打开页面“性能调试模式”。
3. 首问使用“PFAS 水处理工程化面临哪些主要限制？”，展示真实阶段进度、流式 token 与最终引用。
4. 同题再问一次，展示 Retrieval cache hit；随后换一道问题展示缓存不会污染不同 Query。
5. 展开性能详情，明确分开讲本地约几十 ms 与外部模型几十秒；不要承诺固定 TTFT。
6. 最后按需开启直接 LLM 对照或评测看板，并指出这时才会发生额外模型调用。
