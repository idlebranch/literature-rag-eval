# Literature RAG

本项目是面向水处理科学文献的本地 RAG Production Demo：从 PDF 解析、Chunking、BGE-M3 Embedding、Chroma 检索、OpenAI-compatible LLM 生成到引用回溯、Tracing 和离线评测，均由当前 Python 代码链路完成。

> Dify 仅属于早期原型（Legacy prototype）。当前正式链路不依赖 Dify、v0/Bolt，也没有迁回低代码平台。

## 当前正式架构

```text
data/pdfs (66 PDFs)
  → PyMuPDF
  → chunking
  → BAAI/bge-m3 (dense via SentenceTransformer; sparse lexical weights via FlagEmbedding)
  → ChromaDB: literature_chunks (7,337 chunks) + sparse_index/ (BGE-M3 词面权重倒排索引)
  → rule-based bilingual Query Rewrite
  → Retrieval (三种模式，见下文)
  → rag_answer_prompt_v2（快速/详细）
  → deepseek-v4-pro (OpenAI-compatible API)
  → 确定性引用校验 + [Sx] 引用 + fallback + optional trace
```

### 检索模式（RETRIEVAL_MODE）

`src/config.py` 的 `retrieval_mode`（环境变量 `RETRIEVAL_MODE`）显式选择检索链路，三种模式都不会静默降级——前置条件缺失时直接报错：

- `dense_only`（默认）：保持原有行为，Chroma dense 单路检索 + 来源去重。
- `hybrid_dense_sparse`：Dense Top-K + BGE-M3 sparse 词面检索 → RRF 融合 → 来源去重。需要先用 `python -m scripts.build_sparse_index` 构建 `sparse_index/`；索引与 Chroma collection 版本绑定，语料变化后必须重建。
- `hybrid_reranker`：在 hybrid 融合候选（默认 25 条）之上用 `BAAI/bge-reranker-v2-m3` cross-encoder 重排，取 `RERANKER_FINAL_K`（默认 8）条进入生成。

关键参数（均有环境变量）：`HYBRID_DENSE_K=25`、`HYBRID_SPARSE_K=25`、`HYBRID_FUSION_K=25`、`RRF_K=60`、`RERANKER_FINAL_K=8`、`SPARSE_MAX_LENGTH=512`。

稀疏索引构建：

```powershell
.\.venv\Scripts\python.exe -m scripts.build_sparse_index
```

检索消融评测（Recall@K / Hit@K / MRR + 分阶段延迟）：

```powershell
.\.venv\Scripts\python.exe -m scripts.ablation_retrieval --modes dense_only,hybrid_dense_sparse,hybrid_reranker
```

#### 检索消融实测结果（2026-08，groundtruth.example.jsonl 15 题）

| Variant | Recall@5 | Recall@10 | MRR@10 | Hit@10 | 平均延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense（dense_only） | 0.544 | 0.628 | 0.744 | 1.000 | ~282ms |
| Hybrid（dense+sparse RRF） | 0.528 | 0.594 | **0.810** | 0.933 | ~183ms |
| Hybrid + Reranker | 0.500 | 0.556 | 0.593 | 0.867 | ~7580ms |

- reranker_device=cuda（bge-reranker-v2-m3），chunks=7337，sparse index=864813 postings。
- 平均延迟含首题的模型/索引冷启动；稳态单题延迟：dense 约 20ms，hybrid 约 50ms，rerank 阶段约 7.5s。
- 15 题样本较小：Hybrid 唯一稳定正向信号是 MRR（首次命中排位提前），recall 差异在噪声范围内；Reranker 当前配置全指标下降且延迟高，默认不建议启用。完整实测与分析见 [`docs/accuracy_optimization_report.md`](docs/accuracy_optimization_report.md) 第 18 节。

- API：FastAPI / Uvicorn，入口 `api_server:app`
- 问答 UI：Streamlit，入口 `app.py`
- 评测 UI：Streamlit，入口 `app_eval.py`
- 知识库：`data/pdfs/`
- 向量库：`chroma_db/`，collection 为 `literature_chunks`
- 评测：六项 Judge（correctness、evidence relevance、faithfulness、completeness、citation、overall）与七类 bad case
- Tracing：`src/tracing/`，默认关闭，开启后写入 `outputs/traces/traces.jsonl`

## 图形化启动（推荐）

双击桌面的 **Literature RAG** 快捷方式，或直接运行：

```text
dist\LiteratureRAG-Launcher.exe
```

启动器会：

1. 检查 8010 / 8501 端口、PID 身份、知识库、索引和 LLM 配置。
2. 无终端窗口启动 FastAPI，等待 `/health`。
3. 后台加载 BGE-M3 与 Chroma collection，并执行一次最小本地 Embedding / Chroma 查询；不调用 LLM。
4. 无终端窗口启动 Streamlit，等待 `/_stcore/health`。
5. Embedding、Chroma、知识库和两端服务全部就绪后打开浏览器。
6. 用项目 PID + 可执行文件路径 + Windows 进程创建时间验证归属后，才允许停止完整进程树。

重复点击不会创建重复服务；启动器会核验 `/health` 中的 project/application/build/prompt 身份，旧构建不会被误报为 ready。端口属于其他程序或无法确认归属时会明确报冲突且拒绝误杀。关闭启动器窗口不会停止服务，只有“停止项目”按钮会停止由本启动器管理的 API、RAG UI 和评测 UI。

按钮还可打开 API 文档、知识库状态、评测看板和 `logs/`。评测看板按需启动在 8502，不会混入普通问答流程。

### 构建 EXE

项目使用轻量的 tkinter + PyInstaller；EXE 不包含 PDF、Chroma、模型或整个虚拟环境，并依赖其固定位置 `项目根目录\dist\`。

```powershell
cd C:\Users\10475\AI_PROJECT\literature_rag_eval_code
.\build_launcher.cmd
.\create_desktop_shortcut.ps1
```

输出：`dist\LiteratureRAG-Launcher.exe`。快捷方式工作目录指向项目根目录，不需要管理员权限。

## 命令行启动

当前依赖清单是 `requirements.txt`，运行环境是项目内 `.venv`。本机可用 uv 同步该 requirements 环境：

```powershell
cd C:\Users\10475\AI_PROJECT\literature_rag_eval_code
uv venv .venv --python 3.11
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

也可用标准 `python -m venv` + pip 创建等价 `.venv`；不需要迁移到 pyproject。

分别启动：

```powershell
# API
.\.venv\Scripts\python.exe -m uvicorn api_server:app --host 127.0.0.1 --port 8010

# RAG UI
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501

# 评测看板（可选）
.\.venv\Scripts\python.exe -m streamlit run app_eval.py --server.address 127.0.0.1 --server.port 8502
```

端口 8010 是有意选择的：当前机器的 8000 由 Multi-Agent BI 使用，两个项目不共享进程、代码或 PID 文件。

## 地址与 API

| 功能 | 地址 |
| --- | --- |
| RAG 页面 | `http://127.0.0.1:8501` |
| 评测看板 | `http://127.0.0.1:8502` |
| API 文档 | `http://127.0.0.1:8010/docs` |
| 脱敏健康状态 | `http://127.0.0.1:8010/health` |
| 知识库状态 | `http://127.0.0.1:8010/knowledge-base/status` |
| RAG 问答 | `POST http://127.0.0.1:8010/chat` |
| RAG 真流式问答（SSE） | `POST http://127.0.0.1:8010/chat/stream` |
| 直接 LLM | `POST http://127.0.0.1:8010/llm/chat` |

`/health` 只返回配置状态、模型名称、文档/Chunk 数等脱敏信息，不返回 API Key、Authorization Header 或完整 `.env`。Embedding 与 LLM 网络调用不会因频繁健康检查而重复加载/执行。

## 查询性能与流式输出

- BGE-M3、tokenizer、Chroma PersistentClient/collection 和 OpenAI-compatible HTTP client 在进程内复用。
- 默认 `TOP_K=5`；同文档最多 2 个 chunk，并对高度重复的相邻 chunk 去重。
- 上下文按完整 chunk 控制 `CONTEXT_TOKEN_BUDGET`，不会截断字符串破坏引用对应关系。
- Query Embedding 与检索结果使用有大小上限的进程内缓存；检索缓存键包含 Chroma 文件版本，索引变化会自动失效。最终 LLM 回答不缓存。
- 普通问答只执行 Query Embedding → Chroma → 一次 LLM → 引用整理。Judge、bad case 和直接 LLM 仅在显式评测/对照模式运行。
- Streamlit 使用 `/chat/stream` 展示真实模型 token；引用在生成完成后统一显示。若浏览器断开，后端会关闭模型流。
- 开启页面“性能调试模式”可查看逐阶段耗时、TTFT、Prompt Tokens、LLM 调用/重试次数和 cache hit/miss；普通模式只显示紧凑摘要。

真实性能基线、优化后统计及限制见 [`docs/performance_optimization_report.md`](docs/performance_optimization_report.md)。

## 回答模式与证据校验

- **快速回答（默认）**：直接结论、3–5 个主要点和必要限制，`max_tokens=1200`。
- **详细回答**：多来源综合、研究差异、机制、适用条件和证据边界，`max_tokens=2200`。
- 两种模式共享完全相同的事实与引用约束，只改变篇幅和展开程度。
- 生成后执行不调用 LLM 的引用校验：检查 `[Sx]` 越界、页码映射、无引用的“文献证明”式表述、可疑作者/标题和长度截断。
- Prompt 版本、内容哈希、证据状态和校验结果进入 API/Trace；普通页面只显示紧凑状态，详细数据位于折叠区。

Prompt 与模型调用清单见 [`docs/prompt_inventory.md`](docs/prompt_inventory.md)，完整准确性与 8010 治理报告见 [`docs/accuracy_optimization_report.md`](docs/accuracy_optimization_report.md)。

## RAG 与直接 LLM 对照

页面默认只运行一次 RAG。主动开启“RAG / 直接 LLM 对照模式”后，才会额外调用 `/llm/chat` 等价链路。直接 LLM 结果明确标记为“未检索知识库、无 RAG 引用”，便于比较：

- RAG 是否提供真实来源和页码；
- 回答是否有检索证据；
- 无依据时是否 fallback；
- 直接模型是否产生无支撑结论。

Query Rewrite、原始 Chunk、distance、fallback 原因和 Trace ID 默认折叠，业务回答保持在页面主区域。

## Fallback 与输入边界

- 空输入和超过 8,000 字符的问题由 UI/API 拦截。
- 明显的系统提示/密钥窃取或伪造引用请求会确定性拒绝，不调用检索或 LLM。
- 指代不明的问题请求用户补充对象和条件。
- 最佳向量距离超过 `MAX_RETRIEVAL_DISTANCE`（默认 1.15）时，不调用 LLM 编造答案。
- 该阈值来自当前索引的离线探针；更换 Embedding 或语料后应重新评估。

## Tracing

Tracing 默认关闭，因为 trace 会保存问题、回答和不超过 200 字符的检索预览。需要面试展示时在 `.env` 设置：

```env
TRACING_ENABLED=true
```

重启 API/UI 后，普通问答页面的折叠区会显示 Trace ID，数据写入 `outputs/traces/traces.jsonl`。展示结束可重新设为 `false`。Tracing 写入失败是 fail-open，不会让普通 RAG 请求失败。

## 评测

评测 canonical truth 为 `outputs/runs/*.json`，可读视图为 `outputs/views/*.md|csv`。当前历史产物位于 `outputs/archive_20260603/`，`app_eval.py` 会在当前 views 为空时自动读取最新归档。

```powershell
# 纯离线测试
.\.venv\Scripts\python.exe -m pytest

# 模型配置 preflight
.\.venv\Scripts\python.exe -m src.eval.cli health

# ground truth → RAG run
.\.venv\Scripts\python.exe -m src.eval.cli run --groundtruth groundtruth\groundtruth.jsonl --run-id my_run

# Judge（独立执行，不影响普通问答）
.\.venv\Scripts\python.exe -m src.eval.cli judge --run outputs\runs\my_run.json --judge-model <model>

# 导出视图 / bad case / index
.\.venv\Scripts\python.exe -m src.eval.cli export --run outputs\runs\my_run.json --format both
.\.venv\Scripts\python.exe -m src.eval.cli badcase --run outputs\runs\my_run.json
.\.venv\Scripts\python.exe -m src.eval.cli index
```

人工测试见 [`docs/manual_test_checklist.md`](docs/manual_test_checklist.md)，完整产品问题集见 [`docs/product_test_questions.md`](docs/product_test_questions.md)。

## 索引构建与安全

启动器和页面都不会自动重建索引。代码已禁止 `src.ingest` 覆盖现有 active collection。初次建库或刷新时，先在新的候选目录构建：

```powershell
$env:CHROMA_DIR = Join-Path $PWD "chroma_db_candidate_20260806"
.\.venv\Scripts\python.exe -m src.ingest
```

构建失败不会接触 `chroma_db/`。成功后先使用相同 `CHROMA_DIR` 做 collection count、离线检索和引用核对；只有人工确认候选索引可用后，才安排可回滚的目录切换。不要在模型网络不可用时删除当前索引，也不要把 PDF、向量库或模型打进 EXE。

当前正式产品没有 PDF 上传 API/UI。PDF 入库是本地受控流程，因此网页不存在任意路径读取或上传覆盖项目代码的入口。扫描版、空白或损坏 PDF 若无法提取文本，`src.ingest` 会明确失败，不会报告已索引成功。

## 日志与停止

- `logs/launcher.log`
- `logs/rag_api.log`
- `logs/rag_ui.log`
- `logs/rag_eval.log`（按需）

日志不记录 `.env`、API Key、Authorization Header 或文档全文。API 返回面向用户的简洁错误，不直接展示 Python stack。技术异常写入日志。

图形启动时请使用“停止项目”。命令行启动时在对应窗口按 `Ctrl+C`。启动器不会终止没有匹配 PID 身份的 Python/Uvicorn/Streamlit 进程。

## 常见问题

- **后端端口冲突**：确认 `.env` 的 `RAG_API_PORT`；不要停止或修改占用 8000 的 Multi-Agent BI。
- **索引 missing / unavailable**：检查 `chroma_db/chroma.sqlite3` 和 collection 名；不要自动重建或删除旧索引。
- **LLM not_configured**：只在本机 `.env` 配置凭据，不要提交或粘贴到日志。
- **Embedding 首次加载慢**：启动器会在打开页面前预热本地模型；健康状态显示 `loading / ready / failed`，不会调用付费 LLM。
- **评测看板无数据**：确认 `outputs/views`，或保留带 `views` 的 `outputs/archive_*`。
- **EXE 移动后找不到项目**：EXE 必须保留在项目 `dist` 目录；桌面使用快捷方式，不复制 EXE 本体。

## 5 分钟面试展示

1. 双击快捷方式，展示无终端的启动器和五项状态。
2. 打开 RAG 页面，提问 PFAS 工程限制，先看带 `[Sx]` 的业务回答。
3. 展开检索证据和 Query Rewrite / Trace，核对文档名、页码和 distance。
4. 开启对照模式再运行一题，说明直接 LLM 没有本地引用支撑。
5. 输入库外问题或伪造引用请求，展示 deterministic fallback；最后打开独立评测看板查看四维 Judge 与 bad case。
