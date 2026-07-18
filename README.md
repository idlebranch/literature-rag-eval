# Scientific Literature RAG Evaluation Platform

这是一个从 Dify 迁移到代码版的最小可运行 RAG 项目骨架。

> **V3 更新**（2026-05-25）：评测产物从 CSV+MD 双轨改为 **JSON 单一真相 + 可读视图自动生成**，并提供 Streamlit dashboard 用于人工评测。详见 [CHANGELOG_V3.md](./CHANGELOG_V3.md) 与下文「V3 评测工作流」一节。

原 Dify 流程：

PDF 文献上传 → Knowledge Base → Retrieval → LLM Answer → Citation → 人工校验

代码版流程：

PDF → 文本解析 → chunking → embedding → Chroma 向量库 → 检索 → LLM 生成 → citation → ground truth 评测

## 目录结构

```text
literature_rag_eval_code/
  app.py                     # Streamlit RAG 前端
  app_eval.py                # Streamlit 评测 dashboard
  requirements.txt
  .env.example
  data/
    pdfs/                    # 放你的 PDF 文献
  groundtruth/
    groundtruth.example.jsonl
  outputs/
    runs/                    # JSON canonical truth（评测产出）
    views/                   # 自动生成的 MD / CSV 报告
  scripts/
    audit_deepseek.py        # 安全审计工具（独立脚本）
  src/
    __init__.py
    config.py                # 全局配置（从 .env 加载）
    pdf_loader.py            # PyMuPDF PDF 解析
    chunking.py              # 文本分块
    embedder.py              # BGE-M3 / Sentence Transformer 向量化
    vectorstore.py           # Chroma 向量库读写
    ingest.py                # PDF → chunk → embedding → Chroma 入库
    retriever.py             # 检索（含 query expansion + 来源多样性）
    llm_client.py            # OpenAI-compatible LLM client
    rag_chain.py             # RAG 问答主链（含 system prompt）
    eval/                    # V3 评测子包
      __init__.py
      schema.py              # EvalRun / QuestionResult 数据模型
      io.py                  # JSON 读写 + 旧 CSV 迁移
      export.py              # → 可读 MD + 扁平 CSV
      badcase.py             # Badcase 按 error_type 分类 + 优化建议
      health.py              # API key / endpoint preflight 检查
      runner.py              # 端到端 RAG + Judge 执行器
      cli.py                 # 统一 CLI 入口
    utils/                   # 工具
      logging.py             # 项目级日志配置
  tests/
    __init__.py
    conftest.py
    test_eval_schema.py
    test_eval_io.py
    test_eval_export.py
    test_eval_badcase.py
    test_eval_runner.py
    test_eval_cli.py
```

## 1. 创建环境

Windows PowerShell：

```powershell
cd C:\Users\10475\AI_PROJECT\literature_rag_eval_code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. 配置 API

复制 `.env.example` 为 `.env`，填入你的 API：

```env
OPENAI_API_KEY=你的key
OPENAI_BASE_URL=https://anyrouter.top/v1
LLM_MODEL=gpt-5.5
EMBEDDING_MODEL=BAAI/bge-m3
```

如果你用 AnyRouter、OpenRouter 或其他 OpenAI-compatible API，把 `OPENAI_BASE_URL` 改成对应地址，`LLM_MODEL` 改成它支持的模型名。

> 不要把真实 API key 提交进代码或 README；只写在本机 `.env` 里。

## 3. 放 PDF

把经过 DOI 去重和质量筛选的 PDF 放入：

```text
data/pdfs/
```

## 4. 建库

```powershell
python -m src.ingest
```

> ⚠️ **注意**：每次更换、新增或删除 `data/pdfs/` 下的 PDF 后，必须重新运行 `python -m src.ingest` 重建索引。
> 也可以在 Streamlit 前端侧边栏点击 **"Rebuild Index"** 按钮一键重建。
>
> 索引状态（PDF 数量、ChromaDB chunk 数、Top-K）可在前端侧边栏实时查看。

当前本地课题库包含 66 篇去重全文，聚焦水中新污染物、高级氧化/还原、PMS/PDS、臭氧、光催化、活性炭再生、PPCPs 与 PFAS。新增文献的 DOI、开放许可、来源和选取理由见 [Literature curation 2026-07-18](./docs/literature_curation_20260718.md)。PDF、向量库和机器可读清单属于本地研究数据，按 `.gitignore` 不上传 GitHub。

## 5. 命令行提问

```powershell
python -m src.rag_chain "PAC 去除 SMX 的主要机制是什么？"
```

## 6. 启动本地 API

```powershell
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

直接调用模型：

```powershell
curl.exe -X POST http://127.0.0.1:8000/llm/chat `
  -H "Content-Type: application/json" `
  -d "{\"prompt\":\"你好，用一句话介绍你自己\"}"
```

OpenAI-style messages：

```powershell
curl.exe -X POST http://127.0.0.1:8000/llm/chat `
  -H "Content-Type: application/json" `
  -d "{\"messages\":[{\"role\":\"system\",\"content\":\"你是一个简洁助手\"},{\"role\":\"user\",\"content\":\"写一个 Python hello world\"}]}"
```

RAG 问答仍然使用原来的 `/chat`：

```powershell
curl.exe -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"PAC 去除 SMX 的主要机制是什么？\"}"
```

## 7. 启动前端

```powershell
streamlit run app.py
```

## 8. 启动评测 Dashboard

```powershell
streamlit run app_eval.py
```

读取 `outputs/` 下的 CSV/MD，分多个 section 展示：Retrieval、平均分、版本对比、Badcase Explorer（带筛选）、Badcase Analysis、对比报告。

## 9. V3 评测工作流（推荐）

V3 把所有评测产物收敛到 **JSON 单一真相**（`outputs/runs/*.json`），可读 MD 和扁平 CSV 都从 JSON 自动生成。所有操作通过统一 CLI：

```powershell
# 1. 从历史 CSV 迁移 → 生成 canonical JSON
python -m src.eval.cli migrate `
  --eval-csv outputs/answer_eval_manual.csv `
  --judge-csv outputs/answer_judge_summary.csv `
  --run-id 2026-05-24_v1_gpt-oss-120b_claude-opus-4-7 `
  --rag-model gpt-oss-120b --rag-prompt v1 `
  --embedding-model BAAI/bge-m3 --top-k 5 `
  --judge-model claude-opus-4-7

# 2. 从 JSON 导出可读 MD（带 TOC、锚点、分数条）+ 扁平 CSV
python -m src.eval.cli export --run outputs/runs/<id>.json --format both

# 3. 从 JSON 导出 badcase 分析（按 error_type 分类 + 优化建议）
python -m src.eval.cli badcase --run outputs/runs/<id>.json

# 4. 刷新 outputs/INDEX.md 总览
python -m src.eval.cli index

# 5. API/模型 preflight 检查（避免 V2 那种 15 条 401 才发现的坑）
python -m src.eval.cli health

# 6. 端到端跑 RAG（groundtruth → JSON）
python -m src.eval.cli run --groundtruth groundtruth/groundtruth.jsonl --run-id my_run

# 7. LLM-as-judge 评分（对已有 run 的每条回答打分）
python -m src.eval.cli judge --run outputs/runs/my_run.json --judge-model claude-opus-4-7
```

**目录约定**：
- `outputs/runs/<run_id>.json` — canonical truth（不要手改）
- `outputs/views/<run_id>.md` — 给人读的报告（自动生成）
- `outputs/views/<run_id>.csv` — 扁平评分（自动生成，不含长文本）
- `outputs/views/<run_id>_badcase.md` — badcase 分析
- `outputs/INDEX.md` — 所有 run 的总览

## 10. 跑单元测试

```powershell
pytest
```

当前覆盖 schema / io / export / badcase / runner / cli / tracing 共 36 个 case。测试不需要 `.env` 或 API key，纯逻辑测试。

## 11. 产品测试问题集

手工评测 RAG 质量的问题集，覆盖文献总览、单篇事实、多文献比较、机制归纳、反幻觉边界、综述生成六大类。

详见：[Product Test Questions](./docs/product_test_questions.md)

## 简历表达

面向专业文献的可评测 RAG 问答系统：从零实现 PDF 解析、文本分块、BGE-M3 embedding、Chroma 向量检索、LLM 生成、引用溯源与人工 ground truth 检索评测流程；通过 Hit@K、expected source match、badcase 记录定位检索噪声，为后续 RAGAS、reranker 与 Langfuse tracing 优化提供 baseline。
