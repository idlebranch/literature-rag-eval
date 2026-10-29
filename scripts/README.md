# 脚本说明

这里的脚本是语料构建和评测流程的工具，不是服务运行时要依赖的东西。按用途分成几组，实际用到的顺序大致也是从上到下。

## 抓取与整理语料

- `fetch_candidates_openalex.py`、`fetch_step1_oa.py`、`fetch_phase25.py`、`fetch_hn_supplement.py`：从 OpenAlex 和 PMC 拉候选文献的元数据
- `select_candidates.py`、`trim_candidates.py`：按缺口填补和去重规则筛出候选
- `integrate_manual_downloads.py`、`merge_manual_downloads.py`：把手工下载的 PDF 并入语料
- `preview_duplicates.py`、`inspect_supplements.py`：查重、检查补充材料
- `scan_corpus_pdfs.py`：扫描语料 PDF，检查文本层是否可提取

## 建立索引

- `build_final_corpus.py`：生成最终语料目录
- `build_section_chunks.py`：按章节切分，保留页码与章节信息
- `build_indexes.py`、`build_sparse_index.py`：分别建稠密索引和稀疏索引
- `build_inventory.py`：生成语料清单

## 评测与验收

- `build_eval_v2.py`、`build_acceptance.py`：构建开发集和独立验收集
- `run_eval_v2.py`、`run_acceptance.py`、`run_final_test.py`：跑评测
- `run_behavior_phase_d.py`、`rerun_behavior.py`：跑动作判断相关的行为评测
- `validate_eval_v2.py`、`validate_acceptance.py`：校验评测产物的完整性和一致性
- `analyze_misses.py`：逐条归因漏检样本
- `ablation_retrieval.py`、`ablate_conflict.py`：检索方案和冲突规则的消融实验
- `benchmark_answer_modes.py`：对比快速与详细两种回答模式的耗时和篇幅
- `route_regression.py`：路由相关的回归检查

## 诊断与运维

- `diagnose_retrieval.py`、`diagnose_doi.py`、`diagnose_elsevier_api.py`：定位检索、DOI、接口具体哪一环出问题
- `probe_fulltext_channels.py`：试探各来源的全文可用性
- `retest_oa_abstract.py`：复核开放获取链接是否真的能拿到全文
- `preflight_gpu.py`：跑批之前检查 CUDA 与显存
- `watch_build.py`：监视建库进度
- `verify_indexes.py`、`verify_final.py`：校验索引和最终产物
- `report_phase1.py`、`finalize_step1.py`：阶段性统计与收尾
- `export_endnote.py`：导出 EndNote 格式的文献库
- `github_survey.py`：调研开源同类项目的调研脚本

## 模型下载

- `fetch_bge_heads.py`：下载 BGE-M3 的分词与权重头文件
- `fetch_reranker.py`：下载重排模型

## 注意

不是每个脚本都能直接跑。抓取类需要外网和 OpenAlex、PMC、Elsevier 的访问权限；建索引需要本地 PDF 和 GPU；评测类会真实调用模型接口并产生费用。另外这些脚本是边做边写的，参数和路径有不少是当时环境下硬编码的，复用前需要先改配置。
