# 语料筛选记录

## 语料范围

本地语料围绕水处理与环境工程的新兴污染物方向，覆盖高级氧化与还原过程、过硫酸盐体系、臭氧氧化、光催化、活性炭吸附与再生、药物与个人护理品、PFAS 这几类主题。

筛选前的初始语料是 57 篇 PDF、17,339 个索引片段。核对时发现一篇语义重复：`PMS.pdf` 和 `UV,H2O2.pdf` 提取出的正文完全一致，DOI 都是 `10.1016/j.cej.2021.130457`，属于同一篇论文被存了两份。删掉命名错误的那份，保留的一份重新改成了有意义的文件名。

## 筛选标准

补文献的原则是填缺口，不是堆数量。每篇都要求：

- DOI 可核实
- 有合法的开放全文来源
- PDF 能正常解析出文本层
- 与原语料无 DOI 重复

PDF 和索引文件都留在本地，不进 Git。这份记录本身是可复现的目录清单。

## 补充的 10 篇

| 年份 | 论文 | DOI | 开放来源 | 补上的方向 |
|---:|---|---|---|---|
| 2023 | Confined water-encapsulated activated carbon for capturing short-chain PFAS from drinking water | [10.1073/pnas.2219179120](https://doi.org/10.1073/pnas.2219179120) | PMC10318985，CC BY-NC-ND | 短链 PFAS 吸附、选择性、再生 |
| 2024 | Carbon adsorbent properties impact hydrated electron activity and PFCA destruction | [10.1021/acsestengg.4c00211](https://doi.org/10.1021/acsestengg.4c00211) | PMC11406532，CC BY | 活性炭耦合紫外/亚硫酸盐还原 |
| 2016 | Post-ozonation in a municipal wastewater treatment plant improves receiving-stream water quality | [10.1186/s12302-015-0068-z](https://doi.org/10.1186/s12302-015-0068-z) | PMC5044949，CC BY | 全规模臭氧工艺、真实水体、生态毒性 |
| 2019 | Efficient sulfamerazine removal by ozonation after enrichment with granular activated carbon | [10.1039/c8ra10429h](https://doi.org/10.1039/c8ra10429h) | PMC9062022，CC BY-NC | 吸附脱附与臭氧的耦合 |
| 2025 | Ozonation of pharmaceuticals and human metabolites in wastewater: laboratory and field data | [10.1021/acs.est.5c08128](https://doi.org/10.1021/acs.est.5c08128) | PMC12490022，CC BY | 现场数据、代谢产物、转化产物 |
| 2022 | Regeneration of dye-saturated activated carbon through advanced oxidative processes | [10.1016/j.heliyon.2022.e10205](https://doi.org/10.1016/j.heliyon.2022.e10205) | PMC9404357，CC BY-NC-ND | 高级氧化再生机制 |
| 2023 | Micropollutant removal efficiency of advanced wastewater treatment plants: systematic review | [10.1177/11786302231195158](https://doi.org/10.1177/11786302231195158) | PMC10492480，CC BY-NC | 跨工艺、工程尺度的证据 |
| 2023 | Uncovering hydrothermal treatment of PFAS | [10.1016/j.eehl.2023.02.002](https://doi.org/10.1016/j.eehl.2023.02.002) | PMC10702917，CC BY-NC-ND | PFAS 破坏与脱氟机制 |
| 2023 | Pharmaceutical transformation products formed by ozonation — does degradation occur? | [10.3390/molecules28031227](https://doi.org/10.3390/molecules28031227) | PMC9919501，CC BY | 转化产物与残留风险 |
| 2023 | Efficient chemical-free degradation with an immobilized dual-porous TiO2 photocatalyst | [10.1021/acsestengg.3c00191](https://doi.org/10.1021/acsestengg.3c00191) | PMC10644339，CC BY | 固定化光催化与催化剂设计 |

## 本地文件

- 机器可读清单：`data/curated_literature_20260718.json`
- 筛选后的 PDF：`data/pdfs/curated_20260718/`

这两个路径都在 `.gitignore` 里，避免把论文原文和研究数据通过代码仓库再分发出去。
