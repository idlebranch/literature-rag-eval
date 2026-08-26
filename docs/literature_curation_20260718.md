# Historical Literature Curation — 2026-07-18

> This is an archival curation note for a pre-freeze corpus. The v1.0.0 runtime
> uses the frozen 270-PDF corpus documented in `README.md`.

## Scope

The local research corpus supports a water-treatment topic focused on emerging
contaminants, advanced oxidation/reduction processes, PMS/PDS, ozonation,
photocatalysis, activated-carbon adsorption and regeneration, PPCPs, and PFAS.

The pre-curation corpus contained 57 PDFs and 17,339 indexed chunks. All 57 PDFs
were present in Chroma, but the corpus contained one semantic duplicate: `PMS.pdf`
and `UV,H2O2.pdf` had identical extracted text and DOI
`10.1016/j.cej.2021.130457`. The incorrectly named duplicate was removed and the
remaining file was renamed descriptively.

## Selection policy

New papers were selected to fill identified gaps rather than simply increase
volume. Every item has a verified DOI, a legal open-full-text source, a valid PDF,
extractable text, and no DOI match in the original corpus. PDFs and Chroma files
remain local and are excluded from Git; this record is the reproducible catalog.

## Added papers

| Year | Paper | DOI | Access | Coverage added |
|---:|---|---|---|---|
| 2023 | Confined water-encapsulated activated carbon for capturing short-chain PFAS from drinking water | [10.1073/pnas.2219179120](https://doi.org/10.1073/pnas.2219179120) | PMC10318985, CC BY-NC-ND | Short-chain PFAS adsorption, selectivity, regeneration |
| 2024 | Carbon adsorbent properties impact hydrated electron activity and PFCA destruction | [10.1021/acsestengg.4c00211](https://doi.org/10.1021/acsestengg.4c00211) | PMC11406532, CC BY | Activated carbon coupled with UV/sulfite advanced reduction |
| 2016 | Post-ozonation in a municipal wastewater treatment plant improves receiving-stream water quality | [10.1186/s12302-015-0068-z](https://doi.org/10.1186/s12302-015-0068-z) | PMC5044949, CC BY | Full-scale ozonation, real water, ecotoxicity |
| 2019 | Efficient sulfamerazine removal by ozonation after enrichment with granular activated carbon | [10.1039/c8ra10429h](https://doi.org/10.1039/c8ra10429h) | PMC9062022, CC BY-NC | Adsorption/desorption and ozone coupling |
| 2025 | Ozonation of pharmaceuticals and human metabolites in wastewater: laboratory and field data | [10.1021/acs.est.5c08128](https://doi.org/10.1021/acs.est.5c08128) | PMC12490022, CC BY | Field evidence, metabolites, transformation products |
| 2022 | Regeneration of dye-saturated activated carbon through advanced oxidative processes | [10.1016/j.heliyon.2022.e10205](https://doi.org/10.1016/j.heliyon.2022.e10205) | PMC9404357, CC BY-NC-ND | AOP-based activated-carbon regeneration mechanisms |
| 2023 | Micropollutant removal efficiency of advanced wastewater treatment plants: systematic review | [10.1177/11786302231195158](https://doi.org/10.1177/11786302231195158) | PMC10492480, CC BY-NC | Cross-process and engineering-scale evidence |
| 2023 | Uncovering hydrothermal treatment of PFAS | [10.1016/j.eehl.2023.02.002](https://doi.org/10.1016/j.eehl.2023.02.002) | PMC10702917, CC BY-NC-ND | PFAS destruction and defluorination mechanisms |
| 2023 | Pharmaceutical transformation products formed by ozonation — does degradation occur? | [10.3390/molecules28031227](https://doi.org/10.3390/molecules28031227) | PMC9919501, CC BY | Transformation products and residual risk |
| 2023 | Efficient chemical-free degradation with an immobilized dual-porous TiO2 photocatalyst | [10.1021/acsestengg.3c00191](https://doi.org/10.1021/acsestengg.3c00191) | PMC10644339, CC BY | Immobilized photocatalysis and practical catalyst design |

## Local artifacts

- Machine-readable manifest: `data/curated_literature_20260718.json`
- Curated PDFs: `data/pdfs/curated_20260718/`
- Both paths are intentionally ignored by Git to avoid redistributing article
  files and research data through the source repository.
