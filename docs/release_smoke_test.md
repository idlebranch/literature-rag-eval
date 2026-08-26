# Release Smoke Test

This is a release-verification checklist for the frozen v1.0.0 demo. It does
not rebuild an index and does not run an acceptance benchmark.

## Runtime identity

After launching the API and Streamlit UI, `/health` must report:

- `application_version`: `1.0.0`
- `build_id`: `v1.0.0-final`
- knowledge base: 270 PDFs at `data/papers/final_corpus`
- dense collection: `section_aware_270_gpu`, 17,028 chunks
- sparse index: `ready`, 17,028 chunks
- retrieval: `section_hybrid`, BGE-M3 Dense + Sparse, RRF, section-aware

## Two live checks

1. Submit the verified answerable corpus question “CHTC 对亚甲基蓝的实际平衡
   吸附量是多少？”. Confirm the direct answer `34.64 mg/g [S2]`, a valid
   citation, and expandable paper/page/section provenance. `[S2]` must resolve
   to PDF pages 52–53 in `改性水热炭对 MB 的吸附性能研究`.
2. Submit “What year did the French Revolution begin?” Confirm an evidence-
   insufficiency response with no invented citation.

## Screenshot capture

Screenshots are intentionally deferred from the v1.0.0 repository commit. To
add them later, capture only real UI output after the two checks:

- `docs/assets/demo-answer.png`: question, answer, and citation visible.
- `docs/assets/demo-evidence.png`: expanded source card showing paper, page,
  section, and chunk provenance.

Do not use mock data, image generation, or manually fabricated UI screenshots.
