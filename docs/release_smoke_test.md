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

1. Submit a direct answerable MB adsorption-capacity question. Confirm the
   literature-supported `34.64 mg/g` answer, at least one valid `[Sx]`
   citation, and expandable paper/page/section provenance. The source ordinal
   can vary with the retrieved context ordering; it must map to the displayed
   context card.
2. Submit “What year did the French Revolution begin?” Confirm an evidence-
   insufficiency response with no invented citation.

## Screenshot capture

The checked-in images were captured only from real final-runtime UI output:

- `docs/assets/demo-answer.png`: question, answer, and citation visible.
- `docs/assets/demo-evidence.png`: expanded source card showing paper, page,
  section, and chunk provenance.
- `docs/assets/demo-refusal.png`: an underspecified question receives a
  clarification/refusal response without an unsupported answer.

Do not use mock data, image generation, or manually fabricated UI screenshots.
