# System Scope & Limitations

This documents the deliberate, by-design scope of the water-treatment literature
RAG system. These are system boundaries, not bugs.

1. **Corpus** = 270 primary PDFs (`data/papers/final_corpus`), frozen.
2. **Coverage** = machine-text-extractable PDF body text. Scanned/image-only pages
   without a text layer are not covered.
3. **Supplementary files** = not indexed; each paper's standalone Supplementary
   Information files are out of scope.
4. **Tabular supplementary data** (XLSX/CSV) = not parsed.
5. **PDF tables** = if text extraction loses row/column structure, exact numeric
   evidence inside those tables may be unrecoverable.
6. **Figure-only evidence** = outside the current text-RAG capability (no OCR /
   image model / visual grounding).
7. **References section** = intentionally excluded from the retrieval index to
   avoid citation leakage (a citation sentence is not body evidence).
8. **Automatic conflict classification** = removed. The previous keyword-based
   increase/decrease heuristic produced systematic mis-classification on held-out
   data (ordinary ANSWERABLE text flagged as "conflicting"). Reliable conflict
   detection would require claim/condition alignment, which is out of scope for
   the freeze.
9. **Condition-dependent / source-disagreement** = handled by the answer layer
   (generation prompt instructs the model to present each source, metric and
   experimental condition, and not to force a single conclusion), not by a
   pre-classified "conflict" action.
10. **Eval V2 held-out behavior TEST is consumed.** Post-freeze V2 numbers are
    regression/postmortem only and are NOT re-claimed as held-out generalization
    results.

RAG pipeline summary: 270 PDFs → section-aware ingestion → BGE-M3 Dense + Sparse
+ RRF (section_hybrid) → evidence-aware answerability (ANSWER / CLARIFY / REFUSE /
PARTIAL_ANSWER / CORRECT_PREMISE) → traceable citations.
