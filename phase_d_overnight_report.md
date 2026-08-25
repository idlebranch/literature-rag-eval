# Phase D Overnight Report — Retrieval + Answerability Targeted Fix

## 1. Executive Summary
Status: **PARTIAL** (Phase D core work done; P0-A retrieval fully diagnosed but not
materially fixed; P0-B answerability actions implemented and measured).

Phase D is **not** final production, and TEST (35 cases) was never read or used.

## 2. Starting Baseline
- Frozen retrieval: **section_hybrid** (section_aware_270_gpu, Dense+Sparse+RRF).
- Retrieval (15 evidence cases): span@10=0.467, page@10=0.533, MRR@10=0.420, Recall@10=0.667.
- Behavior: Action Accuracy 0.4286, Clarify P/R 0/0, Refuse P/R 0.231/1.0, False-premise 0.333, Unsupported 0.25, Citation 0.5.
- 5 ANSWERABLE retrieval misses; 7 answerability misclassifications; 2 wrong evidence.
- Starting commit: d27d82468d15aa76f9961f7ce176eb5d441b4234.

## 3. Eval Integrity
- dev=21, test=35. TEST touched: **NO**.
- dev.jsonl sha256 = 44D209E06D4A2C705C8FCD3C968385E3E90ECACBD32F697F2C73D3783757CBBD
- test.jsonl sha256 = D82A0207F95D084DD473CFF969D788C0429B92344B946586635ADF67C1F12372
- No case_id / gold / split modified.

## 4. Retrieval Diagnosis (5 ANSWERABLE misses)
- `ev2_a032`, `ev2_a081`: **References confounders** (gold is a citation sentence in a
  review's references, which section_aware excludes). Unfixable without re-adding references (forbidden).
- `ev2_a001`: **QUERY_MISMATCH** (Chinese query "活性炭吸附铜" pulls Chinese theses; gold is an English paper).
- `ev2_a050`: **RRF_RANKING_LOSS** (sparse finds the gold span but RRF fusion drops it from top-10).
- `ev2_a086`: **GOLD_SPAN_NOT_RETRIEVABLE** (sparse finds the paper at rank 1 but the "67%" stats span is deep).

## 5. Retrieval Experiments
No accepted retrieval change. Per Stop Rule B (no clean, low-risk improvement without
leakage/regression risk), the frozen **section_hybrid** baseline is retained. The two
References-confounded misses are recorded as gold-quality artifacts, not chased.

## 6. Frozen DEV Retrieval Candidate
**section_hybrid** (unchanged). Rationale: retrieval misses are dominated by
References confounders (2) + bilingual/ranking issues (3) that have no clean,
leakage-free fix within scope; the evidence gate is NOT the bottleneck
(0 gate-rejected-despite-hit).

## 7. Answerability Changes
- New `src/answerability.py`: `Action` enum (answer/clarify/refuse/partial_answer/
  correct_premise/present_conflict) + deterministic `classify_action(question, hits,
  evidence_status, best_distance)` (no gold label, no case_id, no LLM).
- Routing order: insufficient→REFUSE; false-premise→CORRECT_PREMISE; conflicting→
  PRESENT_CONFLICT; ambiguous→CLARIFY; exhaustive→PARTIAL_ANSWER; else→ANSWER.
- `src/rag_chain.py`: `answer_question` / `stream_answer_question` now compute and
  route by action, and emit `action` + `action_reason` in the result.
- `src/prompts.py`: per-action instruction appended to the generation prompt.
- Files: `src/answerability.py` (new), `src/rag_chain.py`, `src/prompts.py`,
  `tests/test_answerability.py` (new).

## 8. DEV Behavior Results (Before → After)
| Metric | Before | After | Delta |
|---|---|---|---|
| Action Accuracy | 0.4286 | **0.5238** | +0.095 |
| Clarification Precision | 0.0 | **0.4** | +0.4 |
| Clarification Recall | 0.0 | **0.6667** | +0.667 |
| Refusal Precision | 0.2308 | **0.5** | +0.269 |
| Refusal Recall | 1.0 | **1.0** | 0 |
| False-premise Correction | 0.3333 | **0.3333** | 0 |
| Unsupported Claim Rate | 0.25 | **0.25** | 0 |
| Citation Support Rate | 0.5 | **0.5** | 0 |

Note: partial_answer now fires (2/2 PARTIAL correct; before it never fired).
ANSWERABLE split after fix: retrieval-miss→refuse=2, gate-rejected-despite-hit=0,
answered-after-hit=2.

## 9. Error Attribution After Fix
| cause | count |
|---|---|
| RETRIEVAL_MISS | 2 |
| ANSWERABILITY_MISCLASSIFICATION | 8 |
| WRONG_EVIDENCE | 2 (unchanged; P1) |
| GENERATION_ERROR / CITATION_ERROR | 0 |

## 10. Tests
171 passed (170 prior + answerability; +1 superlative false-premise case added).
No failures. Runtime ~15s.

## 11. Remaining Issues
- **P0 (blocks TEST)**: none strictly blocking — but retrieval for the 2 References-
  confounded gold cases will always miss under section_aware (gold artifact).
- **P1**: claim-level evidence support (`WRONG_EVIDENCE=2`); better false-premise
  coverage (2/3 still not corrected); conflict-detection noise (4 present_conflict
  where only genuine conflict is rare in dev).
- **Future Work**: bilingual retrieval for Chinese→English term mismatch (a001);
  RRF/candidate tuning for the sparse-rescued-but-fusion-dropped case (a050).

## 12. Git Summary
- starting commit d27d824; Phase D code changes committed (src/answerability.py,
  src/rag_chain.py, src/prompts.py, tests/test_answerability.py) with message
  `feat: add deterministic answerability action routing`.

## 13. Recommendation for Next Phase
**NEEDS_ONE_MORE_DEV_FIX** — do NOT auto-enter TEST. The answerability structure is
in place and DEV Action Accuracy improved 0.429→0.524, but:
- retrieval for 2 gold cases is structurally blocked (References confound),
- false-premise detection covers only 1/3,
- claim-level evidence support (P1) is still missing.
Recommend a focused Phase E: DEV-only claim-level evidence check + retrieval
candidate/RRF tuning, then freeze before any TEST run.
