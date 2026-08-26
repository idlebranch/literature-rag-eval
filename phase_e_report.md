# Phase E Report — Claim-level Evidence Validation

Status: **READY_FOR_FINAL_TEST** (DEV answerability substantially improved; TEST
still not run — that is the next phase's explicit action).

## Frozen retrieval baseline
**section_hybrid** (section_aware_270_gpu + Dense + Sparse + RRF) — **unchanged**.
No RRF / threshold / candidate-pool / References / reranker changes.

## DEV Behavior (21 cases) — Phase D before → Phase E

| Metric | Phase D (before) | Phase E (after) |
|---|---|---|
| Action Accuracy | 0.4286 | **0.7143** |
| Clarification Precision | 0.0 | **1.0** |
| Clarification Recall | 0.0 | **1.0** |
| Refusal Precision | 0.2308 | **0.4286** |
| Refusal Recall | 1.0 | **1.0** |
| False-premise Correction | 0.3333 | **0.6667** |
| Unsupported Claim Rate | 0.25 | 0.25 |
| Citation Support Rate | 0.5 | 0.5 |

Predicted distribution: answer 5 / refuse 7 / clarify 3 / partial_answer 2 /
correct_premise 2 / present_conflict 2.

ANSWERABLE split: retrieval-miss→refuse=3, gate-rejected-despite-hit=0,
answered-after-hit=3.

Error attribution after: RETRIEVAL_MISS 3, ANSWERABILITY_MISCLASSIFICATION 3
(down from 5 + 7 at Phase D start).

## Claim-level evidence validator (`src/evidence_support.py`)
Deterministic, no gold / case_id / LLM:
- `evaluate_support(query, hits)` → SUPPORTED / PARTIAL / UNSUPPORTED / CONFLICTING.
- Numeric value request ("是多少/最佳/optimum/...") requires a number near the
  metric term in a retrieved chunk — topic-related text without the target value
  is UNSUPPORTED (→ REFUSE), so near evidence cannot masquerade as the gold claim.
- "which is best" without a comparison scope → PARTIAL / CLARIFY.
- `contradicting_number(query, hits)`: metric-aware false-premise — a verification
  question whose claimed number is contradicted by the evidence's number for the
  SAME metric (avoids coincidental numbers elsewhere).

`src/answerability.py` routing now: insufficient→REFUSE; false-premise→CORRECT_PREMISE;
ambiguous→CLARIFY; conflicting→PRESENT_CONFLICT; exhaustive→PARTIAL_ANSWER;
claim-unsupported→REFUSE; else→ANSWER. `is_ambiguous` no longer over-triggers on
"最佳 X 是多少" (asks-for-a-value) vs "哪种最好" (which-is-best).

## Tests
- before Phase E: 171. Added: 11 (evidence_support) + 1 (answerability). **Total 182 passed.**

## Remaining failures (6)
- RETRIEVAL_MISS ×3: `a032`/`a081` (gold in References → section_aware excludes),
  `a001` (Chinese→English bilingual mismatch). Not fixable without breaking the
  References-exclusion rule or the frozen retrieval baseline.
- ANSWERABILITY_MISCLASSIFICATION ×3: one out-of-scope question the LLM answers
  (deterministic out-of-scope guard was tried and reverted — net-negative), and
  two false-premise cases whose contradicting evidence is not retrieved.

## Eval integrity
- dev.jsonl sha256 = 44D209E0... unchanged (== Phase D).
- test.jsonl sha256 = D82A0207... unchanged (== Phase D).
- TEST never read / run / modified. No gold / split / case_id change.

## Verdict
**READY_FOR_FINAL_TEST** — DEV answerability improved 0.429 → 0.714 with no
regression on ANSWERABLE (0 gate-rejected-despite-hit) and no leakage. TEST
remains the next phase's explicit, one-time action.
