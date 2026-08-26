# Evaluation

## Final acceptance methodology

The v1.0.0 release was evaluated once on a frozen, fresh 32-case acceptance
set. Its gold-paper IDs are paper-disjoint from Eval V2. Every evidence-bearing
case was re-located against the frozen local PDFs; validation reported **0
errors**.

- Class distribution: ANSWERABLE 14, AMBIGUOUS 4, NO_EVIDENCE 5,
  PARTIAL_EVIDENCE 3, FALSE_PREMISE 4, CONDITIONALLY_DIVERGENT 2.
- Fresh gold-paper IDs: 9.
- Frozen acceptance input SHA-256:
  `a1cc901ad8cda834325f073837385e2daf4d488a28e8a9ebc77fd134a2c2d9e5`.
- Frozen runtime: `section_aware_270_gpu` + BGE-M3 Dense/Sparse + RRF
  (`section_hybrid`), with the final answerability and citation pipeline.

## Pre-fixed release gates and result

| Metric | Gate | Result | Status |
| --- | ---: | ---: | --- |
| Recall@10 | ≥ 70.0% | 91.3% | pass |
| PageHit@10 | ≥ 55.0% | 91.3% | pass |
| EvidenceSpanHit@10 | ≥ 50.0% | 87.0% | pass |
| Action Accuracy | ≥ 60.0% | 75.0% | pass |
| ANSWERABLE Accuracy | ≥ 55.0% | 71.4% | pass |
| NO_EVIDENCE Recall | ≥ 80.0% | 100% | pass |

Verdict: **FINAL_ACCEPT_WITH_LIMITATIONS**.

## Citation and unsupported-claim interpretation

Automatic Citation Support was 61.9% and the automatic Unsupported Claim metric
was 42.86%. These are conservative validator outputs, not a hallucination rate.
They are sensitive to quick-mode 1,200-token truncation, bibliographic-claim
detection, and citation normalization.

The evidence-level audit was 10 SUPPORTED / 1 flagged UNSUPPORTED. The sole
flag was `34.64 mg/g` versus `34.64mg/g`, a whitespace-normalization artifact.
The release does not claim zero hallucination or perfect citation support.

## Scope and interpretation

Eval V2 held-out results were consumed before the release. Later V2 runs are
regression/postmortem material, not fresh generalization claims. The final
acceptance set is the release evidence; it does not publish paper text or gold
evidence.

The conflict-routing postmortem is documented as an engineering decision:
keyword-based auto-routing produced 6 CONFLICT_OFF wins and 1 loss, so the
unreliable automatic classifier was removed. Condition-dependent evidence is
presented at the answer layer with provenance instead of being forced into a
binary conflict action.

See the [release overview](../README.md) and the explicit
[system limitations](../LIMITATIONS.md).
