# Table V.9 — Per-Metric Ablation Export (T2)

*All 5 metrics, overall and per category, for every configuration. Source: `evaluation/data/eval/agg_*.json` (unchanged, no new runs).*

## Overall

| Configuration | AnsRel | CtxRel | Faith | ClaimCov | AttrFaith | mean3 (shared) | mean2 (custom) | mean5 (Table V.9) | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| Standard RAG | 0.8133 | 0.8096 | 0.9422 | 0.4788 | 0.4004 | 0.8550 | 0.4396 | 0.6889 | 24.1 |
| SQuAI | 0.9263 | 0.8483 | 0.9482 | 0.4050 | 0.6344 | 0.9076 | 0.5197 | 0.7524 | 67.3 |
| EviGraph-R (full) | 0.8439 | 0.8012 | 0.9265 | 0.5249 | 0.8948 | 0.8572 | 0.7099 | 0.7983 | 49.8 |
| A1.1 No decomposition | 0.8691 | 0.8395 | 0.9277 | 0.5506 | 0.8511 | 0.8788 | 0.7008 | 0.8076 | 44.8 |
| A1.2 No budget weights | 0.8561 | 0.7945 | 0.9339 | 0.5271 | 0.8787 | 0.8615 | 0.7029 | 0.7981 | 49.0 |
| R1 Dense only | 0.8440 | 0.7276 | 0.9233 | 0.5266 | 0.8608 | 0.8316 | 0.6937 | 0.7765 | 37.0 |
| R2 Sparse only (corrected) | 0.8361 | 0.7535 | 0.9214 | 0.5360 | 0.8266 | 0.8370 | 0.6813 | 0.7747 | 34.5 |
| R3 No section boosting | 0.8447 | 0.8067 | 0.9174 | 0.5335 | 0.8683 | 0.8563 | 0.7009 | 0.7941 | 55.1 |
| G1 Flat chunks | 0.6368 | 0.7216 | 1.0000 | 0.0075 | 0.0567 | 0.7861 | 0.0321 | 0.4845 | 31.0 |
| G2 No citation expan. | 0.8463 | 0.8055 | 0.9323 | 0.5411 | 0.8985 | 0.8614 | 0.7198 | 0.8047 | 44.5 |
| J1 No judge | 0.8581 | 0.7864 | 0.9177 | 0.5434 | 0.8678 | 0.8541 | 0.7056 | 0.7947 | 47.5 |
| J2 NLI only | 0.8496 | 0.7930 | 0.9279 | 0.5436 | 0.8851 | 0.8568 | 0.7144 | 0.7998 | 48.7 |
| J3 LLM only | 0.8451 | 0.7991 | 0.9249 | 0.5271 | 0.9045 | 0.8564 | 0.7158 | 0.8001 | 77.2 |

## Category 1

| Configuration | AnsRel | CtxRel | Faith | ClaimCov | AttrFaith | mean3 (shared) | mean2 (custom) | mean5 |
|---|---|---|---|---|---|---|---|---|
| Standard RAG | 0.7458 | 0.8050 | 0.9572 | 0.4452 | 0.2625 | 0.8360 | 0.3538 | 0.6431 |
| SQuAI | 0.9198 | 0.8344 | 0.9552 | 0.3673 | 0.6492 | 0.9031 | 0.5082 | 0.7452 |
| EviGraph-R (full) | 0.8144 | 0.8100 | 0.9246 | 0.4585 | 0.8547 | 0.8497 | 0.6566 | 0.7724 |
| A1.1 No decomposition | 0.8328 | 0.8282 | 0.9207 | 0.5590 | 0.7406 | 0.8606 | 0.6498 | 0.7763 |
| A1.2 No budget weights | 0.8051 | 0.8045 | 0.9366 | 0.4385 | 0.8406 | 0.8487 | 0.6396 | 0.7651 |
| R1 Dense only | 0.8373 | 0.7246 | 0.9169 | 0.4966 | 0.7781 | 0.8263 | 0.6373 | 0.7507 |
| R2 Sparse only (corrected) | 0.8018 | 0.7455 | 0.8779 | 0.5169 | 0.7429 | 0.8084 | 0.6299 | 0.7370 |
| R3 No section boosting | 0.7794 | 0.8189 | 0.9158 | 0.5113 | 0.8172 | 0.8380 | 0.6643 | 0.7685 |
| G1 Flat chunks | 0.5469 | 0.7913 | 1.0000 | 0.0190 | 0.0406 | 0.7794 | 0.0298 | 0.4796 |
| G2 No citation expan. | 0.8218 | 0.8162 | 0.9258 | 0.4852 | 0.7875 | 0.8546 | 0.6363 | 0.7673 |
| J1 No judge | 0.8074 | 0.8381 | 0.9075 | 0.4967 | 0.8391 | 0.8510 | 0.6679 | 0.7778 |
| J2 NLI only | 0.8071 | 0.8104 | 0.9035 | 0.4935 | 0.8875 | 0.8403 | 0.6905 | 0.7804 |
| J3 LLM only | 0.8186 | 0.8025 | 0.9185 | 0.4542 | 0.8578 | 0.8465 | 0.6560 | 0.7703 |

## Category 2

| Configuration | AnsRel | CtxRel | Faith | ClaimCov | AttrFaith | mean3 (shared) | mean2 (custom) | mean5 |
|---|---|---|---|---|---|---|---|---|
| Standard RAG | 0.8217 | 0.7788 | 0.9027 | 0.4828 | 0.4375 | 0.8344 | 0.4602 | 0.6847 |
| SQuAI | 0.8988 | 0.8273 | 0.9314 | 0.4271 | 0.6898 | 0.8858 | 0.5585 | 0.7549 |
| EviGraph-R (full) | 0.8209 | 0.8022 | 0.9257 | 0.5484 | 0.8000 | 0.8496 | 0.6742 | 0.7794 |
| A1.1 No decomposition | 0.8511 | 0.8643 | 0.9130 | 0.5891 | 0.8719 | 0.8761 | 0.7305 | 0.8179 |
| A1.2 No budget weights | 0.8583 | 0.7926 | 0.9366 | 0.5453 | 0.8219 | 0.8625 | 0.6836 | 0.7909 |
| R1 Dense only | 0.8411 | 0.7417 | 0.9050 | 0.5297 | 0.8828 | 0.8293 | 0.7063 | 0.7801 |
| R2 Sparse only (corrected) | 0.8112 | 0.7948 | 0.9225 | 0.5419 | 0.8250 | 0.8428 | 0.6835 | 0.7791 |
| R3 No section boosting | 0.8343 | 0.8014 | 0.9202 | 0.5375 | 0.8328 | 0.8520 | 0.6851 | 0.7852 |
| G1 Flat chunks | 0.5521 | 0.6840 | 1.0000 | 0.0000 | 0.0812 | 0.7454 | 0.0406 | 0.4635 |
| G2 No citation expan. | 0.8003 | 0.8000 | 0.9360 | 0.5469 | 0.9203 | 0.8454 | 0.7336 | 0.8007 |
| J1 No judge | 0.8079 | 0.7628 | 0.9094 | 0.5453 | 0.8391 | 0.8267 | 0.6922 | 0.7729 |
| J2 NLI only | 0.8356 | 0.7842 | 0.9510 | 0.5250 | 0.8422 | 0.8569 | 0.6836 | 0.7876 |
| J3 LLM only | 0.8213 | 0.8005 | 0.9376 | 0.5406 | 0.8219 | 0.8531 | 0.6812 | 0.7844 |

## Category 3

| Configuration | AnsRel | CtxRel | Faith | ClaimCov | AttrFaith | mean3 (shared) | mean2 (custom) | mean5 |
|---|---|---|---|---|---|---|---|---|
| Standard RAG | 0.8302 | 0.8192 | 0.9585 | 0.4757 | 0.4586 | 0.8693 | 0.4672 | 0.7084 |
| SQuAI | 0.9401 | 0.8503 | 0.9399 | 0.3879 | 0.5788 | 0.9101 | 0.4834 | 0.7394 |
| EviGraph-R (full) | 0.8653 | 0.7911 | 0.9173 | 0.5129 | 0.9486 | 0.8579 | 0.7308 | 0.8070 |
| A1.1 No decomposition | 0.8820 | 0.8290 | 0.9291 | 0.4943 | 0.9029 | 0.8800 | 0.6986 | 0.8075 |
| A1.2 No budget weights | 0.8772 | 0.8092 | 0.9336 | 0.5314 | 0.9357 | 0.8733 | 0.7335 | 0.8174 |
| R1 Dense only | 0.8151 | 0.7050 | 0.9352 | 0.5357 | 0.9143 | 0.8184 | 0.7250 | 0.7811 |
| R2 Sparse only (corrected) | 0.8571 | 0.7332 | 0.9559 | 0.5443 | 0.8700 | 0.8487 | 0.7071 | 0.7921 |
| R3 No section boosting | 0.9036 | 0.8235 | 0.9046 | 0.5229 | 0.9271 | 0.8772 | 0.7250 | 0.8163 |
| G1 Flat chunks | 0.7476 | 0.7203 | 1.0000 | 0.0071 | 0.0600 | 0.8226 | 0.0335 | 0.5070 |
| G2 No citation expan. | 0.8963 | 0.8039 | 0.9382 | 0.5229 | 0.9257 | 0.8795 | 0.7243 | 0.8174 |
| J1 No judge | 0.8904 | 0.8034 | 0.9205 | 0.5500 | 0.9029 | 0.8714 | 0.7265 | 0.8134 |
| J2 NLI only | 0.8657 | 0.8091 | 0.9343 | 0.5557 | 0.8971 | 0.8697 | 0.7264 | 0.8124 |
| J3 LLM only | 0.8677 | 0.7831 | 0.9137 | 0.5257 | 0.9671 | 0.8548 | 0.7464 | 0.8115 |

## Category 4

| Configuration | AnsRel | CtxRel | Faith | ClaimCov | AttrFaith | mean3 (shared) | mean2 (custom) | mean5 |
|---|---|---|---|---|---|---|---|---|
| Standard RAG | 0.8545 | 0.8314 | 0.9487 | 0.5111 | 0.4381 | 0.8782 | 0.4746 | 0.7168 |
| SQuAI | 0.9440 | 0.8771 | 0.9654 | 0.4355 | 0.6270 | 0.9288 | 0.5312 | 0.7698 |
| EviGraph-R (full) | 0.8707 | 0.8024 | 0.9384 | 0.5714 | 0.9643 | 0.8705 | 0.7679 | 0.8294 |
| A1.1 No decomposition | 0.9058 | 0.8385 | 0.9454 | 0.5643 | 0.8814 | 0.8966 | 0.7228 | 0.8271 |
| A1.2 No budget weights | 0.8795 | 0.7724 | 0.9294 | 0.5841 | 0.9086 | 0.8604 | 0.7463 | 0.8148 |
| R1 Dense only | 0.8816 | 0.7399 | 0.9342 | 0.5400 | 0.8629 | 0.8519 | 0.7015 | 0.7917 |
| R2 Sparse only (corrected) | 0.8692 | 0.7450 | 0.9229 | 0.5386 | 0.8600 | 0.8457 | 0.6993 | 0.7871 |
| R3 No section boosting | 0.8550 | 0.7834 | 0.9293 | 0.5600 | 0.8886 | 0.8559 | 0.7243 | 0.8033 |
| G1 Flat chunks | 0.6857 | 0.6962 | 1.0000 | 0.0043 | 0.0457 | 0.7940 | 0.0250 | 0.4864 |
| G2 No citation expan. | 0.8606 | 0.8021 | 0.9288 | 0.6029 | 0.9529 | 0.8638 | 0.7779 | 0.8295 |
| J1 No judge | 0.9179 | 0.7441 | 0.9307 | 0.5757 | 0.8857 | 0.8642 | 0.7307 | 0.8108 |
| J2 NLI only | 0.8847 | 0.7695 | 0.9231 | 0.5929 | 0.9100 | 0.8591 | 0.7514 | 0.8160 |
| J3 LLM only | 0.8684 | 0.8109 | 0.9308 | 0.5786 | 0.9600 | 0.8700 | 0.7693 | 0.8297 |

# T2 — Per-Metric Ablation Audit: Notes

## The question

Table V.9 reports one number per configuration per category: `mean_5`, the average of 5 metrics — 3 shared with SQuAI (Answer Relevancy, Contextual Relevancy, Faithfulness) and 2 custom to EviGraph-R (Claim Coverage, Attribution Faithfulness). Key check: **does the flat-chunks (G1) drop survive on the 3 shared metrics alone, or is it driven by the 2 custom metrics?**

## Answer: partially — the headline number is inflated by the custom metrics

### G1 Flat chunks — overall

| | mean3 (shared) | mean5 (all 5, current report) | Claim Coverage | Attribution Faithfulness |
|---|---|---|---|---|
| Full system | 0.8572 | 0.7983 | 0.5249 | 0.8948 |
| G1 (flat chunks) | 0.7861 | 0.4845 | 0.0075 | 0.0567 |

- On the 3 shared metrics: 0.8572 → 0.7861 (**-0.0711, -8.3%**) — a real, moderate drop.
- On all 5 metrics: 0.7983 → 0.4845 (**-0.3138, -39.3%**) — the number currently in the report.
- Claim Coverage collapses 0.5249 → 0.0075; Attribution Faithfulness collapses 0.8948 → 0.0567.

This pattern holds in every category (1–4), not just on average — Claim Coverage and Attribution Faithfulness both crash to near-zero for G1 across all four categories, while the 3 shared metrics dip only modestly. See the per-category tables above for the full breakdown.

## Why the custom-metric collapse is partly mechanical, not purely quality

Claim Coverage and Attribution Faithfulness are G-Eval metrics that check whether a *cited claim* is grounded in a *specific chunk* via claim-to-chunk attribution (see `evaluation/utils/metrics.py`). G1 replaces the evidence graph with a flat chunk list — there is no claim/citation structure left for these metrics to score, so they don't just detect "worse answers," they partly measure "does the artifact this metric was designed for still exist." A reviewer could reasonably call this circular: removing the graph tanks a metric built to measure the graph.

This does not mean the custom metrics are invalid — they're honest measurements of what they claim to measure — but the *headline framing* ("largest quality drop among all ablations", currently −39% via mean_5) should not lean on them alone. The shared-metric drop (−8.3%) is the more defensible number to lead with for a reviewer comparing against SQuAI's metric set.

## Second finding, corrected after the T3 fix: G1 is the largest real drop, not R2

R2 was initially broken (see T3 audit notes) — its original numbers below measured a silent
retrieval failure, not real sparse-only quality. After the fix and a full 268-question re-run,
scored with the same DeepEval judge:

| Variant | mean3 (shared) | Δ vs full | mean5 | Δ vs full |
|---|---|---|---|---|
| G1 Flat chunks | 0.7861 | -0.0711 (-8.3%) | 0.4845 | -0.3138 (-39.3%) |
| R2 Sparse only — broken (original) | 0.5456 | -0.3116 (-36.4%) | 0.3283 | -0.4700 (-58.9%) |
| R2 Sparse only — corrected (re-run) | 0.8370 | -0.0203 (-2.4%) | 0.7747 | -0.0233 (-2.9%) |
| R1 Dense only (for comparison) | 0.8316 | -0.0257 (-3.0%) | 0.7765 | -0.0215 (-2.7%) |

With the bug fixed, **R2's real drop (-2.4% on shared metrics) is far smaller than G1's
(-8.3%)** and close to R1 (dense-only) — consistent with hybrid retrieval combining two
largely redundant signals, so losing either leg alone costs little. **G1 (removing the
evidence graph) is the largest real quality drop in the ablation study.** The earlier
conclusion that R2 was the largest drop was an artifact of the retrieval bug, not a genuine
finding, and should not be used.

## Recommendation for the demo's evaluation blurb

1. Report mean3 and the 2 custom metrics **separately**, not blended into one mean_5 headline — matches the task sheet's framing and preempts the obvious reviewer objection.
2. Lead with the evidence-graph result, scoped accurately: *"removing the evidence graph causes a measurable drop on standard RAG metrics (−8.3%) — the largest real drop among all ablations — and a much larger drop on claim/attribution-specific metrics we introduce (−94%+), which is expected since those metrics measure graph-derived structure directly."*
3. Sparse-only and dense-only retrieval both perform close to the full hybrid system (~2-3% below) once correctly measured — cite this as evidence that the evidence graph, not the retrieval fusion strategy, is the load-bearing component.
4. A1.1 (no decomposition) and G2 (no citation expansion) still slightly outperform the full system on both mean3 and mean5 — unaffected by this audit, the "agentic components don't always help" finding stands as reported.

