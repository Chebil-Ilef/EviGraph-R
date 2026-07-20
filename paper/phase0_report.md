# Phase 0 — Correctness Fixes: Report

---

## T1 — Qrels + NDCG Audit

### The bug

`_ndcg()` in [`experiments/index_comparison/run_comparison.py`](../experiments/index_comparison/run_comparison.py)
(lines 57–73) assigned relevance 0.5 to any candidate paper merely sharing a topic cluster
with the gold paper, then built IDCG assuming several such 0.5-relevance documents could
ideally fill the ranking:

```python
def rel(pid: str) -> float:
    if pid in gold_ids:
        return 1.0
    if pid in same_cluster_ids:
        return 0.5   # fabricated — never a real judged relevance grade
    return 0.0
```

These 0.5 labels were never assigned by a human or verified judge — they're a proxy (same
cluster ⇒ assume partially relevant) applied uniformly, regardless of whether that paper
actually answers the specific question.

### Verification

- `results/questions.jsonl`: **113 questions**, `gold_paper_ids` length distribution =
  `{1: 113}` → every question has exactly **one** gold paper (single-gold qrels, confirmed
  empirically).
- Compared `map@k` vs `mrr@k` across all 113×2 systems×3 k-values (678 values) in
  `raw_results.json`: **0 mismatches**. MAP≡MRR exactly, as expected for single-relevant-doc
  retrieval — MAP adds no information beyond MRR here.
- Symptom of the bug: EviGraph-R **NDCG@5 (0.303) < NDCG@1 (0.504)** — impossible for a
  well-defined single-gold NDCG. Caused by IDCG@5 being inflated with fictitious 0.5-relevance
  cluster documents that DCG@5 never gets credit for retrieving.

### Decision

Drop NDCG from the reported retrieval table. Report **Hit@k, MRR@k, Recall@k** instead — all
valid under the single verified gold label, with no fabricated relevance grades. `Recall@k ≡
Hit@k` here (0 or 1 relevant document per query), reported under both names since reviewers
expect "Recall" by that name. This is a reporting fix, not a re-annotation project — real
graded NDCG would need new human/LLM-judge partial-relevance annotation, out of Phase 0 scope.

### Before → After (Table V.5, overall, N=113)

| Metric | SQuAI k=1 | k=5 | k=10 | EviGraph-R k=1 | k=5 | k=10 |
|---|---|---|---|---|---|---|
| Hit@k | 0.204 | 0.319 | 0.345 | 0.504 | 0.637 | 0.646 |
| MRR@k | 0.204 | 0.247 | 0.250 | 0.504 | 0.560 | 0.561 |
| Recall@k | 0.204 | 0.319 | 0.345 | 0.504 | 0.637 | 0.646 |
| ~~NDCG@k~~ | ~~0.204~~ | ~~0.138~~ | ~~0.142~~ | ~~0.504~~ | ~~0.303~~ | ~~0.305~~ |
| Avg Rank of Gold | 2.205 | — | — | 1.411 | — | — |

Section Hit (EviGraph-R only, no SQuAI equivalent — abstract-only indexing has no section
granularity): 0.377 / 0.481 / 0.494 at k=1/5/10.

### Breakdown by domain (Hit@k / MRR@k / Recall@k)

**CS** (N=39)

| k | SQuAI Hit | SQuAI MRR | SQuAI Recall | EviGraph-R Hit | EviGraph-R MRR | EviGraph-R Recall |
|---|---|---|---|---|---|---|
| 1 | 0.256 | 0.256 | 0.256 | 0.641 | 0.641 | 0.641 |
| 5 | 0.359 | 0.291 | 0.359 | 0.769 | 0.693 | 0.769 |
| 10 | 0.385 | 0.294 | 0.385 | 0.795 | 0.697 | 0.795 |

**Physics** (N=40)

| k | SQuAI Hit | SQuAI MRR | SQuAI Recall | EviGraph-R Hit | EviGraph-R MRR | EviGraph-R Recall |
|---|---|---|---|---|---|---|
| 1 | 0.150 | 0.150 | 0.150 | 0.375 | 0.375 | 0.375 |
| 5 | 0.250 | 0.188 | 0.250 | 0.475 | 0.421 | 0.475 |
| 10 | 0.275 | 0.191 | 0.275 | 0.475 | 0.421 | 0.475 |

**Math** (N=34)

| k | SQuAI Hit | SQuAI MRR | SQuAI Recall | EviGraph-R Hit | EviGraph-R MRR | EviGraph-R Recall |
|---|---|---|---|---|---|---|
| 1 | 0.206 | 0.206 | 0.206 | 0.500 | 0.500 | 0.500 |
| 5 | 0.353 | 0.265 | 0.353 | 0.676 | 0.570 | 0.676 |
| 10 | 0.382 | 0.268 | 0.382 | 0.676 | 0.570 | 0.676 |

### Breakdown by question source

| Source | N | SQuAI Hit@5 | EviGraph-R Hit@5 | SQuAI MRR@10 | EviGraph-R MRR@10 |
|---|---|---|---|---|---|
| section | 38 | 0.079 | 0.526 | 0.069 | 0.465 |
| fullpaper | 39 | 0.179 | 0.564 | 0.162 | 0.500 |
| abstract | 36 | 0.722 | 0.833 | 0.536 | 0.729 |

*Section Hit reported for EviGraph-R only — SQuAI indexes abstracts with no section granularity.*

**Headline unaffected:** Hit@1 0.204→0.504, Hit@5 0.319→0.637 — the EviGraph-R vs SQuAI story
(full-text hybrid retrieval substantially outperforms abstract-only) is intact and, if
anything, cleaner without the NDCG row inviting reviewer scrutiny.

---

## T2 / T3 — Per-Metric Ablation Export & Sparse-Only (R2) Sanity Check

T2 and T3 share the same source table (`evaluation/data/eval/agg_*.json`, Table V.9), so
they're reported together here. T2 asks whether ablation drops survive on SQuAI-comparable
metrics; T3 investigates why R2 specifically looked anomalous — the two questions turned out
to be linked (R2 was silently broken, which is exactly the kind of thing T2's per-metric
breakdown exposes).


### T2 — the question

Table V.9 reports one number per configuration per category: `mean_5`, the average of 5
metrics — 3 shared with SQuAI (Answer Relevancy, Contextual Relevancy, Faithfulness) and 2
custom to EviGraph-R (Claim Coverage, Attribution Faithfulness). Key check: **does the
flat-chunks (G1) drop survive on the 3 shared metrics alone, or is it driven by the 2 custom
metrics?**

### Overall per-metric table (all 13 configurations)

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

*R2's numbers above are from the corrected re-run (see T3 below) — the original broken
configuration reported 0.328 overall at 4.5s latency, which measured a silent failure, not
real sparse-only retrieval quality.*

### Per-category breakdown (all 13 configurations)

**Category 1**

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

**Category 2**

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

**Category 3**

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

**Category 4**

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

### T2 — answer: partially — the headline number is inflated by the custom metrics

**G1 Flat chunks — overall**

| | mean3 (shared) | mean5 (current report) | Claim Coverage | Attribution Faithfulness |
|---|---|---|---|---|
| Full system | 0.8572 | 0.7983 | 0.5249 | 0.8948 |
| G1 (flat chunks) | 0.7861 | 0.4845 | 0.0075 | 0.0567 |

- On the 3 shared metrics: 0.8572 → 0.7861 (**−0.0711, −8.3%**) — a real, moderate drop.
- On all 5 metrics: 0.7983 → 0.4845 (**−0.3138, −39.3%**) — the number currently in the report.

This pattern holds in every category (1–4), not just on average.

**Why the custom-metric collapse is partly mechanical, not purely quality.** Claim Coverage
and Attribution Faithfulness ([`evaluation/utils/metrics.py`](../evaluation/utils/metrics.py))
check whether a *cited claim* is grounded in a *specific chunk*. G1 replaces the evidence
graph with a flat chunk list — there's no claim/citation structure left for these metrics to
score, so a near-zero result partly measures "does the artifact this metric was designed for
still exist," not purely "is the answer worse." The shared-metric drop (−8.3%) is the more
defensible number to lead with for a reviewer comparing against SQuAI's metric set.

**Second finding, now confirmed: G1 is the largest real drop, not R2.** Before the T3 fix, R2
(sparse-only) appeared to show a much larger shared-metric drop than G1 (−36.4% vs −8.3%), but
that R2 number was measuring a silent failure, not real sparse-only quality (see T3). With the
bug fixed and R2 re-run properly, its actual shared-metric score is 0.837 — only a **−2.4%**
drop from the full system's 0.857, far smaller than G1's real −8.3% drop, and close to R1
(dense-only retrieval, 0.832). In other words: once retrieval actually works, dropping either
half of hybrid retrieval barely hurts standard-metric quality, while dropping the evidence
graph does. **G1 (removing the evidence graph) should be the headline ablation finding**, not
sparse-only retrieval.

**Recommendation for the demo's evaluation blurb:**
1. Report mean3 and the 2 custom metrics **separately**, not blended into one mean_5 headline.
2. Scope the evidence-graph claim accurately: *"removing the evidence graph causes a
   measurable drop on standard RAG metrics (−8.3%) and a much larger drop on
   claim/attribution-specific metrics we introduce (−94%+), which is expected since those
   metrics measure graph-derived structure directly."*
3. A1.1 (no decomposition) and G2 (no citation expansion) still slightly outperform the full
   system on both mean3 and mean5 — unaffected by this audit; see the Teuken-7B caveat in T4
   for why A1.1 specifically needs an extra asterisk.

---

## T3 — Sparse-only retrieval variant was silently broken

**What we found:** The sparse-only configuration finished suspiciously fast (4-5s per question
vs. 40-80s for everything else) because it was silently retrieving nothing and falling back to
a generic "insufficient evidence" answer every time, for all 268 questions.

**The bug:** The ablation was querying the search index the wrong way for the sparse-only
case, so it never returned real results. We fixed it and re-ran all 268 questions.

| | Standard-metrics avg | Δ vs full system | Overall score | Latency |
|---|---|---|---|---|
| Full system | 0.857 | — | 0.798 | 49.8s |
| R2 Sparse only — broken (original) | 0.546 | −36.4% | 0.328 | 4.5s |
| R2 Sparse only — corrected (re-run) | 0.837 | **−2.4%** | 0.775 | 34.5s |
| R1 Dense only (for comparison) | 0.832 | −3.0% | 0.777 | 37.0s |

Once fixed, sparse-only retrieval barely underperforms the full system and lands close to
dense-only — a much smaller, less alarming gap than the broken run suggested. Removing the
evidence graph (T2) is the larger, more important drop.

---

## T4 — Model & Settings Table

### Pipeline LLM agents

| Component | Model | Provider | Temperature | Decoding | Timeout | Max retries |
|---|---|---|---|---|---|---|
| Decomposer (Agent 1) | `meta-llama/Llama-3.1-8B-Instruct`¹ | ScaDS.AI (remote) | 0.0 | greedy | 60s | 2 |
| Evidence Graph Builder (Agent 2) | `meta-llama/Llama-3.3-70B-Instruct` | ScaDS.AI (remote) | 0.0 | greedy | 90s | 2 |
| Judge — LLM tier (Agent 3) | `meta-llama/Llama-3.1-8B-Instruct`¹ | ScaDS.AI (remote) | 0.0 | greedy | 60s | 2 |
| Answer Generator (Agent 4) | `meta-llama/Llama-3.3-70B-Instruct` | ScaDS.AI (remote) | 0.0 | greedy | 120s | 2 |
| DeepEval / G-Eval judge (evaluation-only) | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | ScaDS.AI (remote) | 0.0 | greedy | — | — |


### Local models (embedder / reranker / NLI / classifier)

| Component | Model | Notes |
|---|---|---|
| Embedder | `BAAI/bge-m3` | Dense (1024-dim) + native sparse, single forward pass, fp16 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CPU, this machine |
| NLI (Judge tier) | `sileod/deberta-v3-small-tasksource-nli` | Support threshold 0.65, contradiction 0.70 |
| Citation classifier | `lostelf/scibert_scivocab_uncased_scicite_finetuned` | METHOD / BACKGROUND / RESULT_COMPARISON |
| IMRaD section classifier (indexing-time only) | `lostelf/section-classifier-imrad` | DistilBERT-based, F1 0.776, accuracy 77.1% |

### Hardware and compute budget

Two separate environments — conflating them would misreport GPU-hours.

**Indexing** (2.28M papers → 69M chunks) — Capella, ZIH TU Dresden: 4× NVIDIA H100/node, 200
parallel SLURM tasks × 1 GPU + 8 CPU cores + 170GB RAM each, 24h wall-time cap.
**GPU-hours: 200 × 24h × 1 GPU = 4,800 GPU-hours upper bound** — actual usage was less (tasks
finished under the cap), but the real figure was never calculated; would need Capella `sacct`
job accounting to pin down.

**Evaluation/ablation study** (268-question benchmark, all 13 configs) — Barnard, **CPU-only**
(14 cores, 100GB RAM, no GPU). **Local GPU-hours: 0.** All local compute (embedding,
reranking, NLI) ran on CPU; the only GPU compute involved was the remote ScaDS.AI endpoint
serving the LLM calls — provider-managed, not independently measurable.

---

## T5 — SQuAI correction

**What was wrong:** The report describes SQuAI, our predecessor system, as an "abstracts
only" system in several places, including the related-work comparison table. This isn't
accurate. SQuAI's published CIKM 2025 paper supports two modes: it can answer purely from
abstracts, or it can retrieve and answer from full-text papers as well. Describing it as
abstract-only overstates how different EviGraph-R's full-text retrieval is from SQuAI's own
capabilities, which isn't a fair comparison to publish.

**What we should do in the paper:** The accurate
description: SQuAI indexes paper abstracts and, from that same abstract-level index, can
either (a) answer directly from the abstracts, or (b) use the abstracts to identify and
retrieve the relevant full-text papers, then answer from that full text. The correction
affects the related-work comparison table and a few descriptive sentences elsewhere in the
report — no results or numbers change, this is a factual accuracy fix to how SQuAI is
described, not a re-evaluation.

**Note — which SQuAI mode our own evaluations actually used:** 
For the retrieval-only
comparison (T1), we stopped at the retrieval step and never invoked SQuAI's answer-generation
mode at all, so this distinction doesn't affect that comparison. 

For the full-system and
ablation study, we compared against SQuAI running in its full-text mode — its strongest and
most capable configuration, not the weaker abstract-only mode.