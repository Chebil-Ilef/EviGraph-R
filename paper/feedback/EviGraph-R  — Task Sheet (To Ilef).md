
**Suggested target: EACL 2027 System Demonstrations (Athens, March 9–14, 2027).** Expected deadline ~late November 2026 (CFP not yet published — Jingbo monitors weekly). **Fallback: ECIR 2027 demo, deadline November 2, 2026.** Venue decision: by **October 20.** 

**no re-indexing, no month-long re-evaluation.** The demo paper is 4–6 pages about the system you already built. The plan front-loads everything that needs _you specifically_ into July–August, so your industry start doesn't block anything later. 

Jingbo owns in parallel: venue monitoring, related work + comparison section, demo deployment technique coordinate (if needed), all editing,  and submission. 

---

## Phase 0 — Correctness fixes (Tue July 14 – Fri July 17)

Everything downstream needs correct numbers. Only need recomputation and log analysis, no new pipeline runs.

**T1. Qrels + NDCG audit (~1 day).** Locate the metric code for Table V.5. Confirm single-gold qrels (MAP≡MRR at every k already implies it). Verify the suspected bug: IDCG padded with "partial (0.5)" documents while the ranking has one binary gold. Fix by dropping NDCG and reporting **Hit@k (Success@k), MRR, Recall@k** from the saved per-query rankings. Keep before/after tables in a notes file.

**T2. Per-metric ablation export (~half a day, from existing logs).** Re-export Table V.9 per metric (the 3 SQuAI-comparable vs the 2 custom) × category. Key check: does the flat-chunks drop survive on the shared three metrics? Flag the answer to Jingbo by July 17, it decides how we phrase the demo's evaluation blurb.

**T3. Sparse-only sanity check (~2–3 h).** R2 ran in 4.5 s vs ~40–50 s elsewhere, please check logs for silent failure (empty sparse retrievals). Rerun properly if broken.

**T4. Model & settings table (~half a day).** Component → model/version → provider → temperature/decoding → prompt version → deterministic/sampled → seed, for every LLM in the pipeline plus embedders/rerankers/NLI. Add hardware + GPU hours. Best written now while everything is fresh.

**T5. SQuAI correction (~1 h).** Fix every statement describing SQuAI as abstract-only — the CIKM 2025 paper has both abstract and full-text configurations.

---

## Phase 1 — System hardening + knowledge transfer (July 20 – August 7)

This is the heart of the demo strategy: reviewers will click the link and watch the video, likely in December–February.

**T6. Stable public deployment (~3–4 days).** Pin dependencies, Docker image rebuilt from clean checkout, uptime monitoring/alerting (Langfuse + a simple ping), and a degraded-mode plan: cached results for a set of showcase queries so the demo never returns an error to a reviewer even if a backend component hiccups.

**T7. Package + docs polish (~2 days).** PyPI package installs cleanly on a fresh machine; README with quickstart, architecture figure, and screenshots; choose and declare a license (demo review forms ask explicitly).

**T8. Showcase queries (~1 day).** Curate 8–12 example questions across the four categories that produce visually rich evidence graphs and correct answers. These drive the screencast, the paper's figures.

**T9. Knowledge-transfer document (~1–2 days, critical).** How to rerun indexing, retrieval eval, and the full benchmark; where indices, logs, qrels, and credentials live; how to redeploy (Please include everything that related to the project). This is what lets Jingbo maintain the demo after you start your new job. Please treat this as non-optional.

**T10. Screencast (~1 day).** ≤2.5 minutes: question → decomposition → graph builds up → claim verification badges → per-sentence citations → clicking a claim node to its source chunk. Script it around one showcase query; Jingbo reviews the cut.

---

## Phase 2 — Demo paper draft (August 10 – September 4)

**T11. Draft the demo paper (~1 week of focused work, spread over the month).** Structure: (1) problem + who it's for; (2) system architecture (compressed from thesis Ch. IV, one pipeline figure); (3) the evidence-graph UI walkthrough with screenshots; (4) comparison with existing systems, Jingbo drafts this table (Ai2 Scholar QA, OpenScholar, SciRAG, PaperQA2, SQuAI) on system/UI dimensions; (5) evaluation summary: corrected retrieval table (T1) + the ablation headline with the per-metric caveat handled (T2) — demo tracks require "some form of evaluation," and this comfortably clears it; (6) availability: demo URL, PyPI, repo, license.


---

## Phase 4 — Venue decision + submission (October 13 – early December)

- **October 20:** EACL 2027 demo CFP published with workable deadline? → EACL demo. Not published? → **ECIR 2027 demo, submit by November 2** (4 pages — trim T11 draft; keep the video).
- Final polish, screencast link check, demo uptime check and submit.
- **Through February:** demo stays online for reviewers; respond to any reviewer issues; camera-ready after notification.

---