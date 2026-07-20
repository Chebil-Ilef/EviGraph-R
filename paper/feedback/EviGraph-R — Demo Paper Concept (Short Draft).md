
_Working concept for the EACL 2027 System Demonstrations submission (ECIR 2027 demo as fallback)._

---

## 1. Working title

**EviGraph-R: Interactive Claim-Level Evidence Graphs for Scientific Question Answering over 2.28M Full-Text Papers**

Alternatives:

- _EviGraph-R: Verifying Scientific Answers through Interactive Claim-Level Evidence Graphs_
- _From Citations to Evidence: EviGraph-R, a Transparent Multi-Agent System for Scientific QA at Scale_

---

## 2. Draft abstract (paper-ready prose)

> Scientific question answering systems increasingly provide citations, but citations alone do not let users verify _why_ an answer should be trusted. We present **EviGraph-R**, a multi-agent retrieval-augmented generation system for scientific QA over 2.28M full-text arXiv papers (unarXive 2024, 69M indexed chunks) that materializes an explicit, verified evidence layer between retrieval and generation. Retrieved passages are decomposed into atomic claims that become nodes of an evidence graph, linked to source chunks, papers, and resolved citation edges; each claim is verified by a two-tier NLI/LLM judge before answer generation, and every generated sentence carries inline citations grounded in specific document sections. The evidence graph is exposed to users as an interactive visualization supporting claim-level provenance drill-down. In our evaluation, removing the evidence graph causes the largest quality drop among all ablations, while additional agentic steps, such as query decomposition and citation expansion, do not improve aggregate answer quality, suggesting that an explicit, verified evidence representation, rather than more agents, drives reliability. EviGraph-R is available as a public demo, an open-source package, and a released index and diagnostic benchmark.

---

## 3. Contribution claims

1. **System:** the first open-domain scientific QA system, to our knowledge, to use a _verified claim-level evidence graph_ as an explicit intermediate representation between retrieval and generation at full-text corpus scale (claims as nodes, unlike paper-node [SciRAG] or chunk-node [CG-RAG] graphs; with integrated two-tier NLI/LLM verification).
2. **Interface:** an interactive evidence-graph UI (verification badges, citation-hop provenance, per-sentence attribution, source drill-down) that turns answer verification from reading into inspection.
3. **Finding:** ablation evidence that the graph layer (not additional agentic steps) is the load-bearing component for answer quality (conditional on our corpus and metrics).
4. **Artifacts:** public demo, PyPI package, released 2.28M-paper index configuration, and a 268-question category-structured diagnostic benchmark released alongside the system.

---

## 4. Section skeleton (EACL demo, 6 pp; trims to ECIR 4 pp by compressing §2–3 and reducing §5 to the table)

1. **Introduction & Motivation** (0.75 p) — the trust gap in cited answers; the evidence-layer thesis.
2. **System Overview** (1.5 p) — pipeline figure; Decomposer with IMRaD section routing; hybrid BGE-M3 dense+sparse retrieval, RRF, cross-encoder reranking; Graph Builder (atomic claim extraction, citation expansion via resolved unarXive cite spans); two-tier Judge (DeBERTa-v3 NLI + LLM); Answer Generator with per-sentence inline citations.
3. **Interface Walkthrough** (1 p) — screenshots: graph build-up during streaming, verification badges, citation side drawers, claim-to-chunk drill-down.
4. **Comparison with Existing Systems** (0.75 p) — table below plus one-line differentiators.
5. **Evaluation Summary** (0.75 p) — corrected retrieval table (Hit@1/5/10, MRR@10, Recall@10; EviGraph-R vs. SQuAI); ablation headline with per-metric breakdown (shared three metrics vs. custom two reported separately); honest null-result framing for decomposition and citation expansion, positioning expansion as a traceability feature. Comfortably clears the demo-track "some form of evaluation" requirement.
6. **Availability, Licensing, Limitations** (0.25 p+) — demo URL, PyPI, repository, license; limitations: synthetic benchmark, LLM-as-judge scoring, single corpus. _Plus the mandatory ≤2.5-minute screencast (storyboard = task sheet T10: question → decomposition → graph build-up → verification badges → per-sentence citations → claim-node drill-down)._

---

## 5. Delta to SQuAI (the same-lab question, answered head-on)

| Dimension                 | SQuAI (CIKM 2025)                                    | EviGraph-R                                                                           |
| ------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Evidence representation   | Implicit, as claim–citation pairs inside LLM context | **Explicit materialized graph**: claims as nodes, edges to chunks, papers, citations |
| Claim verification        | None at generation time                              | **Two-tier per-claim verification** (DeBERTa-v3 NLI + LLM judge)                     |
| Citation handling         | Citations attached to claims                         | **Citation-hop expansion** into cited papers' chunks via resolved unarXive spans     |
| Retrieval structure       | Abstract _and_ full-text configurations              | Full-text **chunk/section-level** with IMRaD **section routing** in decomposition    |
| Attribution granularity   | Claim-level with evidence sentences                  | **Per-sentence inline citations** grounded in sections                               |
| User-facing transparency  | Intermediate reasoning steps as text                 | **Interactive graph** with provenance drill-down                                     |
| Headline empirical result | Multi-agent pipeline beats RAG baseline              | **Evidence layer is the load-bearing component; extra agents are not**               |

One-line differentiators for the rest: 
- **OpenScholar / Ai2 Scholar QA** — larger corpora, sentence-level citations, but evidence stays implicit (self-feedback / quote–outline), no claim graph, no verification. 
- **SciRAG (EACL 2026)** — citation graph with _paper_ nodes over abstracts/snippets, ≤1-hop, evaluation-time hallucination check only. 
- **CG-RAG** — _chunk_-node citation graphs on small static datasets, no verification, no open-domain scale. 
- **PaperQA2** — search-time agent, no fixed index, no explicit evidence structure.

Framing sentence for the paper: _"EviGraph-R extends our previously published SQuAI system with an explicit, verified evidence layer and an interactive inspection interface; this demonstration centers on that layer."_

---

## 6. Remaining work, risks, and timeline

**Remaining work:**

- Corrected retrieval metrics and per-metric ablation breakdown — T1/T2; these feed §5 of the paper directly.
- Stable public deployment with uptime plan and cached showcase queries — T6 ; reviewers will click the link in December–February.
- Screencast — T10.
- Comparison section and related-work positioning — Jingbo, from the prior-art analysis.
- Writing: 6 pp draft (task sheet Phase 2).

**Venue rule:** EACL 2027 demo CFP (expected ~late November deadline, not yet published) out by **October 20** → submit EACL; otherwise → **ECIR 2027 demo, November 2** (confirmed).

**Benchmark note:** the 268-question benchmark ships as a released artifact with the demo. 

**Checkpoint:** corrected numbers July 17 → evaluation section drafted and full draft v1 by the end of July → submission November.