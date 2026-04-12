# Evidence Graph Quality Report

## Executive Summary

The current evidence graph is structurally valid and preserves provenance well, but its judge-readiness remains weak to moderate. It captures the right topic and retains traceability from papers to chunks to extracted claims, yet it is still underdeveloped as a reasoning artifact. The central quality problem is not total irrelevance; it is semantic underdevelopment. The graph behaves more like a provenance-preserving retrieval graph than a reasoning-ready evidence graph, so the Judge node inherits too much of the real synthesis and verification burden.

## Evaluation Context

This report evaluates the evidence graph immediately before the Judge node for the query:

`How are Gaussian graphical models used to estimate conditional dependence structure across multiple groups?`

The goal is to assess whether the graph is useful for grounded judging, not merely whether graph construction completed successfully. A high-quality pre-Judge graph should do more than store retrieved material. It should provide a selective, relevant, and semantically organized evidence base that helps the Judge verify the answer with less ambiguity and less avoidable noise.

The current pipeline behavior underlying this assessment is straightforward:

- the builder creates `paper`, `chunk`, `claim`, and `concept` nodes
- the current structural relations are `belongs_to`, `cites`, and `extracted_from`
- semantic support relations such as `supports`, `refines`, or `answers_subquery` are not yet added
- the Judge currently receives a graph that is rich in provenance but relatively poor in reasoning structure

## Observed Quality

| Dimension | Score | Assessment |
| --- | ---: | --- |
| Schema correctness | 8/10 | The graph uses a coherent node and edge schema and preserves traceability cleanly. |
| Relevance | 7/10 | The anchor paper is highly relevant, but a few chunks are weakly aligned to the actual query. |
| Diversity | 3/10 | Most useful evidence comes from one paper, so the graph looks broader than it really is. |
| Connectivity | 4/10 | Structural links are present, but meaningful claim-to-claim reasoning links are absent. |
| Claim usefulness | 6/10 | Several claims are valid and grounded, but too many are generic rather than query-specific. |
| Semantic richness | 3/10 | The graph has almost no explicit semantic structure beyond extraction and citation bookkeeping. |
| Sub-query coverage | 5/10 | Definition and basic estimation are covered better than the multi-group extension. |
| Judge readiness | 4/10 | The graph can support a rough answer, but it does not yet reduce Judge burden enough. |

Overall, the graph is a good retrieval and provenance artifact, but not yet a strong evidence reasoning graph.

## What Works Well

### Correct basic graph schema

The graph already contains the right first-order building blocks for evidence tracking. Papers, chunks, claims, and concepts are represented distinctly, and the core structural edges are consistent enough to reconstruct where each extracted item came from.

### Strong anchor paper relevance

The main paper, `1608.08659`, is well aligned with the query. It covers Gaussian graphical models, conditional dependence via the precision matrix, and dependent multi-group structure through systemic and category-specific layers. This gives the graph a solid topical center.

### Provenance is preserved end to end

Claims and concepts are attached back to chunks, and chunks belong to papers. That is valuable because it keeps the evidence auditable and makes later evaluation easier. The graph succeeds at answering the question, “where did this claim come from?”

### Core GGM facts are present

The graph captures the essential ideas needed to start answering the query:

- Gaussian graphical models encode conditional dependence
- zeros or non-zero structure in the precision matrix express conditional independence structure
- sparse estimation is central to graph recovery
- multi-group estimation can involve shared and group-specific components

These facts make the graph usable as a first-pass evidence object.

## Main Quality Issues

### 1. The graph relies too heavily on one source paper

Most of the answer-bearing evidence comes from a single anchor paper. That paper is relevant, but one-paper dominance reduces robustness and makes the graph fragile. If the answer depends mostly on one framing, the Judge has little opportunity to cross-check the same point across independent sources.

This is especially important for the current query because it spans three distinct tasks: defining GGMs, explaining conditional dependence estimation, and explaining how this changes in multi-group settings. Those parts should ideally be supported by more than one paper.

### 2. Citation breadth is not the same as evidence breadth

The graph contains many cited-paper nodes, but most of them are only bibliography placeholders. They are connected through `cites` edges without retrieved chunk text, extracted claims, or any direct reasoning role. This inflates the graph visually and numerically without improving its evidentiary strength.

As a result, the graph looks larger and richer than it really is. The effective evidence base remains narrow even though the node count suggests breadth.

### 3. The graph lacks semantic reasoning edges

At present, the graph is dominated by structural edges:

- `belongs_to`
- `cites`
- `extracted_from`

Those are useful for provenance, but they do not help the Judge understand which claims support each other, which claims answer which sub-query, or which evidence is central versus peripheral. Without semantic edges, the graph does not yet encode why its contents answer the question.

### 4. Sub-query coverage is uneven

The decomposition is reasonable in form, but the resulting evidence is unbalanced. The graph covers the definitional part of the query well and the generic estimation part moderately well. The weakest portion is the actual multi-group extension, which is the most distinctive part of the question.

This happens because the graph is strongest on introductory and abstract material. It is weaker on method-heavy evidence that would directly explain how multi-group conditional dependence is estimated.

### 5. Too many extracted claims are generic rather than query-specific

Several extracted claims are valid but low-value for this query. Claims such as “Gaussian graphical models represent conditional dependence among random variables” are useful background, but they do not do much to answer the more specific part of the question about multiple groups.

The current extraction layer preserves correctness better than selectivity. That is a reasonable starting point, but it means the graph stores a lot of background that the Judge must mentally filter out.

### 6. Weak chunk quality control allows low-value evidence into the graph

Some retrieved chunks have very low retrieval scores, yet they still enter graph construction and claim extraction. This introduces noise early and can create grounded but low-utility claims. The graph therefore includes evidence that is technically connected to the topic while still being operationally weak for answer support.

Low-score material is especially costly because every extra chunk can lead to more claims, more verdicts, and more downstream reasoning work.

### 7. The Judge is asked to do too much of the actual reasoning

Because the graph is not selective or semantically organized enough, the Judge is forced to handle tasks that ideally should already be partly solved upstream:

- deciding which claims matter
- distinguishing background from answer-bearing evidence
- inferring relations between claims
- compensating for missing cross-source support

The Judge should verify and arbitrate, not perform most of the graph’s missing organization work.

### 8. Metric integrity issues can misstate graph quality

Some reporting fields can make the system appear cleaner or more coherent than it is. In particular, structural counts and verdict counts do not always align cleanly, concept nodes may be treated as if they were factual claims, and fallback-style outcomes can be mistaken for successful evidence-backed answers if metrics are not defined carefully.

These issues do not change the graph itself, but they weaken the reliability of quality evaluation and can hide where the pipeline is actually underperforming.

## Fix Suggestions

### Retrieval fixes

#### Increase emphasis on the multi-group sub-query

The most distinctive part of the user query is the multi-group estimation setting. Retrieval should spend more budget on that part and less on broad definitional prompts.

Why it helps: this increases the chance of retrieving evidence that directly answers the hard part of the question.

Expected impact: moderate to high improvement in sub-query coverage and judge readiness.

#### Route multi-group estimation toward methods-heavy sections

Queries about how multiple groups are modeled should prefer `Methods` or equivalent model-description sections rather than relying mostly on `Introduction`, `Results`, or `Discussion`.

Why it helps: method sections are more likely to contain the actual modeling decisions, assumptions, and estimation procedure.

Expected impact: moderate improvement in claim usefulness and answer specificity.

#### Filter or downweight near-zero-score chunks

Introduce a retrieval threshold or adaptive pruning step before graph construction so very weak matches do not automatically become claims.

Why it helps: this reduces noise and keeps the graph focused on evidence with a reasonable chance of supporting the answer.

Expected impact: moderate improvement in relevance, connectivity quality, and Judge efficiency.

#### Expand key cited papers into actual chunk evidence

When an anchor chunk cites especially important foundational work, retrieve a small number of chunks from the most relevant cited papers instead of keeping them as citation-only nodes.

Why it helps: this converts bibliographic breadth into usable evidence breadth.

Expected impact: high improvement in diversity and cross-source support.

But think of stategy : QUALITY VS LATENCY.

### Graph-building fixes

#### Add semantic relations such as `supports`, `refines`, and `answers_subquery`

The graph should connect claims to each other and to the sub-query they satisfy, rather than only attaching them back to their source chunk.

Why it helps: semantic links reduce the amount of synthesis the Judge must infer from scratch.

Expected impact: high improvement in connectivity, semantic richness, and judge readiness.

claim_A --supports--> claim_B
claim_A --refines--> claim_B
claim_A --contrasts--> claim_B

think for these are well

Extract methodological claims separately

Problem

Definition claims and method claims are mixed.

Change

Classify claim types.

Example:

claim_type = {
  definition
  method
  result
  assumption
}

Implementation

During claim extraction prompt.

Goal

Judge can reason about:

definition vs method vs evidence

#### Deduplicate repeated claims and concepts

Repeated background claims and duplicated concepts should be merged or clustered so the graph reflects unique evidence units rather than extraction repetition.

Why it helps: deduplication reduces graph inflation and makes important evidence easier to see.

Expected impact: moderate improvement in graph clarity and evidence selectivity.

#### Tag claims by sub-query relevance

Each extracted claim should carry a lightweight mapping to the sub-query it most directly answers.

Why it helps: this makes coverage visible and gives downstream components a structured way to judge completeness.

Expected impact: moderate improvement in sub-query coverage and answer organization.

#### Rank claims by specificity and answer value

Promote claims that mention multi-group structure, shared versus group-specific components, estimation mechanisms, or modeling assumptions over generic definitional statements.

Why it helps: not all true claims are equally useful. Ranking improves the graph’s practical value without requiring perfect extraction.

Expected impact: high improvement in claim usefulness and Judge efficiency.

### Judge-input fixes

#### Stop verifying concept nodes as if they were factual claims

Concept nodes should usually serve as labels or anchors, not as ordinary support claims that require the same verification path as extracted propositions.

Why it helps: this prevents unnecessary verdict inflation and reduces confusion in quality metrics.

Expected impact: high improvement in metric integrity and cleaner Judge input.

#### Distinguish central evidence from background context

Before the Judge runs, the graph should separate answer-bearing claims from generic contextual material.

Why it helps: the Judge can spend its budget on the most useful propositions instead of rediscovering which evidence matters.

Expected impact: moderate to high improvement in judge readiness.

#### Pass a cleaner, more selective graph into the Judge node

The Judge should receive fewer but better connected claims, supported by stronger chunks and clearer sub-query alignment.

Why it helps: a smaller, sharper graph improves both verification reliability and interpretability.

Expected impact: high improvement in end-to-end answer quality.

### Metric fixes

#### Align judged-count metrics with verdict-count metrics

Counts should refer to the same underlying unit. If verdicts are per evaluated claim, then summary metrics should not mix them with edge-level counts in a misleading way.

Why it helps: quality evaluation becomes easier to trust and easier to debug.

Expected impact: high improvement in evaluation reliability.

## Stop judging concept nodes


Based on the pipeline design and what Agent 2 is supposed to produce:Here are all the edge labels the graph should have, organized by source:

---

### Structural edges (no model needed — built from chunk metadata)

| Label | Direction | Description |
|---|---|---|
| `CHUNK_OF` | ChunkNode → PaperNode | Every chunk belongs to its paper. Currently implemented as `belongs_to` — the name should be standardized to `CHUNK_OF` to match the design spec. |

---

### Citation edges (from SciCite — chunk → chunk, not paper → paper)

These three replace the current flat `cites` paper→paper edges. The source is the **citing chunk**, the target is a **chunk from the cited paper** (fetched via citation hop expansion). The edge carries the citation sentence as context for SciCite input.

| Label | When applied | Agent 4 phrasing |
|---|---|---|
| `METHOD` | The citing paper uses a technique, model, or algorithm from the cited work | "X uses the approach from Y" |
| `BACKGROUND` | The citation provides context, motivation, or prior work framing | Lower weight, contextual framing |
| `RESULT_COMPARISON` | The citing paper compares its results against the cited work | "X outperforms Y on Z" |

Currently the graph has none of these — only `cites` at the paper level with no SciCite classification.

---

### Semantic edges (from NLI cross-check — claim → claim, cross-paper)

| Label | Direction | Description |
|---|---|---|
| `SUPPORTS` | ClaimNode → ClaimNode | NLI entailment between claims from different papers. Added as a graph edge. |
| `CONTRADICTS` | — | **Not a graph edge** — stored as a node attribute on the ClaimNode to keep the graph DAG-projectable and avoid cycles. |

Currently the graph has neither.

---

### Claim/concept extraction edges (from Claimify LLM call)

| Label | Direction | Description |
|---|---|---|
| `EXTRACTED_FROM` | ClaimNode → ChunkNode | Links each atomic claim back to its source chunk. ✅ Already present. |
| `EXTRACTED_FROM` | ConceptNode → ChunkNode | Same edge type for concept nodes. ✅ Already present. |

---

### What's present vs. missing in the current graph

| Label | Status |
|---|---|
| `CHUNK_OF` / `belongs_to` | ✅ present (wrong name) |
| `EXTRACTED_FROM` | ✅ present |
| `METHOD` | ❌ missing — SciCite not run |
| `BACKGROUND` | ❌ missing — SciCite not run |
| `RESULT_COMPARISON` | ❌ missing — SciCite not run |
| `SUPPORTS` | ❌ missing — NLI cross-check not run |
| `CONTRADICTS` (node attr) | ❌ missing |

The current `cites` (paper→paper, score=1) edges need to be replaced entirely — they carry no semantic information and point to the wrong node type.

