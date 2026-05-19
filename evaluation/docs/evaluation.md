# EviGraph-R Evaluation Specification

## 0. Introduction

**Purpose:** Define the evaluation methodology for EviGraph-R before any data construction begins.

**Scope:** Evaluation phase for the EviGraph-R demo paper using **700–800 synthetic questions** generated from the local Qdrant index (**unarxive_chunks**, **69M chunks**, **2.2M papers**) with DeepEval's Synthesizer.

## 1. The System Being Evaluated

EviGraph-R is a multi-agent system for scientific question answering over a corpus of approximately 2.3M arXiv papers (unarXive), indexed as 69M chunks in a Qdrant vector database. A query flows through five agents in sequence:

```
Query
  │
  ▼
[Agent 1] Decomposer
          → decomposes the query into a list of focused sub-queries,
            each tagged with target IMRaD section(s) and a retrieval budget weight
  │
  ▼
[Agent 2] Hybrid Retriever
          → retrieves top-k chunks per sub-query using BGE-M3 dense vectors
            + BM25 sparse retrieval, followed by cross-encoder reranking;
            applies IMRaD section-aware score boosting
  │
  ▼
[Agent 3] Evidence Graph Builder
          ├─ builds a structural graph: PAPER nodes, CHUNK nodes, citation edges
          ├─ enriches the graph with LLM-extracted CLAIM and CONCEPT nodes
          └─ citation expansion: for each chunk containing a cite_span pointing
             to a paper present in the index, fetches that paper's chunks and
             adds them as EVIDENCE nodes with typed citation edges
             (METHOD / BACKGROUND / RESULT_COMPARISON via SciCite)
  │
  ▼
[Agent 4] Judge — three-route adaptive verifier
          ├─ atomic, single-source claims  → fast NLI batch verification
          ├─ NLI verdict is "neutral"      → escalate to LLM judge
          └─ cross-paper or contradicted  → direct LLM judge
          Each claim receives a verdict: Supported / Contradicted / Inconclusive
  │
  ▼
[Agent 5] Answer Generator
          → synthesizes a multi-sentence answer from verified, supported claims only;
            each sentence carries per-chunk citations, a verdict tag, and a conflict flag;
            if no supported claims exist, returns a fallback abstention response
```

EviGraph-R's two architectural contributions beyond its predecessor SQuAI are the **evidence graph with citation expansion** (Agent 3) and the **three-route adaptive Judge** (Agent 4). The benchmark must demonstrate that each contribution improves measurable answer quality.

## 2. Why a Custom Benchmark?

Existing scientific QA benchmarks each fail to match our evaluation requirements:

| Benchmark | What it tests | Why it does not fit EviGraph-R |
| --- | --- | --- |
| QASPER | QA with evidence spans over NLP papers | Questions are scoped to a single document |
| SciFact | Claim verification against abstracts | No retrieval pipeline; no decomposition; no citation expansion |
| HotpotQA / 2WikiMultiHopQA | QA requiring multiple documents | Wikipedia domain; not scientific; citation links are not real |
| BioASQ | Biomedical factoid QA | Single biomedical domain; no citation graph structure |

The shared limitation is that these benchmarks either restrict questions to a single document or, where multiple documents are involved, the questions come from a different corpus than the one the system retrieves from. A system cannot be fairly evaluated on questions whose gold evidence does not exist in its index. EviGraph-R's core capability is synthesizing answers from multiple papers using real citation structure from our own index. We therefore construct a benchmark directly from our collection, guaranteeing that every question is answerable from the corpus the system operates on.

## 3. Evaluation Philosophy: The DeepEval Framework

### 3.1 Framework

We use [**DeepEval**](https://deepeval.com/docs/introduction) as our evaluation framework, following the same approach used in SQuAI. DeepEval provides: production-ready implementations of standard RAG metrics, a Synthesizer for automated question generation from any text corpus, and G-Eval for defining custom LLM-as-judge criteria. All computation runs locally using our ScaDS.AI group API as the judge model.

### 3.2 Evaluation Metrics

We use five metrics. The first three match SQuAI's evaluation exactly, enabling direct numeric comparison. The fourth adds claim-level completeness coverage. The fifth evaluates attribution quality — whether each cited sentence in the answer is actually supported by the chunk it cites.

| # | Metric | DeepEval class | What it measures | Test case fields used |
| --- | --- | --- | --- | --- |
| 1 | **Answer Relevance** | `AnswerRelevancyMetric` | Does the answer directly address the question? | `input`, `actual_output` |
| 2 | **Contextual Relevance** | `ContextualRelevancy Metric` | Is the retrieved context relevant to the question? | `input`, `retrieval_context` |
| 3 | **Faithfulness** | `FaithfulnessMetric` | Is the answer grounded in the retrieved context? Detects hallucination at the answer level. | `actual_output`, `retrieval_context` |
| 4 | **Claim Coverage** | `GEval` (custom) | Does the answer cover the key claims in the gold expected answer? | `actual_output`, `expected_output` |
| 5 | **Attribution Faithfulness** | `GEval` (custom) | For each sentence in the answer that carries a citation, does the cited chunk actually support that specific sentence? | `actual_output`, `retrieval_context` |

**Why metrics 4 and 5 are necessary alongside metrics 1–3.**
Metrics 1–3 are answer-level and retrieval-level checks. They do not catch two failure modes that are central to EviGraph-R's design:

- Metric 4 (Claim Coverage) catches *incompleteness*: the system produced a relevant, faithful answer but omitted key information present in the gold answer. Metrics 1–3 cannot detect this.
- Metric 5 (Attribution Faithfulness) catches *misattribution*: the system cited the wrong chunk for a sentence, or made a factual claim with no citation at all. Metric 3 (Faithfulness) checks whether the answer as a whole is grounded in the retrieved context — it does not verify that individual sentence-to-citation mappings are correct. A system could score perfectly on Faithfulness while systematically attaching citations to the wrong sentences, which is exactly the failure mode EviGraph-R's Judge is designed to prevent.

**Aggregate score.** We compute the mean of all five metric scores per system configuration as a single comparison row. Per-category means are also reported.

**Claim Coverage definition.** Implemented as a G-Eval with the following evaluation steps:

```
1. Extract the key factual claims from the expected_output.
2. For each claim, check whether the actual_output contains that
   information, even if worded differently or sourced from different evidence.
3. Score 1.0 if all key claims are present, 0.0 if none are covered.
4. Do not penalize the actual_output for containing additional correct
   information beyond what the expected_output states.
```

**Attribution Faithfulness definition.** Implemented as a G-Eval with the following evaluation steps:

```
1. Extract each sentence from the actual_output that carries a citation
   (identified by chunk_id, doc_id, or inline reference marker).
2. For each cited sentence, locate the corresponding chunk text in the
   retrieval_context using the cited identifier.
3. Check whether that specific chunk provides direct factual support
   for the claim made in that sentence.
4. Penalize cases where: (a) the cited chunk does not support the
   sentence it is attached to, or (b) a sentence making a specific
   factual claim carries no citation at all.
5. Score 1.0 if all cited sentences are correctly supported by their
   cited chunks, 0.0 if none are.
```

This metric requires no gold chunk set. It operates entirely on the system's own structured output — the `chunk_id` per sentence — cross-referenced against the retrieved context already present in the test case.

**Latency.** We additionally record average wall-clock response time per query for each system configuration. Reported separately, not included in the aggregate score.

### 3.3 Immunity to the corpus incompleteness problem

A critical design property of our evaluation is that the primary metrics — Answer Relevance, Contextual Relevance, and Faithfulness — do not compare the system's retrieved chunks against a fixed gold chunk set. They operate on what the system actually returns:

- **Answer Relevance**: compares the system's answer against the question
- **Contextual Relevance**: compares the system's retrieved context against the question
- **Faithfulness**: compares the system's answer against the system's retrieved context

This means the evaluation is immune to the corpus incompleteness problem. When the index contains hundreds of valid chunks on the same topic, the system is not penalized for retrieving different-but-valid evidence.

### 3.4 Dataset Construction Strategy

All questions are generated from our index. No external documents are introduced.

**Domain stratification.** We sample proportionally across four discipline groups derived from arXiv category prefixes:

| Group | Categories | Target share |
| --- | --- | --- |
| Computer Science | `cs.*` | 25% |
| Physics | `hep-*`, `astro-*`, `quant-ph` | 25% |
| Mathematics | `math.*` | 25% |
| Statistics / Other | `stat.*`, `econ.*`, and remaining | 25% |

**Context grouping.** The Synthesizer's `generate_goldens_from_contexts()` function takes a list of context groups. Each context group is a Python list of text strings — one string per chunk. The synthesizer generates one question per group, naturally producing a question that requires all chunks in the group to answer. We construct these groups differently for each question category (see Section 4). Groups are built entirely from payload fields already present in the Qdrant index: `embed_text`, `paper_id_arxiv`, `section_title`, `categories`, and `spans.cite_spans`.

**Question generation.** All questions are generated with `include_expected_output=True` and the `Reasoning` evolution applied, which produces synthesis-oriented questions that ask the model to connect, compare, or explain — rather than retrieve a single fact.

## 4. Question Categories

Four categories cover the full capability profile of EviGraph-R. Each category targets one or more specific agents and produces questions of different structural complexity.

### Category 1 : Single-paper synthesis (n ≈ 200)

**What it tests.** Basic retrieval and answer generation from a single paper. This is the capability shared with SQuAI and standard RAG, and serves as the regression baseline: EviGraph-R must not perform worse than these baselines on simple queries.

**Context construction.** Three subsection chunks from the same paper (`paper_id_arxiv`), sampled from different positions in the document (`chunk_index` spread), are grouped into one context. The synthesizer produces a broad thematic question about the paper's topic.

**Agents primarily tested.** Agent 2 (retrieval), Agent 5 (answer generation).

**Expected ablation signal.** No significant drop on any ablation. If a variant drops on Category 1, it has regressed on basic capability.

### Category 2 : Cross-section synthesis (n ≈ 150)

**What it tests.** The Decomposer's ability to split a compound question and route each sub-query to the correct IMRaD section. Questions in this category require connecting information from two structurally distinct parts of the same paper, such as a method described in Methods and an outcome reported in Results.

**Context construction.** Two subsection chunks from the same paper, drawn from two different IMRaD section labels. Valid section pairs: (Introduction, Results), (Methods, Results), (Methods, Discussion), (Introduction, Conclusion). Chunks with noisy or missing section labels (approximately 23% of all chunks, as noted in our indexing report) are excluded.

**Agents primarily tested.** Agent 1 (decomposition), Agent 2 (section-aware retrieval).

**Expected ablation signal.** The `no-decomposer` variant should show the clearest drop on this category, since the decomposer is the only component that routes sub-queries to specific IMRaD sections.

### Category 3 : Citation expansion (n ≈ 200)

**What it tests.** The Evidence Graph Builder's citation expansion mechanism end-to-end. This category justifies EviGraph-R's existence relative to flat-retrieval baselines. Questions require understanding the relationship between a citing paper and a paper it cites — what the cited work established and how the citing paper uses or extends it. This is only fully answerable when evidence from both papers is retrieved and connected.

**Context construction.** Chunk A is sampled from paper A, filtered to chunks where `spans.cite_spans` contains at least one resolved `paper_id_arxiv` pointing to a paper B that is present in the index. One chunk from paper B (preferably from a Methods or Results section) is fetched as chunk B. The context group is `[chunk_A.embed_text, chunk_B.embed_text]`.

**Similarity constraint.** We require the cosine similarity between chunk A and chunk B to fall between 0.40 and 0.80. Below 0.40 the chunks share too little to generate a meaningful question. Above 0.80 the chunks are near-redundant and one alone would suffice.

**Coverage note.** Based on our index report, 47% of cite_spans have a resolved public ID, of which 9.4% resolve to an arXiv ID present in our collection. This gives approximately 11.7M cite_span links to draw from — more than sufficient for sampling 200 pairs.

**Agents primarily tested.** Agent 3 (citation expansion), Agent 4 (cross-paper claim verification).

**Expected ablation signal.** The `no-citation-expansion` variant should show the largest and most consistent drop on this category, particularly on Attribution Faithfulness and Claim Coverage.

### Category 4 : Thematic multi-paper synthesis (n ≈ 200)

**What it tests.** The system's ability to synthesize a coherent answer from evidence distributed across multiple papers on the same scientific theme, without an explicit citation link between them. This tests the full pipeline end-to-end and most closely resembles how a researcher would use the system: asking a broad question about a topic and expecting the system to draw on the literature.

**Context construction.** Three chunks are sampled from three different papers within the same arXiv category group (`categories` field). Chunks are selected by running a Qdrant vector search from a seed chunk and retaining the top results from different `paper_id_arxiv` values, with cosine similarity between 0.50 and 0.80. All three chunks are placed in one context group: `[chunk_A.embed_text, chunk_B.embed_text, chunk_C.embed_text]`.

**Agents primarily tested.** All five agents. This is the most complete test of the full pipeline.

**Expected ablation signal.** Both `no-citation-expansion` and `no-decomposer` should show drops here. Standard RAG and SQuAI should show the largest gap relative to full EviGraph-R.

### Dataset size summary

| Category | Description | n | Share |
| --- | --- | --- | --- |
| Single-paper synthesis | 3 chunks, 1 paper | ~200 | 27% |
| Cross-section synthesis | 2 chunks, 1 paper, 2 IMRaD sections | ~150 | 20% |
| Citation expansion | 2 chunks, 2 papers, cite_span link | ~200 | 27% |
| Thematic multi-paper | 3 chunks, 3 papers, same category | ~200 | 27% |
| **Total** |  | **~750** | **100%** |

Each category is stratified so that CS, Physics, Mathematics, and Stats/Other each contribute approximately 25% of that category's questions.

## 5. Agent Coverage Matrix

The table below shows which agents are primarily tested (★) versus secondarily exercised (●) in each category.

| Category | Decomposer (A1) | Retriever (A2) | Graph Builder (A3) | Judge (A4) | Answer Gen (A5) |
| --- | --- | --- | --- | --- | --- |
| Single-paper | ○ | ● | ○ | ○ | ★ |
| Cross-section | ★ | ★ | ○ | ○ | ● |
| Citation expansion | ● | ● | ★ | ★ | ● |
| Multi-paper | ● | ● | ★ | ★ | ★ |

★ primary capability tested · ● secondary capability exercised · ○ not the focus

## 6. Baselines

### Standard RAG

A single-stage retrieval pipeline: the raw query is embedded with BGE-M3 and used to retrieve top-k chunks from Qdrant by dense vector similarity. The retrieved chunks are concatenated and passed directly to an LLM to generate an answer. No decomposition, no reranking, no evidence graph, no claim verification.

**Why this baseline.** It represents the simplest possible approach that a practitioner would implement first. Comparing against it demonstrates the aggregate value of the full EviGraph-R pipeline.

### SQuAI

Our predecessor system, developed and evaluated in previous work. SQuAI uses hybrid retrieval (dense + sparse) and structured answer generation but does not build an evidence graph, does not perform citation expansion, and does not have an adaptive claim verification judge. Its evaluation on the unarXive dataset provides directly comparable prior numbers.

**Why this baseline.** It isolates the contribution of EviGraph-R's two new components (Agent 3 and Agent 4) from the shared infrastructure. Any performance gain over SQuAI is attributable specifically to the evidence graph and the Judge.

## 7. Ablation Study

The ablation study disables one component at a time and measures the delta against the full system. All ablation variants run on the full 750-question dataset. Results are reported per category and in aggregate.

### Group 1  Agent 1: Query Decomposer

| Variant | What changes | Primary metrics | Rationale |
| --- | --- | --- | --- |
| **A1.1  No decomposition** | Raw query sent directly to retriever; no sub-queries generated | Contextual Relevance, Faithfulness (Cat. 2) | Decomposition routes sub-queries to correct IMRaD sections; removing it collapses section-aware retrieval |
| **A1.2 No budget weights** | All sub-queries receive equal retrieval budget (weight = 1/n) | Contextual Relevance, Claim Coverage | Budget weights prioritise sub-queries that need more evidence; equal weights flatten retrieval focus |

### Group 2 Agent 2: Hybrid Retriever

| Variant | What changes | Primary metrics | Rationale |
| --- | --- | --- | --- |
| **R1 Dense only** | BM25 sparse path removed | Contextual Relevance, Faithfulness | BM25 captures exact scientific terminology; removing it reduces recall for keyword-heavy queries |
| **R2 Sparse only** | Dense embedding path removed | Contextual Relevance, Claim Coverage | Dense retrieval handles semantic similarity; removing it reduces recall for paraphrased or conceptually related evidence |
| **R3 No section-aware boosting** | IMRaD section score boost disabled (γ = 0) | Contextual Relevance, Faithfulness (Cat. 2) | Section boosting de-ranks noise from abstracts and reference lists; removing it degrades precision |
| **R4 Full hybrid + section boost** | Complete retriever | All | Reference point for R1–R3 |

### Group 3 Agent 3: Evidence Graph Builder

| Variant | What changes | Primary metrics | Rationale |
| --- | --- | --- | --- |
| **G1 Flat chunks** | Evidence graph replaced by a flat ordered list of retrieved chunks; no claim extraction, no citation edges | Faithfulness, Claim Coverage (Cat. 3, 4) | The graph's relational structure enables cross-paper claim linking; flat context loses this |
| **G2 No citation expansion** | `hop_enabled = false`; only seed chunks used | Contextual Relevance, Claim Coverage, Attribution Faithfulness (Cat. 3) | Citation expansion fetches evidence from cited papers; disabling it prevents cross-paper reasoning and misaligns citations |
| **G3 Full graph** | Complete Evidence Graph Builder | All | Reference point for G1–G2 |

### Group 4 Agent 4: Reasoning Judge

| Variant | What changes | Primary metrics | Rationale |
| --- | --- | --- | --- |
| **J1 No judge** | All extracted claims pass directly to the Answer Generator without verification | Faithfulness, Attribution Faithfulness, Answer Relevance | The Judge filters unsupported claims; removing it lets hallucinated content and misattributed citations pass into the answer |
| **J2 NLI only** | LLM judge disabled; NLI used for all claims including cross-paper and complex ones | Faithfulness, Attribution Faithfulness, Claim Coverage (Cat. 3) | NLI cannot resolve claims requiring reasoning across documents; complex cross-paper citations will be incorrectly verified |
| **J3 LLM only** | NLI routing disabled; LLM judge used for all claims including simple atomic ones | Attribution Faithfulness, Latency | Tests routing efficiency; LLM is slower but citation accuracy should remain stable |
| **J4 Full routing (NLI → LLM)** | Complete three-route Judge | All | Reference point for J1–J3 |

### Group 5 Answer Generator

| Variant | What changes | Notes |
| --- | --- | --- |
| **AG1 From flat chunks only** | Answer generated from retrieved chunks without graph structure | Covered by G1 — no separate run needed. G1 results serve as this baseline. |

### Ablation results table format

| Configuration | Cat.1 | Cat.2 | Cat.3 | Cat.4 | Overall Avg. | Latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Standard RAG |  |  |  |  |  |  |
| SQuAI |  |  |  |  |  |  |
| A1.1 No decomposition |  |  |  |  |  |  |
| A1.2 No budget weights |  |  |  |  |  |  |
| R1 Dense only |  |  |  |  |  |  |
| R2 Sparse only |  |  |  |  |  |  |
| R3 No section boosting |  |  |  |  |  |  |
| G1 Flat chunks |  |  |  |  |  |  |
| G2 No citation expansion |  |  |  |  |  |  |
| J1 No judge |  |  |  |  |  |  |
| J2 NLI only |  |  |  |  |  |  |
| J3 LLM only |  |  |  |  |  |  |
| **EviGraph-R (full)** |  |  |  |  |  |  |

Each cell reports the mean of all five evaluation metrics for that category. The Overall Avg. column is the mean across all four categories. Latency is wall-clock seconds per query, averaged over the category.

## 8. What This Benchmark Cannot Tell Us

We state these limitations explicitly to anticipate reviewer objections. They are appropriate for a demo paper scope and would be addressed in a full dataset paper.

**Corpus boundary.** All 750 questions are drawn from the unarXive arXiv collection. We make no claim about how EviGraph-R would perform on PubMed, the ACL Anthology, S2ORC, or any other scientific corpus. The system's architecture is corpus-agnostic, but evaluation numbers are not.

**Question naturalness.** Questions are generated by an LLM synthesizer from grouped chunks, not typed by real researchers. The Reasoning evolution produces broad, open-ended questions that resemble realistic research queries but the distribution may differ from actual user intent in ways we cannot fully control.

**Citation expansion quality.** For Category 3, we verify that a cite_span resolves to a paper present in the index and that the chunk pair falls within the cosine similarity window. We do not manually verify that the cited paper's content is actually necessary to answer the generated question (i.e., that the citation is load-bearing rather than decorative). This is a known limitation accepted in exchange for the scale of the dataset.


## 9. Pilot Run (10-sample smoke test)

Before running the full 750-question benchmark, validate the entire pipeline end-to-end on a small sample. This catches data quality issues, import errors, and metric failures cheaply.

### Step 1 — Generate 10 goldens per category (40 total)

```bash
CAT1_TARGET=10 CAT2_TARGET=10 CAT3_TARGET=10 CAT4_TARGET=10 \
  sbatch evaluation/run_full_evaluation_capella.sh --generate-only
```

**Check:** Inspect `_data/benchmark/groups/cat{1..4}.jsonl` — verify chunk texts are non-empty, `paper_ids` are valid arXiv IDs, and Cat 3 metadata contains a `cosine_sim` in `[0.40, 0.80]`. Confirm `_data/benchmark/goldens.jsonl` has 40 lines with non-empty `input` and `expected_output`.

### Step 2 — Run ablation variants on the 40 goldens

Run the full system plus the two most diagnostic ablations (citation expansion and judge):

```bash
ABLATION_VARIANTS="full G2 J1" \
  sbatch evaluation/run_full_evaluation_capella.sh --ablation-only
```

**Check:** `_data/benchmark/results/` should contain `full.jsonl`, `G2.jsonl`, `J1.jsonl`. Verify `actual_output` is non-empty for most records and `errors` lists are empty or minimal.

### Step 3 — Run Standard RAG baseline on the same 40 goldens

```bash
sbatch evaluation/run_full_evaluation_capella.sh --baselines-only
```

**Check:** `_data/benchmark/results/standard_rag.jsonl` exists with 40 records.

### Step 4 — Inspect scores and ablation table

```bash
# Re-score and rebuild table from all result files in results/
uv run python -m evaluation.full_evaluation \
  --results_dir _data/benchmark/results \
  --output_dir  _data/benchmark/eval \
  --model "${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
```

**Check:** `_data/benchmark/eval/ablation_table.md` is populated. Confirm no metric column is all zeros (would indicate a scoring or import failure).

### Step 5 — Full run (all variants, all 750 questions)

Once the pilot passes all checks above:

```bash
sbatch evaluation/run_full_evaluation_capella.sh --everything
```

This runs the complete pipeline: dataset generation → all 11 ablation variants → Standard RAG baseline → scoring → final table.