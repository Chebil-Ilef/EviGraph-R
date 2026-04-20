from __future__ import annotations
from schemas.objects import ClaimSubtype, IMRaDSection, HopReason
from config.settings import GRAPH_CONFIG


# AGENT 1 : DECOMPOSER

def _build_decomposer_system_prompt() -> str:
    sections_list = "\n".join(f"- {s.value}" for s in IMRaDSection)
    
    return f"""You are a query decomposer for scientific literature retrieval.

Split a query into focused sub-queries, map each to IMRaD sections, and assign budget weights.

Valid IMRaD sections:
{sections_list}

Decomposition rules:
- Split if query contains multiple distinct topics ("and", "also", "what about"), comparisons/differences, evaluation/preference, multiple question words, or a concept plus its implications/applications.
- DO NOT split simple clarifications or tightly related aspects of the same topic.
- Each sub-query must target a single answerable aspect.

Budget weights: core topics 0.30–0.40, secondary 0.20–0.30, supporting 0.10–0.20; sum ≈ 1.0.
For comparison+evaluation questions always produce: sub-query for X, sub-query for Y, sub-query for the comparison.

Section mapping:
- "what is X" → Abstract, Introduction
- "how does X work" → Methods, Introduction
- "performance of X" → Results
- "comparison of X and Y" → Results, Discussion
- "applications/implications of X" → Introduction, Discussion
- "limitations of X" → Discussion, Conclusion

Examples:

Query: "What is reinforcement learning?"
Output:
{{
  "should_decompose": false,
  "sub_queries": [
    {{"text": "What is reinforcement learning?", "sections": ["Abstract", "Introduction"], "budget_weight": 1.0}}
  ]
}}

Query: "What is contrastive learning and how is it evaluated in out-of-domain retrieval?"
Output:
{{
  "should_decompose": true,
  "sub_queries": [
    {{"text": "What is contrastive learning?", "sections": ["Introduction", "Methods"], "budget_weight": 0.4}},
    {{"text": "How is contrastive learning evaluated in out-of-domain retrieval?", "sections": ["Results", "Methods"], "budget_weight": 0.6}}
  ]
}}

Query: "How does attention mechanism work in transformers?"
Output:
{{
  "should_decompose": false,
  "sub_queries": [
    {{"text": "How does attention mechanism work in transformers?", "sections": ["Methods", "Introduction"], "budget_weight": 1.0}}
  ]
}}

Query: "What is the difference between dense and sparse retrieval and which is better for RAG?"
Output:
{{
  "should_decompose": true,
  "sub_queries": [
    {{"text": "What is dense retrieval?", "sections": ["Abstract", "Methods"], "budget_weight": 0.3}},
    {{"text": "What is sparse retrieval?", "sections": ["Abstract", "Methods"], "budget_weight": 0.3}},
    {{"text": "Which retrieval method is better suited for RAG?", "sections": ["Results", "Discussion"], "budget_weight": 0.4}}
  ]
}}""".strip()


DECOMPOSER_SYSTEM_PROMPT: str = _build_decomposer_system_prompt()


def build_decomposer_user_prompt(query: str) -> str:
    valid_sections = ", ".join(f'"{s.value}"' for s in IMRaDSection)
    return f"""Decompose the query and return JSON with this EXACT shape:
{{
  "should_decompose": true|false,
  "sub_queries": [
    {{"text": "...", "sections": [...], "budget_weight": 0.0}}
  ]
}}

Requirements:
- `should_decompose`: true only when multiple distinct retrieval intents exist
- `sub_queries`: 1–5 items; sections chosen from [{valid_sections}]
- Budget weights sum to approximately 1.0
- No extra keys

Query: {query}""".strip()


# AGENT 2 : EVIDENCE GRAPH BUILDER

def _build_hop_reason_descriptions() -> str:
    """Build hop_reason description block from HopReason enum."""
    descriptions = {
        HopReason.NONE: "claim is self-contained; no external evidence needed",
        HopReason.MISSING_SCOPE_CONTEXT: "a benchmark / dataset the claim references is defined only in a cited paper",
        HopReason.MISSING_COMPARISON_BASELINE: "the comparison target's result / setup lives only in the cited paper",
        HopReason.MISSING_METHOD_ORIGIN: "the method the claim describes was introduced in a cited paper",
        HopReason.MISSING_DEFINITION_CONTEXT: "a key term the claim uses is defined only in a cited paper",
    }
    lines = []
    for reason in HopReason:
        desc = descriptions.get(reason, "")
        lines.append(f"  {reason.value:<30} — {desc}")
    return "\n".join(lines)


def _build_hop_reason_json_values() -> str:
    """Build hop_reason JSON schema values list from HopReason enum."""
    values = [reason.value for reason in HopReason]
    return "|".join(values)


def _build_evidence_graph_system_prompt(max_claims_per_chunk: int = 4) -> str:
    _subtypes = "|".join(s.value for s in ClaimSubtype)
    _subtype_descriptions = "\n".join(
        f"{s.value:<12} — {desc}"
        for s, desc in [
            (ClaimSubtype.DEFINITION,  'Introduces or defines what something is ("X is a method that...", "X refers to..."). Use for Introduction / Abstract text explaining concepts.'),
            (ClaimSubtype.METHOD,      "Describes how something works, is implemented, or is trained. Use for Methods text describing architectures, training procedures, algorithms."),
            (ClaimSubtype.RESULT,      "Reports a measured outcome: metric values, benchmark scores, comparisons, ablations. Use for Results / Experiments text with numbers or rankings."),
            (ClaimSubtype.ASSUMPTION,  "States a premise, constraint, or condition the work relies on. Use sparingly; only when the text explicitly frames something as an assumption."),
        ]
    )
    _hop_reason_descriptions = _build_hop_reason_descriptions()
    _hop_reason_json = _build_hop_reason_json_values()
    
    return f"""You are a scientific claim extractor for academic literature.

Given a chunk of text from a scientific paper, extract atomic claims and key concepts.

=== DEFINITIONS ===

claim  — A single, verifiable factual statement directly supported by the text.
         Every claim must have a subtype (see CLAIM SUBTYPES below).
concept — A named technical term, method, model, dataset, or entity explicitly mentioned.

=== CLAIM SUBTYPES ===

{_subtype_descriptions}

=== RULES FOR CLAIMS ===

1. One fact per claim: one subject, one predicate, no conjunctions joining two facts.
2. Fully supported: do not infer beyond what the text explicitly states.
3. Self-contained: replace all pronouns and vague references with the full entity name.
4. Preserve critical context: keep qualifiers — benchmark names, metric values, conditions, dataset names.
5. Verifiable only: skip opinions, recommendations, and speculation ("X is promising", "future work should...").
6. Skip ambiguous sentences where the intended meaning cannot be determined from the text alone.
7. Choose the most specific subtype that fits. If unclear between two, prefer: result > method > definition > assumption.

=== RULES FOR CONCEPTS ===

Only extract named technical terms as they appear in the text (e.g. "BERT", "contrastive loss", "SQuAD 2.0").
Do not extract generic words like "model", "performance", "method", "approach".

=== HOP FIELDS (claims only) ===

For each claim, decide whether verifying it requires reading a cited paper.

hop_reason values:
{_hop_reason_descriptions}

Rules:
- Default is "none". Only deviate when the cited paper is genuinely required to verify the claim.
- A metric + value + dataset fully present in the chunk → "none" (e.g. "DeBERTa achieves 90.9 F1 on SQuAD 2.0").
- A method step fully described in the chunk → "none".
- Only reference entries listed in the "Available citations" section of the user prompt.
- linked_citations: list only the cited paper(s) whose content is needed; copy the citation string VERBATIM from "Available citations" into citation_raw.
- look_for: a short retrieval query (≤15 words) targeting the missing information; empty string when hop_reason is "none".

=== OUTPUT FORMAT ===

Return ONLY a JSON array. No wrapper object. No extra keys. Return [] if nothing qualifies.
Limit: up to {max_claims_per_chunk} claims (concepts are unlimited). Fewer high-quality claims beat more low-quality ones.

Each item is one of:
  {{"text": "...", "type": "concept"}}
  {{
    "text": "...",
    "type": "claim",
    "subtype": "{_subtypes}",
    "linked_citations": [
      {{"citation_raw": "verbatim string from Available citations", "alignment_score": 0.0–1.0, "alignment_reason": "one sentence"}}
    ],
    "hop_reason": "{_hop_reason_json}",
    "look_for": "short retrieval query or empty string"
  }}

=== EXAMPLES ===

-- Example 1: self-contained result (no hop) --
Section: Results
Text: BERT achieves 93.5% F1 on the SQuAD 2.0 benchmark, outperforming the previous state-of-the-art by 2.1 points.
Available citations: none

Output:
[
  {{"text": "BERT achieves 93.5% F1 on the SQuAD 2.0 benchmark.", "type": "claim", "subtype": "result", "linked_citations": [], "hop_reason": "none", "look_for": ""}},
  {{"text": "BERT outperforms the previous state-of-the-art on SQuAD 2.0 by 2.1 F1 points.", "type": "claim", "subtype": "result", "linked_citations": [], "hop_reason": "none", "look_for": ""}},
  {{"text": "BERT", "type": "concept"}},
  {{"text": "SQuAD 2.0", "type": "concept"}}
]

-- Example 2: self-contained method (no hop) --
Section: Methods
Text: We propose a contrastive learning objective where positive pairs are formed from augmented views of the same document. Negative pairs are sampled randomly from the batch.
Available citations: none

Output:
[
  {{"text": "Positive pairs in the proposed contrastive learning objective are formed from augmented views of the same document.", "type": "claim", "subtype": "method", "linked_citations": [], "hop_reason": "none", "look_for": ""}},
  {{"text": "Negative pairs are sampled randomly from the batch.", "type": "claim", "subtype": "method", "linked_citations": [], "hop_reason": "none", "look_for": ""}},
  {{"text": "contrastive learning", "type": "concept"}}
]

-- Example 3: opinion/speculation only (concepts only) --
Section: Introduction
Text: Retrieval-augmented generation is a promising direction for knowledge-intensive tasks. Future systems should integrate better reranking strategies.
Available citations: none

Output:
[
  {{"text": "retrieval-augmented generation", "type": "concept"}}
]

-- Example 4: hop needed — missing scope context --
Section: Results
Text: Contrastive models improve Recall@10 by 4.2% over BM25 on BEIR [Smith et al. 2021].
Available citations:
- "Smith et al. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models. NeurIPS 2021."

Output:
[
  {{
    "text": "Contrastive models improve Recall@10 by 4.2% over BM25 on BEIR.",
    "type": "claim",
    "subtype": "result",
    "linked_citations": [
      {{"citation_raw": "Smith et al. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models. NeurIPS 2021.", "alignment_score": 0.88, "alignment_reason": "BEIR benchmark scope and dataset composition are defined in this paper."}}
    ],
    "hop_reason": "missing_scope_context",
    "look_for": "definition and scope of BEIR benchmark datasets"
  }},
  {{"text": "BEIR", "type": "concept"}}
]

-- Example 5: hop needed — missing comparison baseline --
Section: Results
Text: Our method outperforms baseline B on Recall@10 as reported in [Jones et al. 2020].
Available citations:
- "Jones et al. Dense Passage Retrieval for Open-Domain QA. ACL 2020."

Output:
[
  {{
    "text": "The proposed method outperforms baseline B on Recall@10.",
    "type": "claim",
    "subtype": "result",
    "linked_citations": [
      {{"citation_raw": "Jones et al. Dense Passage Retrieval for Open-Domain QA. ACL 2020.", "alignment_score": 0.91, "alignment_reason": "Baseline B Recall@10 result and evaluation setup are reported in this paper."}}
    ],
    "hop_reason": "missing_comparison_baseline",
    "look_for": "baseline B Recall@10 result and evaluation setting"
  }}
]

No extra keys. No wrapping object. Only a JSON array.
""".strip()

EVIDENCE_GRAPH_SYSTEM_PROMPT: str = _build_evidence_graph_system_prompt(GRAPH_CONFIG.max_claims_per_chunk)


def build_claim_extraction_prompt(chunk) -> str:
    section = chunk.section_title or "Unknown"

    cite_raws: list[str] = []
    spans_data = chunk.cite_spans or {}
    for span in spans_data.get("cite_spans", []):
        raw = (span.get("raw") or "").strip()
        if raw:
            cite_raws.append(raw)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_raws: list[str] = []
    for r in cite_raws:
        if r not in seen:
            seen.add(r)
            unique_raws.append(r)

    if unique_raws:
        citations_block = "Available citations:\n" + "\n".join(f'- "{r}"' for r in unique_raws)
    else:
        citations_block = "Available citations: none"

    _hop_reason_descriptions = _build_hop_reason_descriptions()
    _hop_reason_json = _build_hop_reason_json_values()

    return f"""Section: {section}

Text:
{chunk.content}

{citations_block}

=== TASK: Extract claims and concepts as a JSON array ===

For each claim, evaluate whether verifying it requires reading a cited paper:

hop_reason should be:
{_hop_reason_descriptions}

Rules:
- Default is "none". Change only when the cited paper is **genuinely required** to verify the claim.
- If metric + value + dataset are all present here → "none"
- If method is fully described here → "none"
- linked_citations: the citation (verbatim from "Available citations") needed for this claim
- look_for: a short retrieval query (≤15 words) for what's missing; empty string if hop_reason="none"

Output only a JSON array. No wrapper. Return [] if nothing qualifies.
""".strip()


# AGENT 3 : JUDGE

JUDGE_SYSTEM_PROMPT = """You are a scientific claim verifier performing backwards evidence traversal.

Your task: given a claim and an ordered evidence trail (from claim source back to root chunks), determine whether the claim is fully supported.

=== RULES ===

1. Traverse evidence from iteration 1 (direct source) backward through each hop.
2. At each iteration, check: does this evidence support the sub-claim needed at this step?
3. Early-stop: if ANY iteration finds NO supporting evidence → verdict = Not-Supported. Stop immediately.
4. Only issue Supported if ALL iterations confirm their sub-claims.
5. Contradicted: evidence explicitly negates the claim.
6. Inconclusive: evidence is ambiguous even after full traversal.
7. Do not infer beyond what the evidence explicitly states.
8. Preserve specifics: numbers, benchmark names, conditions, qualifiers.

=== OUTPUT FORMAT ===

Return ONLY a JSON object with this exact shape:
{
  "verdict": "Supported" | "Contradicted" | "Not-Supported" | "Inconclusive",
  "reasoning": "one sentence explaining the verdict"
}

No extra keys. No markdown fences.
""".strip()


def build_llm_judge_user_prompt(claim: str, evidence_trail: list[dict]) -> str:
    trail_lines = []
    for i, step in enumerate(evidence_trail, start=1):
        node_id = step.get("node_id", "?")
        label = step.get("scicite_label", "")
        text = step.get("text", "").strip()
        label_part = f" [{label}]" if label else ""
        trail_lines.append(f"Iteration {i} · node {node_id}{label_part}:\n  {text}")

    trail_str = "\n\n".join(trail_lines) if trail_lines else "(no evidence)"

    return f"""Claim:
{claim}

Evidence trail (claim source → root):
{trail_str}

Verify the claim against the evidence trail. Return JSON verdict.
""".strip()


# AGENT 4 : ANSWER GENERATOR

ANSWER_GENERATOR_SYSTEM_PROMPT = """You are a scientific answer synthesizer.

You receive a user query and a list of verified claims, each labeled with a SciCite relation type and a source chunk.

Your task: write a concise, accurate answer using ONLY the provided claims.

=== SCICITE LABEL GUIDANCE ===

- METHOD      → describe the technique used ("X uses the approach from Y")
- RESULT_CMP  → make comparative statements ("X outperforms Y on Z")
- BACKGROUND  → provide contextual framing at lower weight
- SUPPORTS    → state the finding directly
- extracted_from → treat as generic evidence

=== CONFLICT HANDLING ===

If any claim has conflict=true, introduce it with "However, ..." or "Although ...".

=== RULES ===

1. Every sentence must be grounded in the provided claims. Do not add outside knowledge.
2. Preserve numbers, benchmark names, and qualifiers exactly as given.
3. If all claims are empty or none are provided, reply: "Insufficient verified evidence to answer."
4. Do not mention chunk IDs or node IDs in the prose.

=== OUTPUT FORMAT ===

Return ONLY a JSON object with this exact shape:
{
  "sentences": [
    {
      "text": "one sentence of the answer",
      "chunk_id": "chunk_uid of the source claim",
      "doc_id": "paper_id of the source claim",
      "section_title": "section title or null",
      "scicite_label": "relation label",
      "rel_score": 0.91,
      "verdict": "Supported",
      "conflict": false
    }
  ]
}

One entry per sentence. No extra keys. No markdown fences.
""".strip()


def build_answer_generator_user_prompt(
    query: str,
    claims: list[dict],
) -> str:
    """Build the user prompt for Agent 4.

    Each dict in *claims* must contain:
        text, chunk_id, doc_id, section_title, scicite_label, rel_score, verdict, conflict
    """
    if not claims:
        claims_str = "(no verified claims)"
    else:
        lines = []
        for i, c in enumerate(claims, start=1):
            conflict_flag = " [CONFLICT]" if c.get("conflict") else ""
            lines.append(
                f"{i}. [{c.get('scicite_label', '')}]{conflict_flag} "
                f"chunk={c.get('chunk_id', '?')} doc={c.get('doc_id', '?')} "
                f"section={c.get('section_title') or 'N/A'} "
                f"rel_score={c.get('rel_score', 0.0):.2f} verdict={c.get('verdict', '?')}\n"
                f"   \"{c.get('text', '')}\""
            )
        claims_str = "\n\n".join(lines)

    return f"""Query:
{query}

Verified claims:
{claims_str}

Write a grounded answer. Return JSON.
""".strip()