from __future__ import annotations
from collections import Counter
import logging
import time
from schemas.state import WorkflowState, RetrievedDocument, EvidenceGraph, FinalAnswer
from schemas.objects import SubQuery, IMRaDSection
from retrieval.retriever import ChunkResult

_FATAL_PATTERNS = (
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",      
    "ConnectionError",
    "ServiceUnavailableError",
)

logger = logging.getLogger(__name__)

def log_step(state: WorkflowState, message: str) -> WorkflowState:
    state.logs.append(message)
    logger.info(message)
    return state


def _is_fatal(exc: Exception) -> bool:
    exc_type = type(exc).__name__
    exc_str = str(exc)
    return any(pat in exc_type or pat in exc_str for pat in _FATAL_PATTERNS)


def decompose_node(state: WorkflowState, services) -> WorkflowState:

    try:
        state = log_step(state, "[DECOMPOSER NODE] Starting decomposition")

        # sub-queries with section mapping and budget weights
        sub_queries = services.decomposer.decompose(state.query)

        if not sub_queries:
            # fallback to original query
            state = log_step(state, "[DECOMPOSER NODE] No valid sub-queries found, falling back to original query")
            sub_queries = [SubQuery(
                text=state.query,
                sections=["Abstract", IMRaDSection.INTRODUCTION],
                budget_weight=1.0
            )]

        state.sub_queries = sub_queries
        state.decomposition_done = True

        state = log_step(
            state,
            f"[DECOMPOSER NODE] Decomposition complete: {len(state.sub_queries)} sub-query(ies) with section mapping",
        )
        return state

    except Exception as e:
        state.errors.append(f"[DECOMPOSER NODE] decompose_node: {str(e)}")
        if _is_fatal(e):
            logger.error("[DECOMPOSER NODE] Fatal error — aborting pipeline: %s", e)
            state.fatal_error = True
            state.decomposition_done = True
            state = log_step(state, f"[DECOMPOSER NODE] Fatal error, pipeline aborted: {e}")
            return state
        state.sub_queries = [SubQuery(
            text=state.query,
            sections=[IMRaDSection.INTRODUCTION],
            budget_weight=1.0
        )]
        state.decomposition_done = True
        state = log_step(state, "[DECOMPOSER NODE] Decomposition failed, fallback to original query")
        return state


def retrieval_node(state: WorkflowState, services) -> WorkflowState:

    try:
        state = log_step(state, "[RETRIEVAL NODE] Starting hybrid retrieval")

        if not state.sub_queries:
            
            state.sub_queries = [SubQuery(
                text=state.query,
                sections=["Abstract", IMRaDSection.INTRODUCTION],
                budget_weight=1.0
            )]

        all_chunks: list[ChunkResult] = []
        # Maps chunk_uid → set of sub_query indices (0-based) that retrieved it
        chunk_to_sqs: dict[str, set[int]] = {}

        def _retrieve_one(idx_sq):
            idx, sq = idx_sq
            query_text = sq.text
            target_sections = [s.value if isinstance(s, IMRaDSection) else s for s in sq.sections] if sq.sections else None

            query_embeddings = services.embedder.embed_query(query_text)

            sparse_embeddings = None
            if hasattr(query_embeddings, 'dense'):  # BGEOutput
                dense_vec = query_embeddings.dense.tolist()
                sparse_embeddings = query_embeddings.sparse[0] if query_embeddings.sparse else None
            else:
                dense_vec = query_embeddings.tolist()

            # Scale top_k by budget_weight only when above 1.0 (high-priority sub-queries).
            # Equal-weight queries (budget_weight=1.0) use top_k as-is so N sub-queries
            # don't silently multiply retrieval by N.
            per_sq_top_k = max(state.top_k, int(sq.budget_weight * state.top_k)) if sq.budget_weight > 1.0 else state.top_k
            chunk_results = services.retriever.retrieve(
                embeddings=dense_vec,
                query_text=query_text,
                top_k=per_sq_top_k,
                sparse_embeddings=sparse_embeddings,
                target_sections=target_sections,
            )
            return idx, sq, chunk_results

        # Sub-queries are run sequentially to avoid overwhelming the Qdrant instance.
        # Parallel concurrent searches against a single mmap+quantized Qdrant caused
        # "Operation 'Search' timed out after 60s" errors on every sub-query.
        # A brief inter-query sleep lets Qdrant drain its work queue between searches.
        _INTER_QUERY_SLEEP_S = 0.05
        n_sqs = len(state.sub_queries)
        sq_results: list[tuple] = [None] * n_sqs  # type: ignore[list-item]
        for idx, sq in enumerate(state.sub_queries, 1):
            sq_results[idx - 1] = _retrieve_one((idx, sq))
            if idx < n_sqs:
                time.sleep(_INTER_QUERY_SLEEP_S)

        for idx, sq, chunk_results in sq_results:
            sq_idx = idx - 1
            for chunk in chunk_results:
                chunk_to_sqs.setdefault(chunk.chunk_uid, set()).add(sq_idx)

            section_counts = Counter(c.section_title or "unknown" for c in chunk_results)
            section_summary = ", ".join(f"{s}×{n}" for s, n in section_counts.items())
            score_range = (
                f"max={max(c.score for c in chunk_results):.2f} "
                f"min={min(c.score for c in chunk_results):.2f}"
                if chunk_results else "no results"
            )
            state = log_step(
                state,
                f"[RETRIEVAL NODE] Sub-query {idx}/{n_sqs} "
                f'"{sq.text[:60]}" → {len(chunk_results)} chunks | '
                f"sections: [{section_summary}] | scores: {score_range}"
            )
            all_chunks.extend(chunk_results)

        # deduplicate across sub-queries, keeping max score per chunk,
        # apply score_threshold from pipeline config, then sort and hard-cap at top_k
        pre_dedup = len(all_chunks)
        unique_chunks = services.retriever.deduplicate_chunks(all_chunks)
        unique_chunks = [c for c in unique_chunks if c.score >= state.score_threshold]
        unique_chunks = sorted(unique_chunks, key=lambda c: c.score, reverse=True)[:state.top_k]
        dropped = pre_dedup - len(unique_chunks)

        all_docs = [
            RetrievedDocument(
                doc_id=chunk.paper_id,
                chunk_id=chunk.chunk_uid,
                content=chunk.embed_text,
                score=chunk.score,
                section_title=chunk.section_title,
                paper_title=chunk.paper_title,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks,
                cite_spans=chunk.cite_spans,
                sub_query_indices=sorted(chunk_to_sqs.get(chunk.chunk_uid, set())),
            )
            for chunk in unique_chunks
        ]

        state.retrieved_documents = all_docs
        state.retrieval_done = True

        # summary stats over final deduplicated set
        if all_docs:
            scores = [d.score for d in all_docs]
            all_target_sections = {
                s.value if isinstance(s, IMRaDSection) else s for sq in state.sub_queries for s in (sq.sections or [])
            }
            section_hits = sum(
                1 for d in all_docs if d.section_title in all_target_sections
            )
            state = log_step(
                state,
                f"[RETRIEVAL NODE] {len(all_docs)} unique chunks "
                f"({dropped} duplicates merged by max-score) | "
                f"scores: max={max(scores):.2f} mean={sum(scores)/len(scores):.2f} min={min(scores):.2f} | "
                f"section hit rate: {section_hits}/{len(all_docs)} "
                f"({100 * section_hits // len(all_docs)}%)"
            )
        else:
            state = log_step(
                state,
                f"[RETRIEVAL NODE] 0 chunks retrieved across {len(state.sub_queries)} sub-queries",
            )
        return state

    except Exception as e:
        state.errors.append(f"[RETRIEVAL NODE] retrieval_node: {str(e)}")
        logger.error("[RETRIEVAL NODE] Fatal error — aborting pipeline: %s", e)
        state.fatal_error = True
        state.retrieved_documents = []
        state.retrieval_done = True
        state = log_step(state, f"[RETRIEVAL NODE] Fatal error, pipeline aborted: {e}")
        return state


def evidence_graph_node(state: WorkflowState, services) -> WorkflowState:

    try:
        state = log_step(state, "[EVIDENCE GRAPH NODE] Starting evidence graph construction")

        graph, hop_stats = services.evidence_graph_builder.build(
            query=state.query,
            sub_queries=state.sub_queries,
            documents=state.retrieved_documents,
            enable_hop=state.enable_hop,
        )

        state.evidence_graph = graph
        state.graph_done = True

        state = log_step(
            state,
            f"[EVIDENCE GRAPH NODE] Evidence graph complete: {len(graph.nodes)} nodes, {len(graph.edges)} edges",
        )

        # Log hop diagnostic summary into state so it appears in saved state.json
        hs = hop_stats
        n_hc = hs.get("n_hop_claims", 0)
        n_spans = hs.get("n_total_cite_spans", 0)
        succeeded = hs.get("hop_succeeded", 0)
        if n_hc == 0 and n_spans == 0:
            hop_summary = (
                f"[EVIDENCE GRAPH NODE] Multi-hop: 0 hop claims (0 cite_spans available — "
                f"ID resolution may be incomplete)"
            )
        elif n_hc == 0:
            hop_summary = (
                f"[EVIDENCE GRAPH NODE] Multi-hop: 0 hop claims despite {n_spans} cite_spans — "
                f"LLM judged all claims self-contained"
            )
        else:
            hop_summary = (
                f"[EVIDENCE GRAPH NODE] Multi-hop: {n_hc} hop claim(s) attempted → "
                f"{succeeded} succeeded | "
                f"no_linked_citations={hs.get('hop_no_linked_citations', 0)} "
                f"citation_raw_mismatch={hs.get('hop_citation_raw_mismatch', 0)} "
                f"no_resolved_id={hs.get('hop_no_resolved_id', 0)} "
                f"retrieval_zero={hs.get('hop_retrieval_zero', 0)} | "
                f"cite_spans_available={n_spans}"
            )
        state = log_step(state, hop_summary)
        state.hop_stats = hop_stats

        return state

    except Exception as e:
        state.errors.append(f"[EVIDENCE GRAPH NODE] evidence_graph_node: {str(e)}")
        state.evidence_graph = EvidenceGraph()
        state.graph_done = True
        state = log_step(state, f"[EVIDENCE GRAPH NODE] Evidence graph construction failed: {str(e)}")
        return state


def judge_node(state: WorkflowState, services) -> WorkflowState:

    try:
        state = log_step(state, "[JUDGE NODE] Starting evidence judging")

        result = services.judge.filter(
            query=state.query,
            evidence_graph=state.evidence_graph,
            documents=state.retrieved_documents,
        )

        state.evidence_graph = result.evidence_graph
        state.verdict_details = {cid: vd.dict() for cid, vd in result.verdict_details.items()}
        state.judge_done = True

        services.evidence_graph_builder.render_after(result.evidence_graph)

        state = log_step(
            state,
            f"[JUDGE NODE] Judging complete: {len(result.verdict_details)} verdicts merged into evidence graph",
        )
        return state

    except Exception as e:
        state.errors.append(f"[JUDGE NODE] judge_node: {str(e)}")
        state.judge_done = True
        state = log_step(state, "[JUDGE NODE] Judging failed, keeping existing evidence graph")
        return state

def answer_node(state: WorkflowState, services) -> WorkflowState:

    try:
        state = log_step(state, "[ANSWER NODE] Starting final answer generation")

        answer = services.answer_generator.generate(
            query=state.query,
            sub_queries=state.sub_queries,
            evidence_graph=state.evidence_graph,
            documents=state.retrieved_documents,
            verdict_details=state.verdict_details,
        )

        if not isinstance(answer, FinalAnswer):
            answer = FinalAnswer(**answer)

        state.final_answer = answer
        state.answer_done = True

        state = log_step(
            state,
            f"[ANSWER NODE] Answer generation complete: {len(answer.sentences)} sentences, {len(answer.text)} chars",
        )
        return state

    except Exception as e:
        state.errors.append(f"[ANSWER NODE] answer_node: {str(e)}")
        state.final_answer = FinalAnswer(
            text="I could not generate a reliable answer.",
            sentences=[],
            reasoning_summary="Answer generation failed.",
        )
        state.answer_done = True
        state = log_step(state, "[ANSWER NODE] Answer generation failed")
        return state
