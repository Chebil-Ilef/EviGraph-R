from __future__ import annotations
from collections import Counter
from schemas.state import WorkflowState, RetrievedDocument, EvidenceGraph, FinalAnswer
from schemas.objects import SubQuery, IMRaDSection
from retrieval.retriever import ChunkResult

def log_step(state: WorkflowState, message: str) -> WorkflowState:
    state.logs.append(message)
    return state


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

        for idx, sq in enumerate(state.sub_queries, 1):
            query_text = sq.text
            target_sections = [s.value if isinstance(s, IMRaDSection) else s for s in sq.sections] if sq.sections else None

            # embed query
            query_embeddings = services.embedder.embed_query(query_text)

            # BGE-M3 sparse embeddings
            sparse_embeddings = None
            if hasattr(query_embeddings, 'dense'):  # BGEOutput
                dense_vec = query_embeddings.dense.tolist()
                sparse_embeddings = query_embeddings.sparse[0] if query_embeddings.sparse else None
            else:
                dense_vec = query_embeddings.tolist()

            chunk_results = services.retriever.retrieve(
                embeddings=dense_vec,
                query_text=query_text,
                top_k=int(sq.budget_weight * 10),  # budget-weighted top-k
                sparse_embeddings=sparse_embeddings,
                target_sections=target_sections,
            )

            sq_idx = idx - 1  # 0-based
            for chunk in chunk_results:
                chunk_to_sqs.setdefault(chunk.chunk_uid, set()).add(sq_idx)

            # per-sub-query log: count, section breakdown, score range
            section_counts = Counter(c.section_title or "unknown" for c in chunk_results)
            section_summary = ", ".join(f"{s}×{n}" for s, n in section_counts.items())
            score_range = (
                f"max={max(c.score for c in chunk_results):.2f} "
                f"min={min(c.score for c in chunk_results):.2f}"
                if chunk_results else "no results"
            )
            state = log_step(
                state,
                f"[RETRIEVAL NODE] Sub-query {idx}/{len(state.sub_queries)} "
                f'"{query_text[:60]}" → {len(chunk_results)} chunks | '
                f"sections: [{section_summary}] | scores: {score_range}"
            )

            all_chunks.extend(chunk_results)

        # deduplicate across sub-queries, keeping max score per chunk
        pre_dedup = len(all_chunks)
        unique_chunks = services.retriever.deduplicate_chunks(all_chunks)
        dropped = pre_dedup - len(unique_chunks)

        all_docs = [
            RetrievedDocument(
                doc_id=chunk.paper_id,
                chunk_id=chunk.chunk_uid,
                content=chunk.embed_text,
                score=chunk.score,
                section_title=chunk.section_title,
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
        state.retrieved_documents = []
        state.retrieval_done = True
        state = log_step(state, f"[RETRIEVAL NODE] Retrieval failed: {str(e)}")
        return state


def evidence_graph_node(state: WorkflowState, services) -> WorkflowState:
    
    try:
        state = log_step(state, "[EVIDENCE GRAPH NODE] Starting evidence graph construction")

        graph = services.evidence_graph_builder.build(
            query=state.query,
            sub_queries=state.sub_queries,
            documents=state.retrieved_documents,
        )

        state.evidence_graph = graph
        state.graph_done = True

        state = log_step(
            state,
            f"[EVIDENCE GRAPH NODE] Evidence graph complete: {len(graph.nodes)} nodes, {len(graph.edges)} edges",
        )
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

        state.filtered_evidence = list(result.filtered_documents)
        state.judged_relations = list(result.judged_relations)
        state.verdict_details = {cid: vd.dict() for cid, vd in result.verdict_details.items()}
        state.judge_done = True

        state = log_step(
            state,
            f"[JUDGE NODE] Judging complete: {len(state.filtered_evidence)} filtered docs; {len(result.verdict_details)} verdicts",
        )
        return state

    except Exception as e:
        state.errors.append(f"[JUDGE NODE] judge_node: {str(e)}")
        state.filtered_evidence = state.retrieved_documents[:]
        state.judge_done = True
        state = log_step(state, "[JUDGE NODE] Judging failed, fallback to retrieved documents")
        return state

def answer_node(state: WorkflowState, services) -> WorkflowState:

    try:
        state = log_step(state, "[ANSWER NODE] Starting final answer generation")

        answer = services.answer_generator.generate(
            query=state.query,
            sub_queries=state.sub_queries,
            evidence_graph=state.evidence_graph,
            documents=state.filtered_evidence or state.retrieved_documents,
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