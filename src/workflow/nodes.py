from __future__ import annotations

from schemas.state import WorkflowState, RetrievedDocument, EvidenceGraph, FinalAnswer
from schemas.objects import SubQuery, IMRaDSection, EvidenceEdge

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
                sections=[IMRaDSection.ABSTRACT, IMRaDSection.INTRODUCTION],
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
            sections=[IMRaDSection.ABSTRACT, IMRaDSection.INTRODUCTION],
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
                sections=[IMRaDSection.ABSTRACT, IMRaDSection.INTRODUCTION],
                budget_weight=1.0
            )]

        all_docs: list[RetrievedDocument] = []
        seen = set()

        for sq in state.sub_queries:
            query_text = sq.text
            target_sections = [s.value for s in sq.sections] if sq.sections else None

            # embed query
            query_embeddings = services.embedder.embed_query(query_text)

            # BGE-M3 sparse embeddings
            sparse_embeddings = None
            if hasattr(query_embeddings, 'dense'):  # BGEOutput
                dense_vec = query_embeddings.dense.tolist()
                sparse_embeddings = query_embeddings.sparse[0] if query_embeddings.sparse else None
            else:
                dense_vec = query_embeddings.tolist()

            # hybrid retrieval with section filtering and reranking
            chunk_results = services.retriever.retrieve(
                embeddings=dense_vec,
                query_text=query_text,
                top_k=int(sq.budget_weight * 10),  # budget-weighted top-k
                sparse_embeddings=sparse_embeddings,
                target_sections=target_sections,
            )

            for chunk in chunk_results:
                dedup_key = (chunk.paper_id, chunk.chunk_uid)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                all_docs.append(RetrievedDocument(
                    doc_id=chunk.paper_id,
                    chunk_id=chunk.chunk_uid,
                    content=chunk.embed_text,
                    score=chunk.score,
                    section_title=chunk.section_title,
                    chunk_type=chunk.chunk_type,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks,
                    cite_spans=chunk.cite_spans,
                ))

        state.retrieved_documents = all_docs
        state.retrieval_done = True

        state = log_step(
            state,
            f"[RETRIEVAL NODE] Retrieved {len(all_docs)} unique chunks across {len(state.sub_queries)} sub-queries",
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


# NOT YET DONE JUST A PLACEHOLDER SUGGESTATION
def judge_node(state: WorkflowState, services) -> WorkflowState:
    """
    Agent 3: Judge / filter evidence graph by relevance score.
    Expects services.judge.filter(query, evidence_graph, documents) -> dict
    """
    try:
        state = log_step(state, "Starting evidence judging")

        result = services.judge.filter(
            query=state.query,
            evidence_graph=state.evidence_graph,
            documents=state.retrieved_documents,
        )

        filtered_docs = result.get("filtered_documents", [])
        judged_relations = result.get("judged_relations", [])

        state.filtered_evidence = [
            d if isinstance(d, RetrievedDocument) else RetrievedDocument(**d)
            for d in filtered_docs
        ]
        state.judged_relations = [
            e if isinstance(e, EvidenceEdge) else EvidenceEdge(**e)
            for e in judged_relations
        ]
        state.judge_done = True

        state = log_step(
            state,
            f"Judge complete: {len(state.filtered_evidence)} filtered docs",
        )
        return state

    except Exception as e:
        state.errors.append(f"judge_node: {str(e)}")
        state.filtered_evidence = state.retrieved_documents[:]
        state.judge_done = True
        state = log_step(state, "Judge failed, fallback to retrieved documents")
        return state

# NOT YET DONE JUST A PLACEHOLDER SUGGESTATION
def answer_node(state: WorkflowState, services) -> WorkflowState:
    """
    Agent 4: Final answer generation.
    Expects services.answer_generator.generate(...) -> FinalAnswer | dict
    """
    try:
        state = log_step(state, "Starting final answer generation")

        answer = services.answer_generator.generate(
            query=state.query,
            sub_queries=state.sub_queries,
            evidence_graph=state.evidence_graph,
            documents=state.filtered_evidence or state.retrieved_documents,
        )

        if not isinstance(answer, FinalAnswer):
            answer = FinalAnswer(**answer)

        state.final_answer = answer
        state.answer_done = True

        state = log_step(state, "Final answer generation complete")
        return state

    except Exception as e:
        state.errors.append(f"answer_node: {str(e)}")
        state.final_answer = FinalAnswer(
            text="I could not generate a reliable answer.",
            citations=[],
            reasoning_summary="Answer generation failed.",
        )
        state.answer_done = True
        state = log_step(state, "Answer generation failed")
        return state