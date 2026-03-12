from __future__ import annotations

from schemas.state import (
    WorkflowState,
    SubQuery,
    RetrievedDocument,
    EvidenceGraph,
    FinalAnswer,
)


def log_step(state: WorkflowState, message: str) -> WorkflowState:
    state.logs.append(message)
    return state


def decompose_node(state: WorkflowState, services) -> WorkflowState:
    """
    Agent 1: Query decomposition.
    Expects services.decomposer.decompose(query) -> list[SubQuery]
    """
    
    try:
        state = log_step(state, "Starting decomposition")

        sub_questions = services.decomposer.decompose(state.query)

        if not sub_questions:
            sub_questions = [state.query]

        state.sub_queries= sub_questions
        state.decomposition_done = True

        state = log_step(
            state,
            f"Decomposition complete: {len(state.sub_queries)} sub-query(ies)",
        )
        return state

    except Exception as e:
        state.errors.append(f"decompose_node: {str(e)}")
        state.sub_queries = [state.query]
        state.decomposition_done = True
        state = log_step(state, "Decomposition failed, fallback to original query")
        return state

# NOT YET DONE JUST A PLACEHOLDER SUGGESTATION
def retrieval_node(state: WorkflowState, services) -> WorkflowState:
    """
    NOT YET DONE JUST A PLACEHOLDER SUGGESTATION
    Hybrid retrieval over all sub-queries.
    Expects services.retriever.retrieve(query: str) -> list[RetrievedDocument] | list[dict]
    """
    try:
        state = log_step(state, "Starting retrieval")

        all_docs: list[RetrievedDocument] = []
        seen = set()

        queries = state.sub_queries or [SubQuery(id="sq_1", text=state.query)]

        for sq in queries:
            docs = services.retriever.retrieve(sq.text)

            for doc in docs:
                if not isinstance(doc, RetrievedDocument):
                    doc = RetrievedDocument(**doc)

                dedup_key = (doc.doc_id, doc.chunk_id)
                if dedup_key in seen:
                    continue

                seen.add(dedup_key)
                all_docs.append(doc)

        state.retrieved_documents = all_docs
        state.retrieval_done = True

        state = log_step(
            state,
            f"Retrieval complete: {len(state.retrieved_documents)} unique document chunks",
        )
        return state

    except Exception as e:
        state.errors.append(f"retrieval_node: {str(e)}")
        state.retrieved_documents = []
        state.retrieval_done = True
        state = log_step(state, "Retrieval failed")
        return state

# NOT YET DONE JUST A PLACEHOLDER SUGGESTATION
def evidence_graph_node(state: WorkflowState, services) -> WorkflowState:
    """
    Agent 2: Build evidence graph from retrieved documents.
    Expects services.evidence_graph_builder.build(query, sub_queries, documents) -> EvidenceGraph | dict
    """
    try:
        state = log_step(state, "Starting evidence graph construction")

        graph = services.evidence_graph_builder.build(
            query=state.query,
            sub_queries=state.sub_queries,
            documents=state.retrieved_documents,
        )

        if not isinstance(graph, EvidenceGraph):
            graph = EvidenceGraph(**graph)

        state.evidence_graph = graph
        state.graph_done = True

        state = log_step(
            state,
            f"Evidence graph complete: {len(graph.nodes)} nodes, {len(graph.edges)} edges",
        )
        return state

    except Exception as e:
        state.errors.append(f"evidence_graph_node: {str(e)}")
        state.evidence_graph = EvidenceGraph()
        state.graph_done = True
        state = log_step(state, "Evidence graph construction failed")
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
        state.judged_relations = judged_relations
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