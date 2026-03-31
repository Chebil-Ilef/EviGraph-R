from __future__ import annotations
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from schemas.state import WorkflowState
from schemas.interfaces import (
    Embedder,
    Decomposer,
    Retriever,
    EvidenceGraphBuilder,
    Judge,
    AnswerGenerator,
)
from workflow.nodes import (
    decompose_node,
    retrieval_node,
    evidence_graph_node,
    judge_node,
    answer_node,
)


@dataclass
class WorkflowServices:
    decomposer: Decomposer
    embedder: Embedder
    retriever: Retriever
    evidence_graph_builder: EvidenceGraphBuilder
    judge: Judge
    answer_generator: AnswerGenerator

    def __post_init__(self) -> None:
        if not isinstance(self.decomposer, Decomposer):
            raise TypeError("decomposer must implement the Decomposer interface")
        if not isinstance(self.embedder, Embedder):
            raise TypeError("embedder must implement the Embedder interface")
        if not isinstance(self.retriever, Retriever):
            raise TypeError("retriever must implement the Retriever interface")
        if not isinstance(self.evidence_graph_builder, EvidenceGraphBuilder):
            raise TypeError(
                "evidence_graph_builder must implement the EvidenceGraphBuilder interface"
            )
        if not isinstance(self.judge, Judge):
            raise TypeError("judge must implement the Judge interface")
        if not isinstance(self.answer_generator, AnswerGenerator):
            raise TypeError(
                "answer_generator must implement the AnswerGenerator interface"
            )


def build_workflow_graph(services: WorkflowServices):
    graph = StateGraph(WorkflowState)

    graph.add_node("decompose", lambda state: decompose_node(state, services))
    graph.add_node("retrieve", lambda state: retrieval_node(state, services))
    graph.add_node("build_evidence_graph", lambda state: evidence_graph_node(state, services))
    graph.add_node("judge", lambda state: judge_node(state, services))
    graph.add_node("generate_answer", lambda state: answer_node(state, services))

    graph.set_entry_point("decompose")

    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "build_evidence_graph")
    graph.add_edge("build_evidence_graph", "judge")
    graph.add_edge("judge", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()