from typing import Protocol, runtime_checkable, Union, Any, Optional
import numpy as np
from schemas.objects import (
    SubQuery,
    RetrievedDocument,
    EvidenceGraph,
    FinalAnswer,
    JudgementResult,
)

@runtime_checkable
class Decomposer(Protocol):
    def decompose(self, query: str) -> list[SubQuery]:
        ...

@runtime_checkable
class Embedder(Protocol):
    def embed_query(self, text: str) -> Union[np.ndarray, Any]:
        ...

    def embed_passages(self, texts: list[str]) -> Union[np.ndarray, Any]:
        ...

@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self,
        embeddings: list[float],
        query_text: str,
        top_k: int = 5,
        sparse_embeddings: Optional[dict] = None,
        target_sections: Optional[list[str]] = None,
    ) -> list[Any]:  
        ...

@runtime_checkable
class EvidenceGraphBuilder(Protocol):
    def build(
        self,
        query: str,
        sub_queries: list[SubQuery],
        documents: list[RetrievedDocument],
    ) -> EvidenceGraph:
        ...

@runtime_checkable
class Judge(Protocol):
    def filter(
        self,
        query: str,
        evidence_graph: EvidenceGraph,
        documents: list[RetrievedDocument],
    ) -> JudgementResult:
        ...

@runtime_checkable
class AnswerGenerator(Protocol):
    def generate(
        self,
        query: str,
        sub_queries: list[SubQuery],
        evidence_graph: EvidenceGraph,
        documents: list[RetrievedDocument],
    ) -> FinalAnswer:
        ...
