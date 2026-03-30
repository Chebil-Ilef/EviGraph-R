from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field


class IMRaDSection(str, Enum):

    ABSTRACT = "Abstract"
    INTRODUCTION = "Introduction"
    METHODS = "Methods"
    RESULTS = "Results"
    EXPERIMENTS = "Experiments"
    RELATED_WORK = "Related Work"
    DISCUSSION = "Discussion"
    CONCLUSION = "Conclusion"


class SubQuery(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="The decomposed sub-question text.",
    )
    sections: List[IMRaDSection] = Field(
        default_factory=list,
        description="Target IMRaD sections most likely to contain the answer.",
    )
    budget_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Retrieval budget weight (0.0-1.0). Higher = more important.",
    )


class DecompositionResult(BaseModel):
    should_decompose: bool = Field(
        ...,
        description="Whether the query benefits from decomposition.",
    )
    sub_queries: List[SubQuery] = Field(
        default_factory=list,
        description="Normalized list of sub-questions to retrieve against.",
    )


class RetrievedDocument(BaseModel):
    # just a suggestion NOT done yet
    doc_id: str
    chunk_id: Optional[str] = None
    title: Optional[str] = None
    section_title: Optional[str] = None
    content: str
    score: float = 0.0
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceNode(BaseModel):
    # just a suggestion NOT done yet
    node_id: str
    node_type: Literal["claim", "evidence", "paper", "concept"]
    text: str
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceEdge(BaseModel):
    # just a suggestion NOT done yet
    source: str
    target: str
    relation: str
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceGraph(BaseModel):
    # just a suggestion NOT done yet
    nodes: List[EvidenceNode] = Field(default_factory=list)
    edges: List[EvidenceEdge] = Field(default_factory=list)


class Citation(BaseModel):
    # just a suggestion NOT done yet
    doc_id: str
    chunk_id: Optional[str] = None
    title: Optional[str] = None
    section_title: Optional[str] = None
    score: Optional[float] = None


class FinalAnswer(BaseModel):
    # just a suggestion NOT done yet
    text: str = ""
    citations: List[Citation] = Field(default_factory=list)
    reasoning_summary: Optional[str] = None
