from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class SubQuery(BaseModel):
    # just a suggestion NOT done yet
    id: str = Field(..., description="Unique sub-query id, e.g. sq_1")
    text: str = Field(..., description="The decomposed sub-question text")


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