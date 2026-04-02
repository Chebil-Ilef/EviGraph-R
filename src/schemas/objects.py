from __future__ import annotations
from typing import Any, Dict, List, Optional
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
    doc_id: str = Field(..., description="Paper ID (arxiv ID)")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk text content")
    score: float = Field(default=0.0, description="Retrieval/reranker score")
    section_title: Optional[str] = Field(None, description="IMRaD section title")
    chunk_type: Optional[str] = Field(None, description="Chunk type (abstract/section)")
    chunk_index: Optional[int] = Field(None, description="Chunk position in paper")
    total_chunks: Optional[int] = Field(None, description="Total chunks in paper")
    cite_spans: Optional[Dict[str, Any]] = Field(None, description="Citation spans with resolved work_ids (doi/arxiv_id)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NodeType(str, Enum):
    CLAIM = "claim"
    EVIDENCE = "evidence"
    PAPER = "paper"
    CONCEPT = "concept"
    CHUNK = "chunk"


class EvidenceNode(BaseModel):
    node_id: str
    node_type: NodeType
    text: str
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceEdge(BaseModel):
    source: str
    target: str
    relation: str
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceGraph(BaseModel):
    nodes: List[EvidenceNode] = Field(default_factory=list)
    edges: List[EvidenceEdge] = Field(default_factory=list)


class ClaimType(str, Enum):
    ATOMIC_FACTUAL = "atomic_factual"
    INFERENTIAL = "inferential"


class HopDepth(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


class VerdictType(str, Enum):
    SUPPORTED = "Supported"
    CONTRADICTED = "Contradicted"
    NOT_SUPPORTED = "Not-Supported"
    INCONCLUSIVE = "Inconclusive"


class VerdictDetail(BaseModel):
    verdict: str = Field(..., description="Verdict: Supported, Contradicted, Not-Supported, Inconclusive")
    verifier_used: str = Field(..., description="Verifier that produced this verdict: npm, nli, llm_judge")
    evidence_trail: List[Dict[str, Any]] = Field(default_factory=list, description="Evidence chunks used for verification")
    error_stage: Optional[str] = Field(None, description="Error stage if verification failed")
    claim_type: Optional[str] = Field(None, description="Claim type: atomic_factual or inferential")
    hop_depth: Optional[str] = Field(None, description="Hop depth: single or multi")


class JudgementResult(BaseModel):
    filtered_documents: List[RetrievedDocument] = Field(default_factory=list, description="Documents passing verification")
    judged_relations: List[EvidenceEdge] = Field(default_factory=list, description="Edges in verified evidence graph")
    verdict_details: Dict[str, VerdictDetail] = Field(default_factory=dict, description="Per-claim verdicts (claim_id → VerdictDetail)")


class Citation(BaseModel):
    doc_id: str = Field(..., description="Paper ID (arxiv ID)")
    chunk_id: Optional[str] = Field(None, description="Chunk ID where claim originated")
    section_title: Optional[str] = Field(None, description="IMRaD section (Methods, Results, etc.)")
    scicite_label: Optional[str] = Field(None, description="SciCite relation type (METHOD, RESULT_COMPARISON, etc.)")
    rel_score: Optional[float] = Field(None, description="Relevance score (0-1)")
    verdict: Optional[str] = Field(None, description="Verification verdict (Supported, Contradicted, Not-Supported)")
    title: Optional[str] = Field(None, description="Paper title (optional)")


class AnnotatedSentence(BaseModel):
    text: str = Field(..., description="Sentence text")
    citations: List[Citation] = Field(default_factory=list, description="Citation metadata for this sentence")
    conflict_flag: bool = Field(False, description="True if sources contradict each other")


class FinalAnswer(BaseModel):
    text: str = Field("", description="Full answer text (plain sentences joined)")
    sentences: List[AnnotatedSentence] = Field(default_factory=list, description="Sentences with per-sentence citations")
    reasoning_summary: Optional[str] = Field(None, description="Optional summary of generation reasoning")
