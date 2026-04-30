from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from .objects import SubQuery, RetrievedDocument, EvidenceGraph, FinalAnswer


class WorkflowState(BaseModel):
    # Input
    query: str

    # Pipeline config (forwarded from QueryRequest)
    top_k: int = 15
    score_threshold: float = 0.15
    enable_hop: bool = True

    # Agent 1: decomposition
    sub_queries: List[SubQuery] = Field(default_factory=list)
    decomposition_done: bool = False

    # Retrieval
    retrieved_documents: List[RetrievedDocument] = Field(default_factory=list)
    retrieval_done: bool = False

    # Agent 2: evidence graph construction
    evidence_graph: Optional[EvidenceGraph] = None
    graph_done: bool = False

    # Agent 3: judge / filtering
    verdict_details: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    judge_done: bool = False

    # Agent 4: final answer
    final_answer: Optional[FinalAnswer] = None
    answer_done: bool = False

    # Control / debug
    errors: List[str] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)
    hop_stats: Dict[str, Any] = Field(default_factory=dict)

    fatal_error: bool = False
