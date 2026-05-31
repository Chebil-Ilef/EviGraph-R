from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field
from evigraph.schemas.objects import AnnotatedSentence, EvidenceGraph


# Request

class PipelineConfig(BaseModel):
    top_k: int = Field(15, ge=1, le=100)
    score_threshold: float = Field(0.0, ge=0.0, le=20.0)
    enable_hop: bool = True
    embedding_model: str = "bge-m3"
    target_sections: list[str] | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    config: PipelineConfig = Field(default_factory=lambda: PipelineConfig.model_validate({}))


# Response

QueryResponseStatus = Literal["completed", "failed"]


class QueryResponse(BaseModel):
    job_id: str
    status: QueryResponseStatus
    query: str
    answer: str
    sentences: list[AnnotatedSentence]
    graph: EvidenceGraph
    scorecard: dict
    errors: list[str]
    elapsed_s: float


# SSE streaming events

SSEEventType = Literal[
    "decomposed",
    "retrieved",
    "graph_built",
    "judged",
    "completed",
    "error",
]


class SSEEvent(BaseModel):
    event: SSEEventType
    data: dict
