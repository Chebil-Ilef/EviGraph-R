from __future__ import annotations

import asyncio
import time
import uuid
import threading
from collections import Counter
from collections.abc import AsyncGenerator
from functools import partial
from typing import Union

from evigraph.agents.answer_generator import AnswerGeneratorAgent
from evigraph.agents.decomposer import DecomposerAgent
from evigraph.agents.evidence_graph_builder import EvidenceGraphBuilderAgent
from evigraph.agents.judge import JudgeAgent
from evigraph.retrieval.embedder import Embedder
from evigraph.retrieval.retriever import HybridQueryRetriever
from evigraph.schemas.objects import EvidenceGraph, FinalAnswer
from evigraph.schemas.state import WorkflowState
from evigraph.utils.llm import get_llm_client
from evigraph.utils.qdrant import ensure_payload_indexes
from evigraph.workflow.graph import WorkflowServices, build_workflow_graph

from evigraph.api.schemas import QueryRequest, QueryResponse, QueryResponseStatus, SSEEvent, SSEEventType

import logging

logger = logging.getLogger(__name__)

_STREAM_DONE = object()

_QueueItem = Union[tuple[str, WorkflowState], Exception, object]  # object covers _STREAM_DONE sentinel

_NODE_TO_EVENT: dict[str, SSEEventType] = {
    "decompose":            "decomposed",
    "retrieve":             "retrieved",
    "build_evidence_graph": "graph_built",
    "judge":                "judged",
    "generate_answer":      "completed",
}


class WorkflowRunner:

    def __init__(self, model_key: str = "bge-m3") -> None:
        self._model_key = model_key
        self._services = self._build_services(model_key)
        self._workflow = build_workflow_graph(self._services)
        logger.info("WorkflowRunner ready (model=%s)", model_key)


    async def run_query(self, request: QueryRequest) -> QueryResponse:
        job_id = str(uuid.uuid4())
        cfg = request.config
        initial = WorkflowState(
            query=request.query,
            top_k=cfg.top_k,
            score_threshold=cfg.score_threshold,
            enable_hop=cfg.enable_hop,
        ).model_dump()

        _SYSTEM_ERROR_MESSAGE = (
            "We were not able to respond to your query due to a system issue. "
            "Please try again later."
        )

        t0 = time.perf_counter()
        try:
            loop = asyncio.get_event_loop()
            state, _ = await loop.run_in_executor(
                None, partial(self._run_pipeline, initial)
            )
        except Exception as exc:
            logger.exception("Pipeline failed for job %s", job_id)
            return QueryResponse(
                job_id=job_id,
                status="failed",
                query=request.query,
                answer=_SYSTEM_ERROR_MESSAGE,
                sentences=[],
                graph=EvidenceGraph(),
                scorecard={},
                errors=[str(exc)],
                elapsed_s=round(time.perf_counter() - t0, 2),
            )

        if state.fatal_error:
            logger.error("Pipeline aborted with fatal error for job %s: %s", job_id, state.errors)
            return QueryResponse(
                job_id=job_id,
                status="failed",
                query=request.query,
                answer=_SYSTEM_ERROR_MESSAGE,
                sentences=[],
                graph=EvidenceGraph(),
                scorecard={},
                errors=state.errors,
                elapsed_s=round(time.perf_counter() - t0, 2),
            )

        answer = state.final_answer or FinalAnswer(
            text="",
            sentences=[],
            reasoning_summary=None,
        )

        return QueryResponse(
            job_id=job_id,
            status="completed",
            query=request.query,
            answer=answer.text,
            sentences=answer.sentences,
            graph=state.evidence_graph or EvidenceGraph(),
            scorecard=self._scorecard(state),
            errors=state.errors,
            elapsed_s=round(time.perf_counter() - t0, 2),
        )

    async def stream_query(self, request: QueryRequest) -> AsyncGenerator[SSEEvent, None]:
        cfg = request.config
        initial = WorkflowState(
            query=request.query,
            top_k=cfg.top_k,
            score_threshold=cfg.score_threshold,
            enable_hop=cfg.enable_hop,
        ).model_dump()

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[_QueueItem] = asyncio.Queue()

        def _worker() -> None:
            accumulated = dict(initial)
            try:
                for chunk in self._workflow.stream(accumulated, stream_mode="updates"):
                    for node_name, update in chunk.items():
                        accumulated.update(update)
                        state = WorkflowState(**accumulated)
                        loop.call_soon_threadsafe(queue.put_nowait, (node_name, state))
            except Exception as exc:
                logger.exception("Streaming pipeline failed")
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _STREAM_DONE)

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            item = await queue.get()
            if item is _STREAM_DONE:
                return
            if isinstance(item, Exception):
                yield SSEEvent(event="error", data={"error": str(item)})
                return

            assert isinstance(item, tuple)
            node_name, state = item
            assert isinstance(node_name, str) and isinstance(state, WorkflowState)
            if state.fatal_error:
                logger.error("Streaming pipeline aborted with fatal error after node '%s': %s", node_name, state.errors)
                yield SSEEvent(
                    event="error",
                    data={
                        "message": (
                            "We were not able to respond to your query due to a system issue. "
                            "Please try again later."
                        ),
                        "errors": state.errors,
                    },
                )
                return
            event_type = _NODE_TO_EVENT.get(node_name)
            if event_type is None:
                continue
            yield SSEEvent(event=event_type, data=self._event_data(event_type, state))


    def _run_pipeline(self, state_dict: dict) -> tuple[WorkflowState, dict[str, float]]:
        timings: dict[str, float] = {}
        accumulated = dict(state_dict)
        for chunk in self._workflow.stream(accumulated, stream_mode="updates"):
            for node_name, update in chunk.items():
                t0 = time.perf_counter()
                accumulated.update(update)
                timings[node_name] = round(time.perf_counter() - t0, 3)
        return WorkflowState(**accumulated), timings

    def _event_data(self, event_type: SSEEventType, state: WorkflowState) -> dict:
        if event_type == "completed":
            answer = state.final_answer or FinalAnswer(
                text="",
                sentences=[],
                reasoning_summary=None,
            )
            return {
                "answer":    answer.text,
                "sentences": [s.model_dump() for s in answer.sentences],
                "graph":     (state.evidence_graph or EvidenceGraph()).model_dump(),
                "scorecard": self._scorecard(state),
                "errors":    state.errors,
            }
        return self._stage_summary(event_type, state)

    @staticmethod
    def _stage_summary(event_type: str, state: WorkflowState) -> dict:
        if event_type == "decomposed":
            sub_queries = state.sub_queries or []
            return {
                "n_sub_queries": len(sub_queries),
                "sub_queries": [
                    {
                        "text": sq.text,
                        "sections": [s.value if hasattr(s, "value") else str(s) for s in (sq.sections or [])],
                        "budget_weight": round(sq.budget_weight, 2),
                    }
                    for sq in sub_queries
                ],
            }
        if event_type == "retrieved":
            docs = state.retrieved_documents or []
            seen: dict = {}
            for d in docs:
                if d.doc_id and d.doc_id not in seen:
                    seen[d.doc_id] = {
                        "doc_id": d.doc_id,
                        "title": d.paper_title or "",
                        "n_chunks": 0,
                    }
                if d.doc_id:
                    seen[d.doc_id]["n_chunks"] += 1
            return {"n_docs": len(docs), "papers": list(seen.values())}
        if event_type == "graph_built":
            g = state.evidence_graph or EvidenceGraph()
            node_counts = Counter(n.node_type.value for n in g.nodes)
            return {"n_nodes": len(g.nodes), "n_edges": len(g.edges), "node_types": dict(node_counts)}
        if event_type == "judged":
            vd = state.verdict_details or {}
            counts: Counter = Counter(v.get("verdict", "?") for v in vd.values())
            return {"n_judged": len(vd), "verdicts": dict(counts)}
        return {}

    @staticmethod
    def _scorecard(state: WorkflowState) -> dict:
        docs = state.retrieved_documents or []
        scores = [d.score for d in docs]
        graph = state.evidence_graph or EvidenceGraph()
        node_counts = Counter(n.node_type.value for n in graph.nodes)
        verdict_counts: Counter = Counter()
        verifier_counts: Counter = Counter()
        for vd in (state.verdict_details or {}).values():
            verdict_counts[vd.get("verdict", "?")] += 1
            verifier_counts[vd.get("verifier_used", "?")] += 1
        paper_ids = {n.doc_id for n in graph.nodes if n.node_type.value == "paper" and n.doc_id}
        return {
            "n_sub_queries": len(state.sub_queries or []),
            "n_docs":        len(docs),
            "n_papers":      len(paper_ids),
            "score_max":     round(max(scores), 4) if scores else None,
            "score_mean":    round(sum(scores) / len(scores), 4) if scores else None,
            "n_nodes":       len(graph.nodes),
            "n_edges":       len(graph.edges),
            "node_types":    dict(node_counts),
            "verdicts":      dict(verdict_counts),
            "verifiers":     dict(verifier_counts),
            "hop_stats":     state.hop_stats or {},
            "errors":        state.errors,
        }

    @staticmethod
    def _build_services(model_key: str) -> WorkflowServices:
        llm = get_llm_client()
        embedder = Embedder.from_model_key(model_key)
        retriever = HybridQueryRetriever(model_key=model_key)
        ensure_payload_indexes(retriever.client, retriever.collection_name)
        return WorkflowServices(
            decomposer=DecomposerAgent(llm_client=llm),
            embedder=embedder,
            retriever=retriever,
            evidence_graph_builder=EvidenceGraphBuilderAgent(
                llm_client=llm,
                retriever=retriever,
                embedder=embedder,
            ),
            judge=JudgeAgent(llm_client=llm),
            answer_generator=AnswerGeneratorAgent(llm_client=llm),
        )
