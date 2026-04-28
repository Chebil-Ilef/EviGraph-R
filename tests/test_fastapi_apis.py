from __future__ import annotations
import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from api.routes.query import _format_sse, query_stream
from api.runner import WorkflowRunner
from api.schemas import PipelineConfig, QueryRequest, QueryResponse, SSEEvent
from schemas.objects import AnnotatedSentence, Citation, EvidenceGraph, FinalAnswer


class Fixtures:

    @staticmethod
    def make_response(**overrides) -> QueryResponse:
        defaults = dict(
            job_id="test-job-1",
            status="completed",
            query="What is dropout?",
            answer="Dropout is a regularization technique.",
            sentences=[
                AnnotatedSentence(
                    text="Dropout prevents overfitting.",
                    citations=[Citation(doc_id="1234.5678", rel_score=0.92)],
                    conflict_flag=False,
                )
            ],
            graph=EvidenceGraph(),
            scorecard={"n_docs": 5, "verdicts": {"Supported": 3}},
            errors=[],
            elapsed_s=2.4,
        )
        defaults.update(overrides)
        return QueryResponse(**defaults)

    @staticmethod
    def make_sse_sequence() -> list[SSEEvent]:
        return [
            SSEEvent(event="decomposed",  data={"n_sub_queries": 2}),
            SSEEvent(event="retrieved",   data={"n_docs": 10}),
            SSEEvent(event="graph_built", data={"n_nodes": 24, "n_edges": 18}),
            SSEEvent(event="judged",      data={"n_judged": 8, "verdicts": {"Supported": 6}}),
            SSEEvent(event="completed",   data={"answer": "Dropout regularizes.", "sentences": [], "graph": {}, "scorecard": {}, "errors": []}),
        ]

    @staticmethod
    def make_runner(response: QueryResponse, stream: list[SSEEvent] | None = None) -> MagicMock:
        runner = MagicMock()
        runner._model_key = "bge-m3"
        runner._services.retriever.client = MagicMock()
        runner._services.retriever.collection_name = "unarxive_chunks"
        runner.run_query = AsyncMock(return_value=response)

        seq = stream or [
            SSEEvent(event="decomposed",  data={"n_sub_queries": 2}),
            SSEEvent(event="retrieved",   data={"n_docs": 10}),
            SSEEvent(event="graph_built", data={"n_nodes": 24, "n_edges": 18}),
            SSEEvent(event="judged",      data={"n_judged": 8, "verdicts": {"Supported": 6}}),
            SSEEvent(event="completed",   data={"answer": "Dropout regularizes.", "sentences": [], "graph": {}, "scorecard": {}, "errors": []}),
        ]

        async def _stream_gen(*_args, **_kwargs) -> AsyncGenerator[SSEEvent, None]:
            for ev in seq:
                yield ev

        runner.stream_query = _stream_gen
        return runner

    @staticmethod
    def make_client(runner: MagicMock) -> TestClient:
        from api.main import app
        app.state.runner = runner
        return TestClient(app, raise_server_exceptions=True)

    @staticmethod
    def parse_sse(raw: str) -> list[dict]:
        results = []
        current: dict = {}
        for line in raw.splitlines():
            if line.startswith("event:"):
                current["event"] = line[len("event:"):].strip()
            elif line.startswith("data:"):
                current["data"] = json.loads(line[len("data:"):].strip())
            elif line == "" and current:
                results.append(current)
                current = {}
        if current:
            results.append(current)
        return results

    @staticmethod
    async def get_sse_response(runner: MagicMock, query: str):
        app = MagicMock()
        app.state.runner = runner
        scope = {"type": "http", "method": "GET", "path": "/api/v1/query/stream", "headers": [], "app": app}
        request = Request(scope)

        response = await query_stream(request=request, q=query)
        parts: list[str] = []
        async for chunk in response.body_iterator:
            parts.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        body = "".join(parts)
        return response.status_code, dict(response.headers), body

class TestSchemas(Fixtures):

    def test_query_request_defaults(self):
        req = QueryRequest(query="What is attention?")
        assert req.config.top_k == 15
        assert req.config.score_threshold == 0.15
        assert req.config.enable_hop is True
        assert req.config.embedding_model == "bge-m3"
        assert req.config.target_sections is None

    def test_query_request_custom_config(self):
        req = QueryRequest(
            query="What is BatchNorm?",
            config=PipelineConfig(top_k=25, score_threshold=0.5, enable_hop=False),
        )
        assert req.config.top_k == 25
        assert req.config.enable_hop is False

    def test_query_request_rejects_short_query(self):
        with pytest.raises(Exception):
            QueryRequest(query="hi")

    def test_query_request_rejects_empty_query(self):
        with pytest.raises(Exception):
            QueryRequest(query="")

    def test_query_request_rejects_too_long_query(self):
        with pytest.raises(Exception):
            QueryRequest(query="x" * 2001)

    def test_pipeline_config_top_k_bounds(self):
        with pytest.raises(Exception):
            PipelineConfig(top_k=0)
        with pytest.raises(Exception):
            PipelineConfig(top_k=101)

    def test_pipeline_config_score_threshold_bounds(self):
        with pytest.raises(Exception):
            PipelineConfig(score_threshold=-0.1)
        with pytest.raises(Exception):
            PipelineConfig(score_threshold=1.1)

    def test_query_response_serializes_sentences(self):
        data = json.loads(self.make_response().model_dump_json())
        assert data["sentences"][0]["text"] == "Dropout prevents overfitting."
        assert data["sentences"][0]["citations"][0]["doc_id"] == "1234.5678"

    def test_query_response_failed_status(self):
        resp = self.make_response(status="failed", answer="", errors=["timeout"])
        assert resp.status == "failed"
        assert "timeout" in resp.errors

    def test_sse_event_all_types(self):
        for event_type in ("decomposed", "retrieved", "graph_built", "judged", "completed", "error"):
            ev = SSEEvent(event=event_type, data={"k": "v"})
            assert ev.event == event_type

    def test_sse_event_rejects_unknown_type(self):
        with pytest.raises(Exception):
            SSEEvent(event="unknown_stage", data={})


class TestSSEFormat(Fixtures):

    def test_format_structure(self):
        out = _format_sse(SSEEvent(event="decomposed", data={"n_sub_queries": 3}))
        lines = out.split("\n")
        assert lines[0] == "event: decomposed"
        assert lines[1].startswith("data: ")
        assert lines[2] == ""
        assert lines[3] == ""

    def test_format_data_is_valid_json(self):
        out = _format_sse(SSEEvent(event="retrieved", data={"n_docs": 42, "nested": {"x": 1}}))
        data_line = next(l for l in out.split("\n") if l.startswith("data: "))
        parsed = json.loads(data_line[len("data: "):])
        assert parsed["n_docs"] == 42
        assert parsed["nested"]["x"] == 1

    def test_format_terminates_with_double_newline(self):
        assert _format_sse(SSEEvent(event="completed", data={})).endswith("\n\n")

    def test_format_each_event_type(self):
        for ev in self.make_sse_sequence():
            out = _format_sse(ev)
            assert f"event: {ev.event}" in out
            assert out.endswith("\n\n")


class TestEndpoints(Fixtures):

    def test_health_ok(self):
        client = self.make_client(self.make_runner(self.make_response()))
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["collection"] == "unarxive_chunks"
        assert r.json()["model"] == "bge-m3"

    def test_health_degraded_when_qdrant_down(self):
        client = self.make_client(self.make_runner(self.make_response()))
        with patch("api.routes.health.check_qdrant_alive", side_effect=RuntimeError("Qdrant unreachable")):
            r = client.get("/health")
        assert r.status_code == 503
        assert r.json()["status"] == "degraded"

    def test_query_returns_200_and_answer(self):
        client = self.make_client(self.make_runner(self.make_response()))
        r = client.post("/api/v1/query", json={"query": "What is dropout regularization?"})
        assert r.status_code == 200
        assert r.json()["status"] == "completed"
        assert r.json()["answer"] == "Dropout is a regularization technique."
        assert r.json()["job_id"] == "test-job-1"
        assert r.json()["elapsed_s"] == 2.4

    def test_query_returns_sentences_with_citations(self):
        client = self.make_client(self.make_runner(self.make_response()))
        r = client.post("/api/v1/query", json={"query": "What is dropout regularization?"})
        sentences = r.json()["sentences"]
        assert len(sentences) == 1
        assert sentences[0]["citations"][0]["doc_id"] == "1234.5678"

    def test_query_with_custom_config(self):
        runner = self.make_runner(self.make_response())
        client = self.make_client(runner)
        r = client.post("/api/v1/query", json={
            "query": "What is batch normalization?",
            "config": {"top_k": 20, "enable_hop": False, "score_threshold": 0.3},
        })
        assert r.status_code == 200
        runner.run_query.assert_called_once()
        req_arg: QueryRequest = runner.run_query.call_args[0][0]
        assert req_arg.config.top_k == 20
        assert req_arg.config.enable_hop is False

    def test_query_returns_failed_status_on_pipeline_error(self):
        client = self.make_client(self.make_runner(
            self.make_response(status="failed", answer="", errors=["LLM timeout"])
        ))
        r = client.post("/api/v1/query", json={"query": "What is dropout regularization?"})
        assert r.status_code == 200
        assert r.json()["status"] == "failed"
        assert "LLM timeout" in r.json()["errors"]

    def test_query_rejects_short_query(self):
        client = self.make_client(self.make_runner(self.make_response()))
        assert client.post("/api/v1/query", json={"query": "hi"}).status_code == 422

    def test_query_rejects_missing_query_field(self):
        client = self.make_client(self.make_runner(self.make_response()))
        assert client.post("/api/v1/query", json={"config": {"top_k": 5}}).status_code == 422

    def test_stream_returns_text_event_stream(self):
        runner = self.make_runner(self.make_response())
        status_code, headers, _ = asyncio.run(self.get_sse_response(runner, "What is attention in transformers?"))
        assert status_code == 200
        assert "text/event-stream" in headers["content-type"]
        assert headers["cache-control"] == "no-cache"
        assert headers["x-accel-buffering"] == "no"

    def test_stream_events_in_correct_order(self):
        runner = self.make_runner(self.make_response())
        _, _, body = asyncio.run(self.get_sse_response(runner, "What is attention in transformers?"))
        event_types = [e["event"] for e in self.parse_sse(body)]
        assert event_types == ["decomposed", "retrieved", "graph_built", "judged", "completed"]

    def test_stream_completed_event_has_answer(self):
        runner = self.make_runner(self.make_response())
        _, _, body = asyncio.run(self.get_sse_response(runner, "What is attention in transformers?"))
        completed = next(e for e in self.parse_sse(body) if e["event"] == "completed")
        assert "answer" in completed["data"]
        assert "sentences" in completed["data"]

    def test_stream_intermediate_events_have_progress_data(self):
        runner = self.make_runner(self.make_response())
        _, _, body = asyncio.run(self.get_sse_response(runner, "What is attention in transformers?"))
        events = {e["event"]: e["data"] for e in self.parse_sse(body)}
        assert "n_sub_queries" in events["decomposed"]
        assert "n_docs"        in events["retrieved"]
        assert "n_nodes"       in events["graph_built"]
        assert "n_judged"      in events["judged"]

    def test_stream_rejects_missing_q_param(self):
        client = self.make_client(self.make_runner(self.make_response()))
        assert client.get("/api/v1/query/stream").status_code == 422


class TestWorkflowRunnerStreaming:

    def test_stream_query_yields_before_workflow_finishes(self):
        release_worker = threading.Event()

        class FakeWorkflow:
            def stream(self, accumulated, stream_mode="updates"):
                assert stream_mode == "updates"
                yield {"decompose": {"sub_queries": [], "decomposition_done": True}}
                release_worker.wait(timeout=2)
                yield {"generate_answer": {"final_answer": FinalAnswer(text="Done", sentences=[], reasoning_summary=None), "answer_done": True}}

        runner = WorkflowRunner.__new__(WorkflowRunner)
        runner._workflow = FakeWorkflow()

        async def _run() -> None:
            request = QueryRequest(query="What is attention?")
            agen = runner.stream_query(request)

            first_event = await asyncio.wait_for(agen.__anext__(), timeout=0.5)
            assert first_event.event == "decomposed"

            release_worker.set()
            second_event = await asyncio.wait_for(agen.__anext__(), timeout=0.5)
            assert second_event.event == "completed"
            await agen.aclose()

        asyncio.run(_run())


class TestAppStartup:

    def test_lifespan_prewarms_nli_model(self):
        from api.main import lifespan

        app = MagicMock()

        async def _run() -> None:
            with patch("api.main.ensure_qdrant_runtime"), \
                 patch("api.main.WorkflowRunner", return_value=MagicMock()), \
                 patch("api.main.NLIModel.prewarm") as mock_prewarm:
                async with lifespan(app):
                    pass
            mock_prewarm.assert_called_once()

        asyncio.run(_run())
