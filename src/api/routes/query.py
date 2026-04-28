from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.schemas import PipelineConfig, QueryRequest, QueryResponse, SSEEvent

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    runner = request.app.state.runner
    return await runner.run_query(body)


@router.get("/query/stream")
async def query_stream(
    request: Request,
    q: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    enable_hop: Optional[bool] = None,
    embedding_model: Optional[str] = None,
) -> StreamingResponse:
    runner = request.app.state.runner

    config_kwargs: dict = {}
    if top_k is not None:
        config_kwargs["top_k"] = top_k
    if score_threshold is not None:
        config_kwargs["score_threshold"] = score_threshold
    if enable_hop is not None:
        config_kwargs["enable_hop"] = enable_hop
    if embedding_model is not None:
        config_kwargs["embedding_model"] = embedding_model

    query_request = QueryRequest(
        query=q,
        config=PipelineConfig(**config_kwargs),
    )

    async def _event_generator():
        async for event in runner.stream_query(query_request):
            yield _format_sse(event)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _format_sse(event: SSEEvent) -> str:
    return f"event: {event.event}\ndata: {json.dumps(event.data)}\n\n"
