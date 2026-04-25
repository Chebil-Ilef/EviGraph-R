from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.schemas import QueryRequest, QueryResponse, SSEEvent

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    runner = request.app.state.runner
    return await runner.run_query(body)


@router.get("/query/stream")
async def query_stream(request: Request, q: str) -> StreamingResponse:
    runner = request.app.state.runner
    query_request = QueryRequest(query=q)

    async def _event_generator():
        async for event in runner.stream_query(query_request):
            yield _format_sse(event)

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


def _format_sse(event: SSEEvent) -> str:
    return f"event: {event.event}\ndata: {json.dumps(event.data)}\n\n"
