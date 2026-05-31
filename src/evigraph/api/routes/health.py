from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from evigraph.utils.qdrant import check_qdrant_alive

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    runner = request.app.state.runner

    try:
        check_qdrant_alive(runner._services.retriever.client)
        qdrant_status = "ok"
    except Exception as exc:
        qdrant_status = str(exc)

    ok = qdrant_status == "ok"

    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "status":     "ok" if ok else "degraded",
            "qdrant":     qdrant_status,
            "collection": runner._services.retriever.collection_name,
            "model":      runner._model_key,
        },
    )
