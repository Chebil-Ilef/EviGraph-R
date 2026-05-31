from __future__ import annotations
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from evigraph.config.settings import (
    AGENT_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    GRAPH_CONFIG,
    LLM,
    RERANKER,
    QDRANT_CONNECTION,
    QDRANT_ACTIVE,
    _ENV_PROFILE,
)

router = APIRouter()


@router.get("/config")
async def get_config(request: Request) -> JSONResponse:
    runner = request.app.state.runner

    return JSONResponse({
        "embedding": {
            "model":           runner._model_key,
            "default":         DEFAULT_EMBEDDING_MODEL,
        },
        "reranker": {
            "model":           RERANKER.model_id,
            "enabled":         RERANKER.enabled,
            "top_n":           RERANKER.top_n,
            "min_score":       RERANKER.min_score_threshold,
            "device":          RERANKER.device,
        },
        "llm": {
            "api_base":        LLM.api_base,
            "api_key_set":     bool(LLM.api_key),
            "agents": {
                agent: {
                    "model":           cfg.model,
                    "temperature":     cfg.temperature,
                    "timeout_s":       cfg.timeout_seconds,
                    "max_retries":     cfg.max_retries,
                    "max_tokens":      cfg.max_tokens,
                }
                for agent, cfg in AGENT_MODELS.items()
            },
        },
        "graph": {
            "nli_model":                   GRAPH_CONFIG.nli_model_id,
            "nli_threshold":               GRAPH_CONFIG.nli_threshold,
            "nli_contradiction_threshold": GRAPH_CONFIG.nli_contradiction_threshold,
            "hop_max_per_build":           GRAPH_CONFIG.hop_max_per_build,
            "hop_max_chunks_per_claim":    GRAPH_CONFIG.hop_max_chunks_per_claim,
            "max_claims_per_chunk":        GRAPH_CONFIG.max_claims_per_chunk,
            "claim_extraction_batch_size": GRAPH_CONFIG.claim_extraction_batch_size,
        },
        "qdrant": {
            "profile":    _ENV_PROFILE,
            "host":       QDRANT_CONNECTION.host,
            "port":       QDRANT_CONNECTION.port,
            "url":        QDRANT_CONNECTION.url,
            "collection": runner._services.retriever.collection_name,
            "hnsw_m":     QDRANT_ACTIVE.hnsw.m,
            "quantized":  QDRANT_ACTIVE.quantize,
        },
    })
