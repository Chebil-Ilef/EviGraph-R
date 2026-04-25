from __future__ import annotations
import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.config import router as config_router
from api.routes.health import router as health_router
from api.routes.query import router as query_router
from api.runner import WorkflowRunner

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s  %(name)s  %(message)s",
)
for _noisy in ("httpcore", "httpx", "urllib3", "transformers",
               "sentence_transformers", "huggingface_hub", "langsmith", "langchain"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_CORS_ORIGINS = [o.strip() for o in os.getenv("API_CORS_ORIGINS", "*").split(",")]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_key = os.getenv("EMBEDDING_MODEL", "bge-m3")
    logger.info("Starting up — loading services (model=%s) ...", model_key)
    app.state.runner = WorkflowRunner(model_key=model_key)
    logger.info("Services ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="EviGraph-R API",
    version="1.0.0",
    description="Evidence graph reasoning over scientific literature.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(query_router, prefix="/api/v1")
app.include_router(config_router, prefix="/api/v1")
