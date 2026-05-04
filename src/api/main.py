from __future__ import annotations
import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.config import router as config_router
from api.routes.graph import router as graph_router
from api.routes.health import router as health_router
from api.routes.query import router as query_router
from api.runner import WorkflowRunner
from utils.nli import NLIModel
from utils.qdrant import ensure_qdrant_runtime

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s  %(name)s  %(message)s",
)
for _noisy in ("httpcore", "httpx", "urllib3", "transformers",
               "sentence_transformers", "huggingface_hub", "langsmith", "langchain"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

def _configure_file_logging() -> str | None:
    preferred_dir = os.getenv("LOG_DIR", "/app/logs")
    fallback_dir = os.path.join(os.getcwd(), "logs")

    for candidate in (preferred_dir, fallback_dir):
        try:
            os.makedirs(candidate, exist_ok=True)
            log_path = os.path.join(candidate, "debug_logs.txt")
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                datefmt="%H:%M:%S",
            ))
            logging.getLogger().addHandler(file_handler)
            return log_path
        except OSError:
            continue
    return None


_log_path = _configure_file_logging()

logger = logging.getLogger(__name__)
if _log_path:
    logger.info("File logging active → %s", _log_path)
else:
    logger.warning("File logging disabled: no writable log directory found.")

_CORS_ORIGINS = [o.strip() for o in os.getenv("API_CORS_ORIGINS", "*").split(",")]


@asynccontextmanager
async def lifespan(app: FastAPI):
    profile = os.getenv("INDEXING_PROFILE", "local")
    model_key = os.getenv("EMBEDDING_MODEL", "bge-m3")
    logger.info("Ensuring Qdrant is running (profile=%s) ...", profile)
    ensure_qdrant_runtime(profile)
    logger.info("Qdrant ready. Loading services (model=%s) ...", model_key)
    app.state.runner = WorkflowRunner(model_key=model_key)
    try:
        logger.info("Prewarming NLI model (%s) ...", os.getenv("NLI_MODEL_ID", "sileod/deberta-v3-small-tasksource-nli"))
        NLIModel.prewarm()
        logger.info("NLI model ready.")
    except Exception as exc:
        logger.warning("NLI model prewarm failed; first NLI-backed query may cold-start: %s", exc)
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
app.include_router(graph_router, prefix="/api/v1")
app.include_router(config_router, prefix="/api/v1")
