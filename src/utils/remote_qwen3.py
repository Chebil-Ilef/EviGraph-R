from __future__ import annotations

import json
import logging
import os
import time
from typing import Iterable
from urllib import error, request
import numpy as np
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import EMBEDDING_MODELS
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


REMOTE_QWEN3_MODEL_KEY = "qwen3-4b-remote"
DEFAULT_SCADS_API_BASE = os.getenv("EMBEDDINGS_API_BASE")
DEFAULT_SCADS_API_KEY = os.getenv("EMBEDDINGS_API_KEY")
DEFAULT_MAX_RETRIES = int(os.getenv("EMBEDDINGS_MAX_RETRIES", "5"))
DEFAULT_BACKOFF_SECONDS = float(os.getenv("EMBEDDINGS_BACKOFF_SECONDS", "1.0"))


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _batch(values: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


class RemoteQwen3Embedder:
    """Thin client for the SCADS-hosted OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        *,
        api_base: str = DEFAULT_SCADS_API_BASE,
        api_key: str | None = DEFAULT_SCADS_API_KEY,
        timeout_seconds: int = 200,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    ) -> None:
        self.cfg = EMBEDDING_MODELS[REMOTE_QWEN3_MODEL_KEY]
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.embeddings_url = f"{self.api_base}"

    def embed_query(self, text: str) -> np.ndarray:
        prepared = self._apply_query_prefix(text)
        vectors = self._embed_texts([prepared])
        return vectors[0]

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.cfg.dim), dtype=np.float32)
        return self._embed_texts(texts)

    def _apply_query_prefix(self, text: str) -> str:
        instruction = self.cfg.qwen_task_instruction
        if instruction:
            return f"Instruct: {instruction}\nQuery: {text}"
        return text

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for chunk in _batch(texts, self.cfg.batch_size):
            payload = {
                "model": self.cfg.hf_model_id,
                "input": chunk,
            }
            response = self._post(payload)
            data = sorted(response["data"], key=lambda item: item["index"])
            vecs = np.asarray([item["embedding"] for item in data], dtype=np.float32)
            if self.cfg.normalize:
                vecs = _l2_normalize(vecs)
            batches.append(vecs)

        return np.vstack(batches)

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            print(f"Making request to {self.embeddings_url} (attempt {attempt}/{self.max_retries})")
            req = request.Request(self.embeddings_url, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                logger.error("SCADS embeddings request failed: %s %s", exc.code, detail)
                raise RuntimeError(
                    f"SCADS embeddings request failed with HTTP {exc.code}: {detail}"
                ) from exc
            except error.URLError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_seconds = self.backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "SCADS request attempt %d/%d failed: %s. Retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc.reason,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        reason = getattr(last_error, "reason", last_error)
        raise RuntimeError(
            f"SCADS embeddings endpoint is unreachable at {self.embeddings_url} "
            f"after {self.max_retries} attempts: {reason}"
        ) from last_error


# smoke CLI test

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embedder = RemoteQwen3Embedder()
    query_vec = embedder.embed_query("What is the capital of France?")
    passage_vecs = embedder.embed_passages(
        ["Paris is the capital of France.", "Berlin is the capital of Germany."]
    )
    print("Query vector shape:", query_vec.shape)
    print("Passage vectors shape:", passage_vecs.shape)
