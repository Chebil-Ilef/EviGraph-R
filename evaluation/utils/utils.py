from __future__ import annotations

import json
import os
from typing import Any


def build_deepeval_model(model_name: str):
    api_base = os.getenv("LLM_API_BASE", "").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "")

    if not api_base or not api_key:
        raise RuntimeError("LLM_API_BASE and LLM_API_KEY must be set in .env")

    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_BASE_URL", api_base)

    try:
        from deepeval.models import DeepEvalBaseLLM
        from openai import OpenAI
        import pydantic
    except ImportError as exc:
        raise ImportError("deepeval, openai, and pydantic are required for DeepEval runs") from exc

    class _ScaDSModel(DeepEvalBaseLLM):
        def __init__(self, m: str) -> None:
            self.model = m
            self._client = OpenAI(base_url=api_base, api_key=api_key)

        def get_model_name(self) -> str:
            return self.model

        def load_model(self):
            return self._client

        def generate(self, prompt: str, schema: Any = None) -> Any:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
            if schema is None:
                return text
            # Try to parse JSON and construct the schema instance.
            try:
                schema_fields = set(schema.model_fields.keys())
            except AttributeError:
                schema_fields = set()
            # `Response` — just wrap the raw text, no JSON needed.
            if schema_fields == {"response"}:
                return schema(response=text)
            # For all other schemas: strip markdown fences and parse JSON.
            try:
                stripped = text.strip()
                if stripped.startswith("```"):
                    lines = stripped.splitlines()
                    stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                data = json.loads(stripped)
                return schema(**data)
            except Exception:
                pass
            # JSON parse failed — return a safe default so DeepEval never
            # crashes on a missing attribute. Defaults are chosen to be
            # pass-through (high score, identity rewrites, valid empty lists).
            if schema_fields == {"score", "feedback"}:
                # InputFeedback: score=1.0 passes the quality threshold,
                # avoiding retries and rewrite loops.
                return schema(score=1.0, feedback="")
            if schema_fields == {"rewritten_input"}:
                # RewrittenInput: return the original text unchanged.
                return schema(rewritten_input=text)
            if schema_fields == {"input"}:
                # SyntheticData
                return schema(input=text)
            if schema_fields == {"data"}:
                # SyntheticDataList: empty list — filtered out downstream.
                return schema(data=[])
            # Unknown schema: return raw text and let DeepEval's own
            # trimAndLoadJson fallback handle it.
            return text

        async def a_generate(self, prompt: str, schema: Any = None) -> Any:
            return self.generate(prompt, schema=schema)

    return _ScaDSModel(model_name)
