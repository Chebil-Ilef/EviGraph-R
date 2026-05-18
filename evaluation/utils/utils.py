from __future__ import annotations

import os


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
    except ImportError as exc:
        raise ImportError("deepeval and openai are required for DeepEval runs") from exc

    class _ScaDSModel(DeepEvalBaseLLM):
        def __init__(self, m: str) -> None:
            self.model = m
            self._client = OpenAI(base_url=api_base, api_key=api_key)

        def get_model_name(self) -> str:
            return self.model

        def load_model(self):
            return self._client

        def generate(self, prompt: str) -> str:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""

        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)

    return _ScaDSModel(model_name)
