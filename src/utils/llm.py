from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from config.settings import LLM


OPENAI_PROVIDER_PREFIX = "openai/"
KNOWN_DSPY_PROVIDER_PREFIXES = (
    "anthropic/",
    "azure/",
    "bedrock/",
    "databricks/",
    "deepseek/",
    "gemini/",
    "lm_studio/",
    "mistral/",
    "ollama_chat/",
    "ollama/",
    "openai/",
    "sagemaker/",
    "vertex_ai/",
    "watsonx/",
)
OPENAI_COMPATIBLE_PATH_SUFFIXES = (
    "/chat/completions",
    "/completions",
    "/responses",
)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class LLMClient:
    """
    DSPy-backed wrapper for chat-style generation.

    Model-agnostic with DSPy handling the LM.
    """

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        raw_api_base = api_base if api_base is not None else LLM.api_base
        self.api_base = self._normalize_api_base(raw_api_base)
        self.api_key = api_key if api_key is not None else LLM.api_key
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else LLM.timeout_seconds
        self.max_retries = max_retries if max_retries is not None else LLM.max_retries

    def chat(
        self,
        *,
        model: str,
        messages: Iterable[ChatMessage | dict[str, str]],
        temperature: float = 0.0,
        timeout: float | None = None,
        **extra: Any,
    ) -> str:
        
        dspy = self._import_dspy()

        # Use provided timeout or fall back to instance default
        effective_timeout = timeout if timeout is not None else self.timeout_seconds

        lm = dspy.LM(
            self._resolve_model_name(model),
            model_type="chat",
            api_base=self.api_base,
            api_key=self.api_key,
            timeout=effective_timeout,
            num_retries=self.max_retries,
            temperature=temperature,
            **extra,
        )

        response = lm(messages=[self._coerce_message(message) for message in messages])
        return self._extract_text(response).strip()

    def chat_text(
        self,
        *,
        model: str,
        system_prompt: str | None,
        user_prompt: str,
        temperature: float = 0.0,
        timeout: float | None = None,
        **extra: Any,
    ) -> str:
        
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=user_prompt))
        
        # Don't pass timeout in extra if it's already being handled explicitly
        return self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            **extra,
        )

    def _resolve_model_name(self, model: str) -> str:
        normalized_model = model.strip()
        if normalized_model.startswith(KNOWN_DSPY_PROVIDER_PREFIXES):
            return normalized_model
        if normalized_model == "<provider>/<model_name>":
            raise ValueError(
                "LLM_MODEL is still set to the placeholder '<provider>/<model_name>'. "
                "Set it to a real model name such as 'meta-llama/Llama-3.3-70B-Instruct'."
            )
        if self._is_openai_compatible_api_base(self.api_base):
            return f"{OPENAI_PROVIDER_PREFIX}{normalized_model}"

        raise ValueError(
            f"Model name '{model}' does not start with a known DSPy provider prefix. "
            f"Model names must be in the format '<provider>/<model_name>'. "
            "For OpenAI-compatible endpoints, a bare model name is also accepted when "
            "`LLM_API_BASE` points at a `/v1` API root or full `/chat/completions` URL. "
            f"Known provider prefixes are: {KNOWN_DSPY_PROVIDER_PREFIXES}"
        )

    @staticmethod
    def _coerce_message(message: ChatMessage | dict[str, str]) -> dict[str, str]:
        if isinstance(message, ChatMessage):
            return {"role": message.role, "content": message.content}
        return {"role": message["role"], "content": message["content"]}

    @staticmethod
    def _extract_text(response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            if not response:
                raise RuntimeError("DSPy returned an empty response")
            first = response[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                if "text" in first and isinstance(first["text"], str):
                    return first["text"]
                if "content" in first and isinstance(first["content"], str):
                    return first["content"]
        if hasattr(response, "text") and isinstance(response.text, str):
            return response.text
        raise RuntimeError(f"Unsupported DSPy response type: {type(response)!r}")

    @staticmethod
    def _import_dspy():
        try:
            import dspy
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "DSPy is not installed. Add the `dspy` package to the environment "
                "before using the LLM client."
            ) from exc
        return dspy

    @staticmethod
    def _normalize_api_base(api_base: str | None) -> str | None:
        if api_base is None:
            return None
        normalized = api_base.strip().rstrip("/")
        if not normalized:
            return None
        for suffix in OPENAI_COMPATIBLE_PATH_SUFFIXES:
            if normalized.endswith(suffix):
                return normalized[: -len(suffix)]
        return normalized

    @staticmethod
    def _is_openai_compatible_api_base(api_base: str | None) -> bool:
        if not api_base:
            return False
        return (
            api_base.endswith("/v1")
            or "/v1/" in api_base
            or "/openai/" in api_base
        )


def get_llm_client() -> LLMClient:
    return LLMClient()
