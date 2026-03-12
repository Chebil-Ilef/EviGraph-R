from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from config.settings import LLM


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
        self.api_base = api_base if api_base is not None else LLM.api_base
        self.api_key = api_key if api_key is not None else LLM.api_key
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else LLM.timeout_seconds
        self.max_retries = max_retries if max_retries is not None else LLM.max_retries

    def chat(
        self,
        *,
        model: str,
        messages: Iterable[ChatMessage | dict[str, str]],
        temperature: float = 0.0,
        **extra: Any,
    ) -> str:
        
        dspy = self._import_dspy()

        lm = dspy.LM(
            self._resolve_model_name(model),
            model_type="chat",
            api_base=self.api_base,
            api_key=self.api_key,
            timeout_s=self.timeout_seconds,
            num_retries=self.max_retries,
            temperature=temperature
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
        **extra: Any,
    ) -> str:
        
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=user_prompt))
        
        return self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            **extra,
        )

    def _resolve_model_name(self, model: str) -> str:
        if model.startswith(KNOWN_DSPY_PROVIDER_PREFIXES):
            return model
        else:
            return model

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


def get_llm_client() -> LLMClient:
    return LLMClient()
