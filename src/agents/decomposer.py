from __future__ import annotations

import json
import logging
from typing import Any

from config.prompts import DECOMPOSER_SYSTEM_PROMPT, build_decomposer_user_prompt
from config.settings import AGENT_MODELS
from schemas.objects import DecompositionResult, SubQuery
from utils.llm import LLMClient, get_llm_client


logger = logging.getLogger(__name__)


class DecomposerAgent:
    """
    LLM-based query decomposer for multi-hop retrieval.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.config = AGENT_MODELS["decomposer"]

    def decompose(self, query: str) -> list[str]:

        normalized_query = query.strip()
        if not normalized_query:
            return []

        try:
            result = self._generate_decomposition(normalized_query)
        except Exception as exc:
            logger.warning("Decomposition failed, falling back to original query: %s", exc)
            return [normalized_query]

        sub_queries = [item.text.strip() for item in result.sub_queries if item.text.strip()]
        if not sub_queries:
            logger.warning("Decomposition produced no valid sub-questions, falling back to original query")
            return [normalized_query]
        
        return sub_queries

    def _generate_decomposition(self, query: str) -> DecompositionResult:
        
        raw_response = self.llm_client.chat_text(
            model=self.config.model,
            system_prompt=DECOMPOSER_SYSTEM_PROMPT,
            user_prompt=build_decomposer_user_prompt(query),
            temperature=self.config.temperature,
            timeout=self.config.timeout_seconds
        )

        payload = self._parse_json_object(raw_response)
        result = DecompositionResult.model_validate(payload)

        if not result.should_decompose:
            return DecompositionResult(
                should_decompose=False,
                sub_queries=[SubQuery(text=query)],
            )

        if not result.sub_queries:
            return DecompositionResult(
                should_decompose=False,
                sub_queries=[SubQuery(text=query)],
            )

        return result

    @staticmethod
    def _parse_json_object(raw_response: str) -> dict[str, Any]:
        response = raw_response.strip()
        if response.startswith("```"):
            response = response.strip("`")
            if response.startswith("json"):
                response = response[4:].strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            start = response.find("{")
            end = response.rfind("}")
            if start == -1 or end == -1 or start >= end:
                raise ValueError("Model did not return a JSON object")
            return json.loads(response[start : end + 1])
