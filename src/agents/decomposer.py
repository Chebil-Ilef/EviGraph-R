from __future__ import annotations
import json
import logging
from typing import Any
from config.prompts import DECOMPOSER_SYSTEM_PROMPT, build_decomposer_user_prompt
from config.settings import AGENT_MODELS
from schemas.objects import DecompositionResult, SubQuery, IMRaDSection
from utils.llm import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


class DecomposerAgent:

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.config = AGENT_MODELS["decomposer"]

    def decompose(self, query: str) -> list[SubQuery]:

        normalized_query = query.strip()
        if not normalized_query:
            return []

        try:
            result = self._generate_decomposition(normalized_query)
        except Exception as exc:
            logger.warning("[DECOMPOSER] Decomposition failed, falling back to original query: %s", exc)
            return [self._create_fallback_subquery(normalized_query)]

        if not result.should_decompose or not result.sub_queries:
            logger.info("[DECOMPOSER AGENT] Query does not benefit from decomposition")
            return [self._create_fallback_subquery(normalized_query)]

        # Normalize budget weights to sum to 1.0
        sub_queries = self._normalize_budget_weights(result.sub_queries)

        if not sub_queries:
            logger.warning("[DECOMPOSER AGENT] Decomposition produced no valid sub-queries, falling back to original query")
            return [self._create_fallback_subquery(normalized_query)]

        logger.info("[DECOMPOSER AGENT] Decomposition successful: %d sub-queries", len(sub_queries))

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

        if "sub_queries" in payload:
            for sq in payload["sub_queries"]:
                if "sections" in sq:
                    sq["sections"] = self._parse_sections(sq["sections"])

        result = DecompositionResult.model_validate(payload)

        if not result.should_decompose or not result.sub_queries:
            return DecompositionResult(
                should_decompose=False,
                sub_queries=[self._create_fallback_subquery(query)],
            )

        return result

    @staticmethod
    def _parse_sections(sections: list[str]) -> list[IMRaDSection]:

        parsed = []
        for section in sections:
            try:
                # Try exact match first
                parsed.append(IMRaDSection(section))
            except ValueError:
                # Try case-insensitive match
                section_upper = section.strip().upper().replace(" ", "_")
                for imrad_section in IMRaDSection:
                    if imrad_section.name == section_upper:
                        parsed.append(imrad_section)
                        break
                else:
                    logger.warning("[DECOMPOSER AGENT] Invalid IMRaD section: %s, skipping", section)
        return parsed

    @staticmethod
    def _normalize_budget_weights(sub_queries: list[SubQuery]) -> list[SubQuery]:

        if not sub_queries:
            return []

        # Filter valid sub-queries
        valid_queries = [sq for sq in sub_queries if sq.text.strip()]
        if not valid_queries:
            return []

        # Calculate total weight
        total_weight = sum(sq.budget_weight for sq in valid_queries)

        if total_weight <= 0:
            # Equal distribution if all weights are 0
            weight = 1.0 / len(valid_queries)
            for sq in valid_queries:
                sq.budget_weight = weight
        else:
            # Normalize to sum to 1.0
            for sq in valid_queries:
                sq.budget_weight = sq.budget_weight / total_weight

        return valid_queries

    @staticmethod
    def _create_fallback_subquery(query: str) -> SubQuery:

        logger.info("[DECOMPOSER AGENT] Creating fallback sub-query for original query")
        return SubQuery(
            text=query,
            sections=[IMRaDSection.ABSTRACT, IMRaDSection.INTRODUCTION],
            budget_weight=1.0,
        )

    @staticmethod
    def _parse_json_object(raw_response: str) -> dict[str, Any]:

        response = raw_response.strip()

        # Strip markdown code fences if present
        if response.startswith("```"):
            response = response.strip("`")
            if response.startswith("json"):
                response = response[4:].strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}")
            if start == -1 or end == -1 or start >= end:
                raise ValueError("Model did not return a JSON object")
            logger.warning("[DECOMPOSER AGENT] Model response is not valid JSON, attempting to extract JSON object from response")
            return json.loads(response[start : end + 1])
