from __future__ import annotations
import json
import sys
from pathlib import Path
from unittest import mock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.decomposer import DecomposerAgent
from schemas.objects import DecompositionResult, SubQuery, IMRaDSection


# Mock LLM Responses

MOCK_RESPONSE_SIMPLE = json.dumps({
    "should_decompose": False,
    "sub_queries": [
        {
            "text": "What is reinforcement learning?",
            "sections": ["Abstract", "Introduction"],
            "budget_weight": 1.0
        }
    ]
})

MOCK_RESPONSE_DECOMPOSED = json.dumps({
    "should_decompose": True,
    "sub_queries": [
        {
            "text": "What is dense retrieval?",
            "sections": ["Abstract", "Methods"],
            "budget_weight": 0.4
        },
        {
            "text": "What is sparse retrieval?",
            "sections": ["Abstract", "Methods"],
            "budget_weight": 0.3
        },
        {
            "text": "Which is better for RAG?",
            "sections": ["Results", "Experiments", "Discussion"],
            "budget_weight": 0.3
        }
    ]
})

MOCK_RESPONSE_WITH_MARKDOWN = f"""```json
{MOCK_RESPONSE_SIMPLE}
```"""

MOCK_RESPONSE_INVALID_SECTIONS = json.dumps({
    "should_decompose": True,
    "sub_queries": [
        {
            "text": "What is quantum computing?",
            "sections": ["Abstract", "InvalidSection", "Methods"],
            "budget_weight": 0.5
        },
        {
            "text": "How is it used in cryptography?",
            "sections": ["Methods"],
            "budget_weight": 0.5
        }
    ]
})


class TestDecomposerAgent:

    @pytest.fixture
    def mock_llm_client(self):
        client = mock.MagicMock()
        return client

    @pytest.fixture
    def decomposer(self, mock_llm_client):
        return DecomposerAgent(llm_client=mock_llm_client)


    # Basic Decomposition Tests

    def test_decompose_simple_query_no_split(self, decomposer, mock_llm_client):
        # should_decompose=false → fallback sub-query is returned
        mock_llm_client.chat_text.return_value = MOCK_RESPONSE_SIMPLE

        result = decomposer.decompose("What is reinforcement learning?")

        assert len(result) == 1
        assert result[0].text == "What is reinforcement learning?"
        assert IMRaDSection.INTRODUCTION in result[0].sections
        assert result[0].budget_weight == 1.0

    def test_decompose_complex_query_with_split(self, decomposer, mock_llm_client):
        mock_llm_client.chat_text.return_value = MOCK_RESPONSE_DECOMPOSED

        result = decomposer.decompose(
            "What is the difference between dense and sparse retrieval and which is better for RAG?"
        )

        assert len(result) == 3
        assert result[0].text == "What is dense retrieval?"
        assert result[1].text == "What is sparse retrieval?"
        assert result[2].text == "Which is better for RAG?"

    def test_decompose_handles_markdown_wrapped_json(self, decomposer, mock_llm_client):
        mock_llm_client.chat_text.return_value = MOCK_RESPONSE_WITH_MARKDOWN

        result = decomposer.decompose("What is reinforcement learning?")

        assert len(result) == 1
        assert result[0].text == "What is reinforcement learning?"


    # IMRaD Section Mapping Tests

    def test_section_mapping_for_concept_queries(self, decomposer, mock_llm_client):
        # should_decompose=false → fallback, which uses INTRODUCTION
        mock_llm_client.chat_text.return_value = MOCK_RESPONSE_SIMPLE

        result = decomposer.decompose("What is reinforcement learning?")

        assert IMRaDSection.INTRODUCTION in result[0].sections

    def test_section_mapping_for_technical_queries(self, decomposer, mock_llm_client):
        mock_response = json.dumps({
            "should_decompose": True,
            "sub_queries": [{
                "text": "How does attention work?",
                "sections": ["Abstract", "Methods"],
                "budget_weight": 1.0
            }]
        })
        mock_llm_client.chat_text.return_value = mock_response

        result = decomposer.decompose("How does attention work?")

        assert IMRaDSection.METHODS in result[0].sections

    def test_section_mapping_for_evaluation_queries(self, decomposer, mock_llm_client):
        mock_response = json.dumps({
            "should_decompose": True,
            "sub_queries": [{
                "text": "What is the performance of BERT?",
                "sections": ["Results"],
                "budget_weight": 1.0
            }]
        })
        mock_llm_client.chat_text.return_value = mock_response

        result = decomposer.decompose("What is the performance of BERT?")

        assert IMRaDSection.RESULTS in result[0].sections


    # Budget Weight Tests

    def test_budget_weights_sum_to_one(self, decomposer, mock_llm_client):
        mock_llm_client.chat_text.return_value = MOCK_RESPONSE_DECOMPOSED

        result = decomposer.decompose("What is dense vs sparse retrieval?")

        total_weight = sum(sq.budget_weight for sq in result)
        assert abs(total_weight - 1.0) < 0.001

    def test_unnormalized_weights_are_normalized(self, decomposer, mock_llm_client):
        mock_response = json.dumps({
            "should_decompose": True,
            "sub_queries": [
                {"text": "What are transformers?", "sections": ["Abstract"], "budget_weight": 0.3},
                {"text": "What are RNNs?", "sections": ["Abstract"], "budget_weight": 0.3}
            ]
        })
        mock_llm_client.chat_text.return_value = mock_response

        result = decomposer.decompose("Compare transformers and RNNs")

        total_weight = sum(sq.budget_weight for sq in result)
        assert abs(total_weight - 1.0) < 0.001
        assert abs(result[0].budget_weight - 0.5) < 0.001
        assert abs(result[1].budget_weight - 0.5) < 0.001

    def test_single_query_gets_full_budget(self, decomposer, mock_llm_client):
        mock_llm_client.chat_text.return_value = MOCK_RESPONSE_SIMPLE

        result = decomposer.decompose("What is reinforcement learning?")

        assert result[0].budget_weight == 1.0


    # Edge Cases

    def test_empty_query_returns_empty_list(self, decomposer, mock_llm_client):
        result = decomposer.decompose("")

        assert result == []
        mock_llm_client.chat_text.assert_not_called()

    def test_whitespace_only_query_returns_empty_list(self, decomposer, mock_llm_client):
        result = decomposer.decompose("   \n\t  ")

        assert result == []
        mock_llm_client.chat_text.assert_not_called()

    def test_llm_failure_returns_fallback_query(self, decomposer, mock_llm_client):
        mock_llm_client.chat_text.side_effect = Exception("LLM API error")

        result = decomposer.decompose("What is quantum computing?")

        assert len(result) == 1
        assert result[0].text == "What is quantum computing?"
        assert IMRaDSection.INTRODUCTION in result[0].sections
        assert result[0].budget_weight == 1.0

    def test_invalid_json_returns_fallback_query(self, decomposer, mock_llm_client):
        mock_llm_client.chat_text.return_value = "This is not JSON at all!"

        result = decomposer.decompose("What is quantum computing?")

        assert len(result) == 1
        assert result[0].text == "What is quantum computing?"

    def test_should_decompose_false_with_empty_sub_queries(self, decomposer, mock_llm_client):
        mock_response = json.dumps({"should_decompose": False, "sub_queries": []})
        mock_llm_client.chat_text.return_value = mock_response

        result = decomposer.decompose("What is quantum computing?")

        assert len(result) == 1
        assert result[0].text == "What is quantum computing?"

    def test_empty_sub_query_text_is_filtered(self, decomposer, mock_llm_client):
        mock_response = json.dumps({
            "should_decompose": True,
            "sub_queries": [
                {"text": "What is quantum computing?", "sections": ["Abstract"], "budget_weight": 0.5},
                {"text": "", "sections": ["Methods"], "budget_weight": 0.5}
            ]
        })
        mock_llm_client.chat_text.return_value = mock_response

        result = decomposer.decompose("What is quantum computing?")

        assert len(result) == 1
        assert result[0].text == "What is quantum computing?"
        assert result[0].budget_weight == 1.0


    # Integration-Style Tests

    def test_full_pipeline_three_way_comparison(self, decomposer, mock_llm_client):
        mock_llm_client.chat_text.return_value = MOCK_RESPONSE_DECOMPOSED

        result = decomposer.decompose(
            "What is the difference between dense and sparse retrieval and which is better for RAG?"
        )

        assert len(result) == 3

        assert result[0].text == "What is dense retrieval?"
        assert "Abstract" in result[0].sections
        assert IMRaDSection.METHODS in result[0].sections

        assert result[1].text == "What is sparse retrieval?"
        assert "Abstract" in result[1].sections
        assert IMRaDSection.METHODS in result[1].sections

        assert result[2].text == "Which is better for RAG?"
        assert IMRaDSection.RESULTS in result[2].sections
        assert IMRaDSection.DISCUSSION in result[2].sections

        total_weight = sum(sq.budget_weight for sq in result)
        assert abs(total_weight - 1.0) < 0.001
        assert result[0].budget_weight == 0.4
        assert result[1].budget_weight == 0.3
        assert result[2].budget_weight == 0.3


    # Helper Method Tests

    def test_parse_sections_abstract_passes_through_as_string(self, decomposer):
        result = decomposer._parse_sections(["Abstract", "Methods"])

        assert "Abstract" in result
        assert IMRaDSection.METHODS in result

    def test_parse_sections_abstract_case_insensitive(self, decomposer):
        result = decomposer._parse_sections(["abstract", "ABSTRACT"])

        assert result.count("Abstract") == 2

    def test_parse_sections_with_valid_imrad_names(self, decomposer):
        result = decomposer._parse_sections(["Introduction", "Methods", "Results", "Discussion"])

        assert IMRaDSection.INTRODUCTION in result
        assert IMRaDSection.METHODS in result
        assert IMRaDSection.RESULTS in result
        assert IMRaDSection.DISCUSSION in result

    def test_parse_sections_with_mixed_case(self, decomposer):
        result = decomposer._parse_sections(["introduction", "METHODS", "Results"])

        assert IMRaDSection.INTRODUCTION in result
        assert IMRaDSection.METHODS in result
        assert IMRaDSection.RESULTS in result

    def test_parse_sections_filters_invalid(self, decomposer):
        result = decomposer._parse_sections(["Abstract", "InvalidSection", "Methods", "AnotherBadOne"])

        assert "Abstract" in result
        assert IMRaDSection.METHODS in result
        assert len(result) == 2

    def test_normalize_budget_weights_empty_list(self, decomposer):
        assert decomposer._normalize_budget_weights([]) == []

    def test_normalize_budget_weights_zero_total(self, decomposer):
        sub_queries = [
            SubQuery(text="Query 1", sections=[], budget_weight=0.0),
            SubQuery(text="Query 2", sections=[], budget_weight=0.0),
        ]
        result = decomposer._normalize_budget_weights(sub_queries)

        assert result[0].budget_weight == 0.5
        assert result[1].budget_weight == 0.5

    def test_create_fallback_subquery(self, decomposer):
        result = decomposer._create_fallback_subquery("What is quantum computing?")

        assert result.text == "What is quantum computing?"
        assert IMRaDSection.INTRODUCTION in result.sections
        assert result.budget_weight == 1.0
