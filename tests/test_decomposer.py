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
        mock_llm_client.chat_text.return_value = MOCK_RESPONSE_SIMPLE

        result = decomposer.decompose("What is reinforcement learning?")

        assert len(result) == 1, "Simple query should return 1 sub-query"
        assert result[0].text == "What is reinforcement learning?"
        assert IMRaDSection.ABSTRACT in result[0].sections
        assert IMRaDSection.INTRODUCTION in result[0].sections
        assert result[0].budget_weight == 1.0

    def test_decompose_complex_query_with_split(self, decomposer, mock_llm_client):
        mock_llm_client.chat_text.return_value = MOCK_RESPONSE_DECOMPOSED

        result = decomposer.decompose(
            "What is the difference between dense and sparse retrieval and which is better for RAG?"
        )

        assert len(result) == 3, "Complex query should be split into 3 sub-queries"
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
        mock_llm_client.chat_text.return_value = MOCK_RESPONSE_SIMPLE

        result = decomposer.decompose("What is reinforcement learning?")

        assert IMRaDSection.ABSTRACT in result[0].sections
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
                "sections": ["Results", "Experiments"],
                "budget_weight": 1.0
            }]
        })
        mock_llm_client.chat_text.return_value = mock_response

        result = decomposer.decompose("What is the performance of BERT?")

        assert IMRaDSection.RESULTS in result[0].sections
        assert IMRaDSection.EXPERIMENTS in result[0].sections

    def test_invalid_section_names_are_filtered(self, decomposer, mock_llm_client):
        mock_llm_client.chat_text.return_value = MOCK_RESPONSE_INVALID_SECTIONS

        result = decomposer.decompose("What is quantum computing?")

        # Should have 2 valid sections from first sub-query
        valid_sections = [s for s in result[0].sections if isinstance(s, IMRaDSection)]
        assert len(valid_sections) == 2  # Abstract and Methods
        assert IMRaDSection.ABSTRACT in result[0].sections
        assert IMRaDSection.METHODS in result[0].sections


    # Budget Weight Tests

    def test_budget_weights_sum_to_one(self, decomposer, mock_llm_client):
        mock_llm_client.chat_text.return_value = MOCK_RESPONSE_DECOMPOSED

        result = decomposer.decompose("What is dense vs sparse retrieval?")

        total_weight = sum(sq.budget_weight for sq in result)
        assert abs(total_weight - 1.0) < 0.001, f"Budget weights should sum to 1.0, got {total_weight}"

    def test_unnormalized_weights_are_normalized(self, decomposer, mock_llm_client):
        mock_response = json.dumps({
            "should_decompose": True,
            "sub_queries": [
                {
                    "text": "What are transformers?",
                    "sections": ["Abstract"],
                    "budget_weight": 0.3
                },
                {
                    "text": "What are RNNs?",
                    "sections": ["Abstract"],
                    "budget_weight": 0.3
                }
            ]
        })
        mock_llm_client.chat_text.return_value = mock_response

        result = decomposer.decompose("Compare transformers and RNNs")

        total_weight = sum(sq.budget_weight for sq in result)
        assert abs(total_weight - 1.0) < 0.001
        # 0.3 / 0.6 = 0.5, 0.3 / 0.6 = 0.5
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
        assert IMRaDSection.ABSTRACT in result[0].sections
        assert IMRaDSection.INTRODUCTION in result[0].sections
        assert result[0].budget_weight == 1.0

    def test_invalid_json_returns_fallback_query(self, decomposer, mock_llm_client):
        mock_llm_client.chat_text.return_value = "This is not JSON at all!"

        result = decomposer.decompose("What is quantum computing?")

        assert len(result) == 1
        assert result[0].text == "What is quantum computing?"

    def test_should_decompose_false_with_empty_sub_queries(self, decomposer, mock_llm_client):
        mock_response = json.dumps({
            "should_decompose": False,
            "sub_queries": []
        })
        mock_llm_client.chat_text.return_value = mock_response

        result = decomposer.decompose("What is quantum computing?")

        assert len(result) == 1
        assert result[0].text == "What is quantum computing?"

    def test_empty_sub_query_text_is_filtered(self, decomposer, mock_llm_client):
        mock_response = json.dumps({
            "should_decompose": True,
            "sub_queries": [
                {
                    "text": "What is quantum computing?",
                    "sections": ["Abstract"],
                    "budget_weight": 0.5
                },
                {
                    "text": "",
                    "sections": ["Methods"],
                    "budget_weight": 0.5
                }
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

        # Verify structure
        assert len(result) == 3

        # Verify first sub-query
        assert result[0].text == "What is dense retrieval?"
        assert IMRaDSection.ABSTRACT in result[0].sections
        assert IMRaDSection.METHODS in result[0].sections

        # Verify second sub-query
        assert result[1].text == "What is sparse retrieval?"
        assert IMRaDSection.ABSTRACT in result[1].sections
        assert IMRaDSection.METHODS in result[1].sections

        # Verify third sub-query (evaluation)
        assert result[2].text == "Which is better for RAG?"
        assert IMRaDSection.RESULTS in result[2].sections
        assert IMRaDSection.EXPERIMENTS in result[2].sections
        assert IMRaDSection.DISCUSSION in result[2].sections

        # Verify budget allocation
        total_weight = sum(sq.budget_weight for sq in result)
        assert abs(total_weight - 1.0) < 0.001
        # First two get equal weights, third gets slightly more
        assert result[0].budget_weight == 0.4
        assert result[1].budget_weight == 0.3
        assert result[2].budget_weight == 0.3


    # Helper Method Tests

    def test_parse_sections_with_valid_names(self, decomposer):
        sections = ["Abstract", "Methods", "Results"]

        result = decomposer._parse_sections(sections)

        assert len(result) == 3
        assert IMRaDSection.ABSTRACT in result
        assert IMRaDSection.METHODS in result
        assert IMRaDSection.RESULTS in result

    def test_parse_sections_with_mixed_case(self, decomposer):
        sections = ["abstract", "METHODS", "Results"]

        result = decomposer._parse_sections(sections)

        assert len(result) == 3
        assert IMRaDSection.ABSTRACT in result
        assert IMRaDSection.METHODS in result
        assert IMRaDSection.RESULTS in result

    def test_parse_sections_filters_invalid(self, decomposer):
        sections = ["Abstract", "InvalidSection", "Methods", "AnotherBadOne"]

        result = decomposer._parse_sections(sections)

        assert len(result) == 2
        assert IMRaDSection.ABSTRACT in result
        assert IMRaDSection.METHODS in result

    def test_normalize_budget_weights_empty_list(self, decomposer):

        result = decomposer._normalize_budget_weights([])

        assert result == []

    def test_normalize_budget_weights_zero_total(self, decomposer):
        sub_queries = [
            SubQuery(text="Query 1", sections=[], budget_weight=0.0),
            SubQuery(text="Query 2", sections=[], budget_weight=0.0),
        ]

        result = decomposer._normalize_budget_weights(sub_queries)

        # Should distribute equally
        assert len(result) == 2
        assert result[0].budget_weight == 0.5
        assert result[1].budget_weight == 0.5

    def test_create_fallback_subquery(self, decomposer):
        query = "What is quantum computing?"

        result = decomposer._create_fallback_subquery(query)

        assert result.text == query
        assert IMRaDSection.ABSTRACT in result.sections
        assert IMRaDSection.INTRODUCTION in result.sections
        assert result.budget_weight == 1.0
