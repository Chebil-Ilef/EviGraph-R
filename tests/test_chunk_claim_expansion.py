from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.graph import _extract_ids_from_citation_raw, resolve_cited_paper_id
from retrieval.retriever import HybridQueryRetriever
from schemas.objects import ClaimRelevance
from config.prompts import (
    build_claim_extraction_prompt,
    build_claim_extraction_prompt_batch,
    EVIDENCE_GRAPH_SYSTEM_PROMPT,
)
from agents.evidence_graph_builder import EvidenceGraphBuilderAgent
from schemas.objects import RetrievedDocument


class TestExtractIdsFromCitationRaw:

    def test_arxiv_colon_format(self):
        raw = 'S. Ioffe and C. Szegedy, "Batch normalization," in arXiv:1502.03167, 2015.'
        arxiv, doi = _extract_ids_from_citation_raw(raw)
        assert arxiv == "1502.03167"
        assert doi == "10.48550/arxiv.1502.03167"

    def test_arxiv_url_abs_format(self):
        raw = 'Brown et al., "Language Models are Few-Shot Learners," https://arxiv.org/abs/2005.14165, 2020.'
        arxiv, doi = _extract_ids_from_citation_raw(raw)
        assert arxiv == "2005.14165"
        assert doi == "10.48550/arxiv.2005.14165"

    def test_arxiv_version_stripped(self):
        raw = 'A. Vaswani et al., "Attention is All You Need," arXiv:1706.03762v5, 2017.'
        arxiv, doi = _extract_ids_from_citation_raw(raw)
        assert arxiv == "1706.03762"
        assert doi == "10.48550/arxiv.1706.03762"

    def test_explicit_doi(self):
        raw = 'He et al., "Deep Residual Learning," doi:10.1109/CVPR.2016.90, 2016.'
        arxiv, doi = _extract_ids_from_citation_raw(raw)
        assert doi == "10.1109/CVPR.2016.90"

    def test_doi_url_format(self):
        raw = 'LeCun et al., "Gradient-based learning," https://doi.org/10.1109/5.726791, 1998.'
        arxiv, doi = _extract_ids_from_citation_raw(raw)
        assert doi == "10.1109/5.726791"

    def test_arxiv_preprint_prefix(self):
        raw = "arXiv preprint arXiv:1706.03762"
        arxiv, doi = _extract_ids_from_citation_raw(raw)
        assert arxiv == "1706.03762"

    def test_no_ids_returns_empty(self):
        raw = 'Felix Weninger, "Introducing CURRENNT," JMLR, vol. 16, pp. 547-551, 2015.'
        arxiv, doi = _extract_ids_from_citation_raw(raw)
        assert arxiv == ""
        assert doi == ""

    def test_arxiv_synthesised_doi_not_overwritten_by_explicit_doi(self):
        # When both arxiv and an explicit doi are present, keep the real doi
        raw = 'Author, "Title," https://doi.org/10.1145/1234.5678, arXiv:2101.00001, 2021.'
        arxiv, doi = _extract_ids_from_citation_raw(raw)
        assert arxiv == "2101.00001"
        assert doi == "10.1145/1234.5678"   # explicit doi wins over synthesised one

    def test_trailing_punctuation_stripped(self):
        raw = "See arXiv:1502.03167."
        arxiv, doi = _extract_ids_from_citation_raw(raw)
        assert arxiv == "1502.03167"

    def test_old_style_arxiv_id(self):
        raw = "Collobert et al., NLP scratch, arXiv:cs/0705001, 2011."
        arxiv, doi = _extract_ids_from_citation_raw(raw)
        assert arxiv == "cs/0705001"


class TestResolveCitedPaperId:

    def test_span_with_doi_resolved(self):
        linked = [{"citation_raw": "Author et al. Title. 2020.", "alignment_score": 0.9}]
        cite_spans = {"cite_spans": [
            {"raw": "Author et al. Title. 2020.", "doi": "10.1000/xyz", "arxiv_id": "", "openalex_id": ""}
        ]}
        result = resolve_cited_paper_id(linked, cite_spans)
        assert len(result) == 1
        assert result[0]["doi"] == "10.1000/xyz"
        assert result[0]["alignment_score"] == 0.9

    def test_span_with_arxiv_resolved(self):
        linked = [{"citation_raw": "Ioffe 2015.", "alignment_score": 0.85}]
        cite_spans = {"cite_spans": [
            {"raw": "Ioffe 2015.", "arxiv_id": "1502.03167", "doi": "", "openalex_id": ""}
        ]}
        result = resolve_cited_paper_id(linked, cite_spans)
        assert result[0]["arxiv_id"] == "1502.03167"

    def test_alignment_score_preserved(self):
        linked = [{"citation_raw": "X.", "alignment_score": 0.77, "alignment_reason": "direct cite"}]
        cite_spans = {"cite_spans": [{"raw": "X.", "doi": "10.0/a", "arxiv_id": "", "openalex_id": ""}]}
        result = resolve_cited_paper_id(linked, cite_spans)
        assert result[0]["alignment_score"] == 0.77
        assert result[0]["alignment_reason"] == "direct cite"


    def test_span_no_ids_arxiv_in_raw_rescued(self):

        linked = [{"citation_raw": 'S. Ioffe, "Batch norm," arXiv:1502.03167, 2015.', "alignment_score": 0.9}]
        cite_spans = {"cite_spans": [
            {"raw": 'S. Ioffe, "Batch norm," arXiv:1502.03167, 2015.',
             "arxiv_id": "", "doi": "", "openalex_id": ""}
        ]}
        result = resolve_cited_paper_id(linked, cite_spans)
        assert len(result) == 1
        assert result[0]["arxiv_id"] == "1502.03167"
        assert result[0]["doi"] == "10.48550/arxiv.1502.03167"

    def test_span_no_ids_no_arxiv_in_raw_skipped(self):

        linked = [{"citation_raw": 'Felix Weninger, "CURRENNT," JMLR, 2015.', "alignment_score": 0.8}]
        cite_spans = {"cite_spans": [
            {"raw": 'Felix Weninger, "CURRENNT," JMLR, 2015.',
             "arxiv_id": "", "doi": "", "openalex_id": ""}
        ]}
        result = resolve_cited_paper_id(linked, cite_spans)
        assert result == []

    def test_openalex_only_span_but_arxiv_in_raw_rescued(self):

        linked = [{"citation_raw": "https://arxiv.org/abs/2005.14165, Brown 2020.", "alignment_score": 0.7}]
        cite_spans = {"cite_spans": [
            {"raw": "https://arxiv.org/abs/2005.14165, Brown 2020.",
             "arxiv_id": "", "doi": "", "openalex_id": "https://openalex.org/W123"}
        ]}
        result = resolve_cited_paper_id(linked, cite_spans)
        assert len(result) == 1
        assert result[0]["arxiv_id"] == "2005.14165"

    def test_citation_raw_mismatch_skipped(self):
        linked = [{"citation_raw": "LLM hallucinated this reference.", "alignment_score": 0.5}]
        cite_spans = {"cite_spans": [
            {"raw": "Real citation text.", "doi": "10.0/real", "arxiv_id": "", "openalex_id": ""}
        ]}
        result = resolve_cited_paper_id(linked, cite_spans)
        assert result == []

    def test_empty_linked_citations_returns_empty(self):
        assert resolve_cited_paper_id([], {"cite_spans": []}) == []

    def test_none_cite_spans_returns_empty(self):
        linked = [{"citation_raw": "anything", "alignment_score": 0.5}]
        assert resolve_cited_paper_id(linked, None) == []

    def test_multiple_citations_only_matched_ones_returned(self):
        linked = [
            {"citation_raw": "Good ref.", "alignment_score": 0.9},
            {"citation_raw": "Missing ref.", "alignment_score": 0.6},
        ]
        cite_spans = {"cite_spans": [
            {"raw": "Good ref.", "doi": "10.0/good", "arxiv_id": "", "openalex_id": ""}
        ]}
        result = resolve_cited_paper_id(linked, cite_spans)
        assert len(result) == 1
        assert result[0]["doi"] == "10.0/good"

    def test_first_occurrence_wins_for_duplicate_raws(self):
        linked = [{"citation_raw": "Dup.", "alignment_score": 0.9}]
        cite_spans = {"cite_spans": [
            {"raw": "Dup.", "doi": "10.0/first", "arxiv_id": "", "openalex_id": ""},
            {"raw": "Dup.", "doi": "10.0/second", "arxiv_id": "", "openalex_id": ""},
        ]}
        result = resolve_cited_paper_id(linked, cite_spans)
        assert result[0]["doi"] == "10.0/first"


class TestExtractTitleHint:

    def test_double_quoted_title(self):
        raw = 'S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in arXiv:1502.03167, 2015.'
        hint = HybridQueryRetriever._extract_title_hint(raw)
        assert hint == "Batch normalization: Accelerating deep network training by reducing internal covariate shift"

    def test_double_quoted_title_trailing_comma_stripped(self):
        raw = 'A. Vaswani et al., "Attention is all you need," NeurIPS 2017.'
        hint = HybridQueryRetriever._extract_title_hint(raw)
        assert hint == "Attention is all you need"

    def test_single_quoted_title(self):
        raw = "Author, 'Deep learning for NLP', ACL 2019."
        hint = HybridQueryRetriever._extract_title_hint(raw)
        assert hint == "Deep learning for NLP"

    def test_no_quotes_heuristic_fallback(self):
        # No quotes → split on first '.', strip trailing venue segment
        raw = "Brown et al. Language Models are Few-Shot Learners. NeurIPS 2020."
        hint = HybridQueryRetriever._extract_title_hint(raw)
        assert "Language Models are Few-Shot Learners" in hint

    def test_currennt_real_world(self):
        raw = 'Felix Weninger, "Introducing CURRENNT: The Munich open-source CUDA recurrent neural network toolkit," Journal of Machine Learning Research, vol. 16, pp. 547-551, 2015.'
        hint = HybridQueryRetriever._extract_title_hint(raw)
        assert "CURRENNT" in hint
        assert hint.startswith("Introducing CURRENNT")

    def test_capped_at_120_chars(self):
        long_title = "A " + "very " * 50 + "long title"
        raw = f'Author, "{long_title}", Venue 2020.'
        hint = HybridQueryRetriever._extract_title_hint(raw)
        assert len(hint) <= 120

    def test_title_at_least_10_chars_to_match_quotes(self):
        # A very short quoted string (< 10 chars) should not be picked up as a title;
        # falls back to heuristic
        raw = 'Author, "Hi," Venue. Full title here. 2020.'
        hint = HybridQueryRetriever._extract_title_hint(raw)
        # "Hi" is < 10 chars so it won't match; heuristic kicks in
        assert "Hi" not in hint or len(hint) > 2


def _make_retriever():

    with patch("retrieval.retriever.QDRANT_ACTIVE") as mock_profile, \
         patch("retrieval.retriever.QDRANT_CONNECTION") as mock_conn, \
         patch("retrieval.retriever.RETRIEVAL") as mock_retrieval, \
         patch("retrieval.retriever.EMBEDDING_MODELS") as mock_models, \
         patch("retrieval.retriever.RERANKER") as mock_reranker, \
         patch("retrieval.retriever.DEFAULT_EMBEDDING_MODEL", "bge-m3"), \
         patch("retrieval.retriever.QdrantClient") as mock_qdrant_cls:

        mock_profile.qdrant_profile = "local"
        mock_conn.url = "http://localhost:6333"
        mock_conn.api_key = "test"
        mock_conn.timeout = 10
        mock_retrieval.top_k = 10
        mock_retrieval.score_threshold = 0.1
        mock_retrieval.section_boost_gamma = 0.1
        mock_retrieval.enable_section_boost = False
        mock_retrieval.reranker_enabled = False
        mock_reranker.enabled = False
        mock_reranker.top_n = 5
        mock_models.__getitem__ = Mock(return_value=Mock(
            vector_size=1024, sparse_model=None, use_sparse=False
        ))

        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client

        mock_profile_obj = MagicMock()
        mock_profile_obj.dense_vector_name = "dense"
        mock_profile_obj.sparse_vector_name = "sparse"
        mock_profile_obj.collection_name = "unarxive_chunks"

        retriever = HybridQueryRetriever.__new__(HybridQueryRetriever)
        retriever.client = mock_client
        retriever.collection_name = "unarxive_chunks"
        retriever.model_name = "bge-m3"
        retriever.model_key = "bge-m3"
        retriever.profile = mock_profile_obj
        retriever.cross_encoder = None
        retriever.reranker = None
        retriever.use_bge_sparse = False

    return retriever, mock_client


def _scored_point(chunk_uid, paper_id, text, score=0.85):
    pt = MagicMock()
    pt.score = score
    pt.payload = {
        "chunk_uid": chunk_uid,
        "paper_id_arxiv": paper_id,
        "paper_doi": "",
        "embed_text": text,
        "section_title": "Methods",
        "chunk_index": 0,
        "total_chunks": 5,
        "title": "Batch Normalization Paper",
        "spans": {},
    }
    return pt


class TestRetrieveByTitleFallback:

    def test_hit_returns_chunks(self):
        retriever, mock_client = _make_retriever()
        pt = _scored_point("chunk_abc", "1502.03167", "Batch norm reduces covariate shift.")
        # _retrieve_by_title_fallback uses client.scroll → returns (points, next_offset)
        mock_client.scroll.return_value = ([pt], None)

        query_vec = [0.1] * 1024
        results = retriever._retrieve_by_title_fallback(
            query_vector=query_vec,
            citation_raw='S. Ioffe, "Batch normalization: Accelerating deep network training," arXiv:1502.03167, 2015.',
            look_for="batch normalization method details",
            top_k=2,
        )
        assert len(results) == 1
        assert results[0].chunk_uid == "chunk_abc"

    def test_miss_returns_empty_list(self):
        retriever, mock_client = _make_retriever()
        mock_client.scroll.return_value = ([], None)

        query_vec = [0.0] * 1024
        results = retriever._retrieve_by_title_fallback(
            query_vector=query_vec,
            citation_raw='Felix Weninger, "Introducing CURRENNT," JMLR 2015.',
            look_for="CURRENNT toolkit details",
            top_k=2,
        )
        assert results == []

    def test_empty_citation_raw_returns_empty(self):
        retriever, mock_client = _make_retriever()
        results = retriever._retrieve_by_title_fallback(
            query_vector=[0.0] * 1024,
            citation_raw="",
            look_for="something",
            top_k=2,
        )
        assert results == []
        mock_client.query_points.assert_not_called()

    def test_qdrant_exception_returns_empty(self):
        retriever, mock_client = _make_retriever()
        mock_client.scroll.side_effect = RuntimeError("connection refused")

        results = retriever._retrieve_by_title_fallback(
            query_vector=[0.1] * 1024,
            citation_raw='Author, "Valid title for testing," Venue 2020.',
            look_for="something",
            top_k=2,
        )
        assert results == []

    def test_title_hint_used_in_filter(self):

        retriever, mock_client = _make_retriever()
        mock_client.scroll.return_value = ([], None)

        retriever._retrieve_by_title_fallback(
            query_vector=[0.0] * 1024,
            citation_raw='S. Ioffe, "Batch normalization: Accelerating training," arXiv:1502.03167, 2015.',
            look_for="batch norm",
            top_k=2,
        )

        assert mock_client.scroll.called
        call_kwargs = mock_client.scroll.call_args
        # Extract scroll_filter kwarg — title_hint must appear, not the author prefix
        scroll_filter = call_kwargs.kwargs.get("scroll_filter") or (
            call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
        )
        filter_str = str(scroll_filter)
        assert "Batch normalization" in filter_str
        assert "S. Ioffe" not in filter_str


# ---------------------------------------------------------------------------
# ClaimRelevance enum
# ---------------------------------------------------------------------------

class TestClaimRelevanceEnum:

    def test_all_six_values_exist(self) -> None:
        values = {r.value for r in ClaimRelevance}
        assert values == {"method", "result", "limitation", "background", "dataset", "comparison"}

    def test_is_str_enum(self) -> None:
        assert ClaimRelevance.RESULT == "result"
        assert isinstance(ClaimRelevance.METHOD, str)

    def test_unknown_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ClaimRelevance("unknown_type")


# ---------------------------------------------------------------------------
# Prompt builders — new fields mentioned
# ---------------------------------------------------------------------------

def _make_chunk(content: str = "Text.", section: str = "Methods") -> RetrievedDocument:
    return RetrievedDocument(
        doc_id="paper_A",
        chunk_id="chunk_1",
        content=content,
        score=0.9,
        section_title=section,
    )


class TestPromptContainsNewFields:

    def test_system_prompt_has_why_relevant_field(self) -> None:
        assert "why_relevant_to_question" in EVIDENCE_GRAPH_SYSTEM_PROMPT

    def test_system_prompt_has_claim_type_field(self) -> None:
        assert "claim_type" in EVIDENCE_GRAPH_SYSTEM_PROMPT

    def test_system_prompt_lists_all_claim_types(self) -> None:
        for value in ClaimRelevance:
            assert value.value in EVIDENCE_GRAPH_SYSTEM_PROMPT

    # --- Rule 8: document-structure noise filter ---

    def test_system_prompt_bans_ref_placeholder(self) -> None:
        assert "REF" in EVIDENCE_GRAPH_SYSTEM_PROMPT

    def test_system_prompt_bans_table_figure_navigation(self) -> None:
        assert "document-structure" in EVIDENCE_GRAPH_SYSTEM_PROMPT

    def test_system_prompt_has_negative_example_for_ref(self) -> None:
        assert "Example 4" in EVIDENCE_GRAPH_SYSTEM_PROMPT
        assert "Table REF" in EVIDENCE_GRAPH_SYSTEM_PROMPT

    def test_user_prompt_mentions_why_relevant(self) -> None:
        prompt = build_claim_extraction_prompt(_make_chunk(), query="What is batch norm?")
        assert "why_relevant_to_question" in prompt

    def test_user_prompt_mentions_claim_type(self) -> None:
        prompt = build_claim_extraction_prompt(_make_chunk(), query="What is batch norm?")
        assert "claim_type" in prompt

    def test_batch_user_prompt_mentions_why_relevant(self) -> None:
        prompt = build_claim_extraction_prompt_batch([_make_chunk()], query="What is batch norm?")
        assert "why_relevant_to_question" in prompt

    def test_batch_user_prompt_mentions_claim_type(self) -> None:
        prompt = build_claim_extraction_prompt_batch([_make_chunk()], query="What is batch norm?")
        assert "claim_type" in prompt


# ---------------------------------------------------------------------------
# EvidenceGraphBuilderAgent — new fields stored on graph nodes
# ---------------------------------------------------------------------------

def _mock_llm_response(claims_json: str):
    """Return a mock LLMClient whose chat_text() always returns claims_json."""
    client = MagicMock()
    client.chat_text.return_value = claims_json
    return client


class TestGraphBuilderStoresNewFields:

    def _build(self, claims_json: str) -> dict:
        """Run EvidenceGraphBuilderAgent.build() with a mocked LLM and return node attrs."""
        llm = _mock_llm_response(claims_json)
        agent = EvidenceGraphBuilderAgent(llm_client=llm)
        doc = _make_chunk("Batch norm reduces covariate shift.", "Methods")
        graph, _ = agent.build(query="What is batch normalization?", sub_queries=[], documents=[doc])
        claim_nodes = {
            n.node_id: n
            for n in graph.nodes
            if n.node_type.value == "claim"
        }
        return claim_nodes

    def test_why_relevant_stored_on_node(self) -> None:
        claims = (
            '[{"text": "Batch norm reduces covariate shift.", "type": "claim", "subtype": "method",'
            ' "scicite_label": "METHOD",'
            ' "why_relevant_to_question": "Explains the core mechanism of batch normalization.",'
            ' "claim_type": "method",'
            ' "linked_citations": [], "hop_reason": "none", "look_for": ""}]'
        )
        nodes = self._build(claims)
        assert nodes, "Expected at least one claim node"
        node = next(iter(nodes.values()))
        assert node.metadata.get("why_relevant_to_question") == "Explains the core mechanism of batch normalization."

    def test_claim_type_stored_on_node(self) -> None:
        claims = (
            '[{"text": "Batch norm reduces covariate shift.", "type": "claim", "subtype": "method",'
            ' "scicite_label": "METHOD",'
            ' "why_relevant_to_question": "Explains the core mechanism.",'
            ' "claim_type": "method",'
            ' "linked_citations": [], "hop_reason": "none", "look_for": ""}]'
        )
        nodes = self._build(claims)
        node = next(iter(nodes.values()))
        assert node.metadata.get("claim_type") == "method"

    def test_invalid_claim_type_not_stored(self) -> None:
        """An unrecognised claim_type value must be silently dropped, not crash."""
        claims = (
            '[{"text": "Batch norm reduces covariate shift.", "type": "claim", "subtype": "method",'
            ' "scicite_label": "METHOD",'
            ' "why_relevant_to_question": "Some reason.",'
            ' "claim_type": "totally_invalid_value",'
            ' "linked_citations": [], "hop_reason": "none", "look_for": ""}]'
        )
        nodes = self._build(claims)
        node = next(iter(nodes.values()))
        assert "claim_type" not in node.metadata

    def test_missing_why_relevant_not_stored(self) -> None:
        """Absent why_relevant_to_question must not set an empty-string key on the node."""
        claims = (
            '[{"text": "Batch norm reduces covariate shift.", "type": "claim", "subtype": "method",'
            ' "scicite_label": "METHOD",'
            ' "claim_type": "method",'
            ' "linked_citations": [], "hop_reason": "none", "look_for": ""}]'
        )
        nodes = self._build(claims)
        node = next(iter(nodes.values()))
        assert "why_relevant_to_question" not in node.metadata

    def test_empty_why_relevant_not_stored(self) -> None:
        claims = (
            '[{"text": "Batch norm reduces covariate shift.", "type": "claim", "subtype": "method",'
            ' "scicite_label": "METHOD",'
            ' "why_relevant_to_question": "   ",'
            ' "claim_type": "result",'
            ' "linked_citations": [], "hop_reason": "none", "look_for": ""}]'
        )
        nodes = self._build(claims)
        node = next(iter(nodes.values()))
        assert "why_relevant_to_question" not in node.metadata
