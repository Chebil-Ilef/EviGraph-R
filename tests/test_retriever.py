import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retrieval.retriever import HybridQueryRetriever, ChunkResult
from qdrant_client.models import ScoredPoint, Payload, Filter, FieldCondition, MatchAny


@pytest.fixture
def mock_qdrant_client():

    client = Mock()
    return client


@pytest.fixture
def mock_cross_encoder():

    encoder = Mock()
    encoder.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
    return encoder


@pytest.fixture
def sample_qdrant_response():

    points = []
    for i in range(5):
        point = Mock(spec=ScoredPoint)
        point.score = 0.9 - i * 0.1
        point.payload = {
            "chunk_uid": f"chunk_{i}",
            "paper_id_arxiv": f"2024.{i:05d}",
            "embed_text": f"Sample text content {i}",
            "section_title": "Methods" if i % 2 == 0 else "Results",
            "chunk_type": "section",
            "chunk_index": i,
            "total_chunks": 10,
            "spans": {"citations": [f"doi:10.1000/{i}"]}
        }
        points.append(point)

    response = Mock()
    response.points = points
    return response


@pytest.fixture
def mock_settings():

    with patch('retrieval.retriever.QDRANT_ACTIVE') as mock_profile, \
         patch('retrieval.retriever.QDRANT_CONNECTION') as mock_conn, \
         patch('retrieval.retriever.BENCHMARK') as mock_benchmark, \
         patch('retrieval.retriever.EMBEDDING_MODELS') as mock_models, \
         patch('retrieval.retriever.RERANKER') as mock_reranker:

        # Mock profile
        mock_profile.collection_name = "test_collection"
        mock_profile.dense_vector_name = "dense"
        mock_profile.sparse_vector_name = "sparse"
        mock_profile.bm25_model = "Qdrant/bm25"

        # Mock connection
        mock_conn.url = None
        mock_conn.host = "localhost"
        mock_conn.port = 6333
        mock_conn.grpc_port = 6334
        mock_conn.prefer_grpc = False
        mock_conn.api_key = None

        # Mock benchmark
        mock_benchmark.dense_top_k = 100
        mock_benchmark.sparse_top_k = 100
        mock_benchmark.bm25_top_k = 100

        # Mock model configs
        mock_model_cfg = Mock()
        mock_model_cfg.bge_produces_sparse = False
        mock_models = {"bge-m3": mock_model_cfg}

        # Mock reranker
        mock_reranker.enabled = True
        mock_reranker.model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        mock_reranker.device = "cpu"
        mock_reranker.batch_size = 32
        mock_reranker.top_n = 50
        mock_reranker.min_score_threshold = 0.0

        yield {
            'profile': mock_profile,
            'connection': mock_conn,
            'benchmark': mock_benchmark,
            'models': mock_models,
            'reranker': mock_reranker
        }


@pytest.fixture
def retriever_bm25(mock_qdrant_client, mock_settings):

    with patch('retrieval.retriever.QdrantClient', return_value=mock_qdrant_client), \
         patch('retrieval.retriever.EMBEDDING_MODELS') as mock_models:

        # BM25 mode (non-BGE)
        mock_model_cfg = Mock()
        mock_model_cfg.bge_produces_sparse = False
        mock_models.__getitem__.return_value = mock_model_cfg

        retriever = HybridQueryRetriever(model_key="e5-base-v2")
        return retriever


@pytest.fixture
def retriever_bge_m3(mock_qdrant_client, mock_settings):

    with patch('retrieval.retriever.QdrantClient', return_value=mock_qdrant_client), \
         patch('retrieval.retriever.EMBEDDING_MODELS') as mock_models:

        # BGE-M3 mode
        mock_model_cfg = Mock()
        mock_model_cfg.bge_produces_sparse = True
        mock_models.__getitem__.return_value = mock_model_cfg

        retriever = HybridQueryRetriever(model_key="bge-m3")
        return retriever


class TestHybridQueryRetrieverInit:

    def test_init_with_reranker(self, mock_qdrant_client, mock_cross_encoder):

        with patch('retrieval.retriever.QdrantClient', return_value=mock_qdrant_client), \
             patch('retrieval.retriever.EMBEDDING_MODELS') as mock_models, \
             patch('retrieval.retriever.RERANKER') as mock_reranker, \
             patch('sentence_transformers.CrossEncoder', return_value=mock_cross_encoder):

            mock_model_cfg = Mock()
            mock_model_cfg.bge_produces_sparse = False
            mock_models.__getitem__.return_value = mock_model_cfg
            mock_reranker.enabled = True
            mock_reranker.model_id = "test-model"
            mock_reranker.device = "cpu"

            retriever = HybridQueryRetriever()

            assert retriever.reranker is not None
            assert retriever.use_bge_sparse is False

    def test_init_without_reranker(self, mock_qdrant_client):

        with patch('retrieval.retriever.QdrantClient', return_value=mock_qdrant_client), \
             patch('retrieval.retriever.EMBEDDING_MODELS') as mock_models, \
             patch('retrieval.retriever.RERANKER') as mock_reranker:

            mock_model_cfg = Mock()
            mock_model_cfg.bge_produces_sparse = False
            mock_models.__getitem__.return_value = mock_model_cfg
            mock_reranker.enabled = False

            retriever = HybridQueryRetriever()

            assert retriever.reranker is None

    def test_init_bge_m3_mode(self, mock_qdrant_client):

        with patch('retrieval.retriever.QdrantClient', return_value=mock_qdrant_client), \
             patch('retrieval.retriever.EMBEDDING_MODELS') as mock_models:

            mock_model_cfg = Mock()
            mock_model_cfg.bge_produces_sparse = True
            mock_models.__getitem__.return_value = mock_model_cfg

            retriever = HybridQueryRetriever(model_key="bge-m3")

            assert retriever.use_bge_sparse is True


class TestHybridQueryRetrieverRetrieve:

    def test_retrieve_bm25_mode(self, retriever_bm25, sample_qdrant_response):

        retriever_bm25.client.query_points.return_value = sample_qdrant_response
        retriever_bm25.reranker = None  # Disable reranker for this test

        embeddings = [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # Mock dense embedding
        query_text = "What is machine learning?"

        results = retriever_bm25.retrieve(
            embeddings=embeddings,
            query_text=query_text,
            top_k=3
        )

        assert len(results) == 3
        assert all(isinstance(r, ChunkResult) for r in results)
        assert results[0].chunk_uid == "chunk_0"
        assert results[0].paper_id == "2024.00000"
        assert results[0].cite_spans == {"citations": ["doi:10.1000/0"]}

        # Qdrant was called with correct parameters
        retriever_bm25.client.query_points.assert_called_once()
        call_args = retriever_bm25.client.query_points.call_args
        assert len(call_args.kwargs['prefetch']) == 2  # Dense + BM25 prefetch

    def test_retrieve_bge_m3_mode(self, retriever_bge_m3, sample_qdrant_response):

        retriever_bge_m3.client.query_points.return_value = sample_qdrant_response
        retriever_bge_m3.reranker = None

        embeddings = [0.1, 0.2, 0.3] * 100
        sparse_embeddings = {0: 0.8, 5: 0.6, 10: 0.4}
        query_text = "What is deep learning?"

        results = retriever_bge_m3.retrieve(
            embeddings=embeddings,
            query_text=query_text,
            sparse_embeddings=sparse_embeddings,
            top_k=2
        )

        assert len(results) == 2
        assert results[0].section_title == "Methods"

        retriever_bge_m3.client.query_points.assert_called_once()
        call_args = retriever_bge_m3.client.query_points.call_args
        assert len(call_args.kwargs['prefetch']) == 2  # Dense + Sparse prefetch

    def test_retrieve_with_section_filtering(self, retriever_bm25, sample_qdrant_response):

        retriever_bm25.client.query_points.return_value = sample_qdrant_response
        retriever_bm25.reranker = None

        embeddings = [0.1, 0.2, 0.3] * 100
        target_sections = ["Methods", "Results"]

        results = retriever_bm25.retrieve(
            embeddings=embeddings,
            query_text="test query",
            target_sections=target_sections
        )

        # section filter was applied
        call_args = retriever_bm25.client.query_points.call_args
        assert 'prefetch' in call_args.kwargs
        prefetch_calls = call_args.kwargs['prefetch']

        # prefetch calls should have filter
        for prefetch in prefetch_calls:
            assert prefetch.filter is not None

    def test_retrieve_with_reranking(self, retriever_bm25, sample_qdrant_response, mock_cross_encoder):

        retriever_bm25.client.query_points.return_value = sample_qdrant_response
        retriever_bm25.reranker = mock_cross_encoder

        embeddings = [0.1, 0.2, 0.3] * 100

        with patch('retrieval.retriever.RERANKER') as mock_reranker:
            mock_reranker.top_n = 50
            mock_reranker.batch_size = 32
            mock_reranker.min_score_threshold = 0.0

            results = retriever_bm25.retrieve(
                embeddings=embeddings,
                query_text="test query",
                top_k=3
            )

        assert len(results) == 3
        # reranker was called
        mock_cross_encoder.predict.assert_called_once()

        # scores should be updated by reranker
        expected_scores = [0.9, 0.8, 0.7]
        for i, result in enumerate(results):
            assert result.score == expected_scores[i]

    def test_retrieve_bm25_fallback_to_dense(self, retriever_bm25, sample_qdrant_response):

        retriever_bm25.client.query_points.side_effect = [
            Exception("Not existing vector name"),
            sample_qdrant_response
        ]
        retriever_bm25.reranker = None

        embeddings = [0.1, 0.2, 0.3] * 100

        results = retriever_bm25.retrieve(
            embeddings=embeddings,
            query_text="test query"
        )

        assert len(results) == 5  # all samples returned
        assert retriever_bm25.client.query_points.call_count == 2

    def test_retrieve_bge_m3_fallback_no_sparse(self, retriever_bge_m3, sample_qdrant_response):

        retriever_bge_m3.client.query_points.return_value = sample_qdrant_response
        retriever_bge_m3.reranker = None

        embeddings = [0.1, 0.2, 0.3] * 100

        with patch.object(retriever_bge_m3, '_retrieve_dense_only') as mock_dense:
            mock_dense.return_value = []

            results = retriever_bge_m3.retrieve(
                embeddings=embeddings,
                query_text="test query",
                sparse_embeddings=None  # this should trigger fallback
            )

            mock_dense.assert_called_once()

    def test_retrieve_error_handling(self, retriever_bm25):

        retriever_bm25.client.query_points.side_effect = Exception("Connection failed")

        embeddings = [0.1, 0.2, 0.3] * 100

        results = retriever_bm25.retrieve(
            embeddings=embeddings,
            query_text="test query"
        )

        assert results == []


class TestHybridQueryRetrieverHelpers:

    def test_retrieve_dense_only(self, retriever_bm25, sample_qdrant_response):

        limited_response = Mock()
        limited_response.points = sample_qdrant_response.points[:3]  # Only first 3 points
        retriever_bm25.client.query_points.return_value = limited_response

        embeddings = [0.1, 0.2, 0.3] * 100

        results = retriever_bm25._retrieve_dense_only(embeddings, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, ChunkResult) for r in results)

        call_args = retriever_bm25.client.query_points.call_args
        assert call_args.kwargs['query'] == embeddings
        assert call_args.kwargs['limit'] == 3
        assert call_args.kwargs['using'] == retriever_bm25.profile.dense_vector_name

    def test_retrieve_dense_only_with_filter(self, retriever_bm25, sample_qdrant_response):

        retriever_bm25.client.query_points.return_value = sample_qdrant_response

        embeddings = [0.1, 0.2, 0.3] * 100
        query_filter = Filter(
            must=[FieldCondition(key="section_title", match=MatchAny(any=["Methods"]))]
        )

        results = retriever_bm25._retrieve_dense_only(
            embeddings, top_k=5, query_filter=query_filter
        )

        assert len(results) == 5

        call_args = retriever_bm25.client.query_points.call_args
        assert call_args.kwargs['query_filter'] == query_filter

    def test_retrieve_dense_only_error(self, retriever_bm25):

        retriever_bm25.client.query_points.side_effect = Exception("Database error")

        embeddings = [0.1, 0.2, 0.3] * 100

        results = retriever_bm25._retrieve_dense_only(embeddings, top_k=5)

        assert results == []

    def test_rerank(self, retriever_bm25, mock_cross_encoder):

        chunks = [
            ChunkResult(
                chunk_uid=f"chunk_{i}",
                paper_id=f"paper_{i}",
                score=0.5 + i * 0.1,  
                embed_text=f"text {i}",
                section_title="Methods"
            )
            for i in range(3)
        ]

        retriever_bm25.reranker = mock_cross_encoder
        mock_cross_encoder.predict.return_value = [0.9, 0.8, 0.7]  # Reranker scores

        with patch('retrieval.retriever.RERANKER') as mock_reranker:
            mock_reranker.batch_size = 32
            mock_reranker.min_score_threshold = 0.0

            reranked = retriever_bm25._rerank("test query", chunks, top_k=2)

        assert len(reranked) == 2
        assert reranked[0].score == 0.9  # Highest reranker score first
        assert reranked[1].score == 0.8

        mock_cross_encoder.predict.assert_called_once()
        call_args = mock_cross_encoder.predict.call_args
        assert len(call_args[0][0]) == 3  # 3 query-document pairs
        assert call_args.kwargs['batch_size'] == 32
        assert call_args.kwargs['show_progress_bar'] is False

    def test_rerank_empty_results(self, retriever_bm25, mock_cross_encoder):

        retriever_bm25.reranker = mock_cross_encoder

        result = retriever_bm25._rerank("test query", [], top_k=5)

        assert result == []
        mock_cross_encoder.predict.assert_not_called()


class TestChunkResult:

    def test_chunk_result_creation(self):

        chunk = ChunkResult(
            chunk_uid="test_chunk",
            paper_id="2024.12345",
            score=0.95,
            embed_text="Sample text content",
            section_title="Abstract",
            cite_spans={"citations": ["doi:10.1000/123"]}
        )

        assert chunk.chunk_uid == "test_chunk"
        assert chunk.paper_id == "2024.12345"
        assert chunk.score == 0.95
        assert chunk.section_title == "Abstract"
        assert chunk.cite_spans == {"citations": ["doi:10.1000/123"]}

    def test_chunk_result_optional_fields(self):

        chunk = ChunkResult(
            chunk_uid="test_chunk",
            paper_id="2024.12345",
            score=0.85,
            embed_text="Sample text"
        )

        assert chunk.section_title is None
        assert chunk.chunk_type is None
        assert chunk.chunk_index is None
        assert chunk.total_chunks is None
        assert chunk.cite_spans is None


@pytest.mark.integration
class TestHybridQueryRetrieverIntegration:

    @pytest.mark.skip(reason="Requires actual Qdrant instance")
    def test_real_qdrant_integration(self):
        pass

    @pytest.mark.skip(reason="Requires sentence-transformers")
    def test_real_cross_encoder_integration(self):
        pass