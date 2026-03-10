import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.retriever import UniversalQueryRetriever, ChunkResult


def _make_mock_point(chunk_uid="uid_001", paper_id="1234.5678", score=0.9,
                     embed_text="Some passage", section_title="Intro",
                     chunk_type="paragraph"):
    point = MagicMock()
    point.score = score
    point.payload = {
        "chunk_uid": chunk_uid,
        "paper_id_arxiv": paper_id,
        "embed_text": embed_text,
        "section_title": section_title,
        "chunk_type": chunk_type,
    }
    return point


def _make_retriever():
    """Build a UniversalQueryRetriever with QdrantClient fully mocked."""
    mock_client = MagicMock()
    with patch("src.core.retriever.QdrantClient", return_value=mock_client):
        retriever = UniversalQueryRetriever()
    retriever.client = mock_client
    return retriever, mock_client


class TestUniversalQueryRetrieverInit:
    def test_collection_name_defaults_to_config(self):
        mock_client = MagicMock()
        with patch("src.core.retriever.QdrantClient", return_value=mock_client):
            r = UniversalQueryRetriever()
        from src.core.config import QDRANT_ACTIVE
        assert r.collection_name == QDRANT_ACTIVE.collection_name

    def test_collection_name_override(self):
        mock_client = MagicMock()
        with patch("src.core.retriever.QdrantClient", return_value=mock_client):
            r = UniversalQueryRetriever(collection_name="my_custom_collection")
        assert r.collection_name == "my_custom_collection"

    def test_qdrant_client_is_constructed(self):
        mock_ctor = MagicMock()
        with patch("src.core.retriever.QdrantClient", mock_ctor):
            UniversalQueryRetriever()
        mock_ctor.assert_called_once()

class TestRetrieve:
    def test_returns_chunk_result_list(self):
        retriever, mock_client = _make_retriever()
        mock_response = MagicMock()
        mock_response.points = [_make_mock_point()]
        mock_client.query_points.return_value = mock_response

        results = retriever.retrieve([0.1] * 768, "What is optimization?", top_k=5)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], ChunkResult)

    def test_chunk_result_fields_mapped_correctly(self):
        retriever, mock_client = _make_retriever()
        point = _make_mock_point(
            chunk_uid="uid_abc",
            paper_id="9876.5432",
            score=0.75,
            embed_text="Dense text here",
            section_title="Methods",
            chunk_type="section",
        )
        mock_response = MagicMock()
        mock_response.points = [point]
        mock_client.query_points.return_value = mock_response

        result = retriever.retrieve([0.0] * 768, "methods", top_k=3)[0]
        assert result.chunk_uid == "uid_abc"
        assert result.paper_id == "9876.5432"
        assert result.score == 0.75
        assert result.embed_text == "Dense text here"
        assert result.section_title == "Methods"
        assert result.chunk_type == "section"

    def test_query_points_called_with_correct_collection(self):
        retriever, mock_client = _make_retriever()
        mock_client.query_points.return_value = MagicMock(points=[])

        retriever.collection_name = "test_collection"
        retriever.retrieve([0.0] * 768, "text", top_k=5)

        _, kwargs = mock_client.query_points.call_args
        assert kwargs.get("collection_name") == "test_collection"

    def test_query_points_called_with_no_prefetches_when_dense_only(self):
        """Default profile has enable_sparse=False → simple dense query, no prefetches."""
        retriever, mock_client = _make_retriever()
        mock_client.query_points.return_value = MagicMock(points=[])

        retriever.retrieve([0.0] * 768, "some query text", top_k=5)

        _, kwargs = mock_client.query_points.call_args
        prefetches = kwargs.get("prefetch", [])
        assert prefetches == []  # dense-only path sends no prefetches

    def test_query_points_called_with_two_prefetches_when_sparse_enabled(self):
        """When enable_sparse=True, two prefetches (dense + sparse) are sent."""
        from src.core.config import QDRANT_LAPTOP
        import dataclasses
        sparse_profile = dataclasses.replace(QDRANT_LAPTOP, enable_sparse=True)

        mock_client = MagicMock()
        with patch("src.core.retriever.QdrantClient", return_value=mock_client):
            retriever = UniversalQueryRetriever(profile=sparse_profile)
        retriever.client = mock_client
        mock_client.query_points.return_value = MagicMock(points=[])

        retriever.retrieve([0.0] * 768, "some query text", top_k=5)

        _, kwargs = mock_client.query_points.call_args
        prefetches = kwargs.get("prefetch", [])
        assert len(prefetches) == 2  # dense + sparse

    def test_top_k_respected(self):
        retriever, mock_client = _make_retriever()
        mock_response = MagicMock()
        mock_response.points = [_make_mock_point(score=1.0 - i * 0.1) for i in range(3)]
        mock_client.query_points.return_value = mock_response

        results = retriever.retrieve([0.0] * 768, "query", top_k=3)
        assert len(results) == 3

    def test_returns_empty_list_on_exception(self):
        retriever, mock_client = _make_retriever()
        mock_client.query_points.side_effect = RuntimeError("connection lost")

        results = retriever.retrieve([0.0] * 768, "query", top_k=5)
        assert results == []

    def test_multiple_results_ordered_by_response(self):
        retriever, mock_client = _make_retriever()
        points = [
            _make_mock_point(chunk_uid=f"uid_{i}", score=1.0 - i * 0.1)
            for i in range(4)
        ]
        mock_client.query_points.return_value = MagicMock(points=points)

        results = retriever.retrieve([0.0] * 768, "query", top_k=4)
        assert [r.chunk_uid for r in results] == ["uid_0", "uid_1", "uid_2", "uid_3"]
