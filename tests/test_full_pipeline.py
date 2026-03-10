"""
Mock end-to-end pipeline test.

Simulates the full workflow without any real model loading or Qdrant server:
    raw paper dict
      → build_paper_chunks()
      → save_chunks()           (written to tmp_path)
      → index_batches()         (all Qdrant + embedder calls mocked)
      → UniversalQueryRetriever.retrieve()  (Qdrant client mocked)
      → ChunkResult objects returned

Asserts that:
  - chunks are produced from a real-looking paper dict
  - save/load round-trip preserves data
  - upsert is called during indexing
  - retrieve returns ChunkResult objects with correct structure
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.builder import build_paper_chunks, save_chunks, load_chunks
from src.core.config import QDRANT_LAPTOP
from src.core.indexer import index_batches
from src.core.retriever import UniversalQueryRetriever, ChunkResult


# Minimal synthetic paper 

PAPER = {
    "paper_id": "1234.56789",
    "metadata": {
        "doi": "10.1000/xyz123",
        "title": "A Test Paper on Machine Learning",
    },
    "abstract": {
        "text": (
            "We study optimization methods for deep learning. "
            "Our approach combines gradient descent with novel regularization. "
            "Experiments show strong results on benchmark datasets."
        )
    },
    "body_text": [
        {
            "section": "Introduction",
            "text": (
                "Deep learning has transformed many fields. "
                "In this paper we propose a new method. "
                "The method is inspired by classical optimization theory."
            ),
            "cite_spans": [],
            "ref_spans": [],
        },
        {
            "section": "Method",
            "text": (
                "Our algorithm iterates over mini-batches. "
                "We apply momentum to accelerate convergence. "
                "The learning rate is scheduled according to a cosine decay."
            ),
            "cite_spans": [],
            "ref_spans": [],
        },
    ],
    "bib_entries": {},
    "ref_entries": {},
    "back_matter": [],
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def patched_paths(tmp_path):
    """Redirect PATHS.chunks to a temp directory for safe I/O."""
    with patch("src.core.builder.PATHS") as mock_paths:
        mock_paths.chunks = tmp_path
        yield tmp_path


@pytest.fixture()
def chunks_from_paper(patched_paths):
    """Build and save chunks from the synthetic paper; return (chunks, stem)."""
    stem = "test_batch"

    with patch("src.core.chunker.get_tokenizer") as mock_tok:
        tokenizer = MagicMock()
        tokenizer.encode = lambda text, **kw: list(range(max(1, len(text) // 4)))
        mock_tok.return_value = tokenizer

        with patch("src.core.preprocessor.build_work_id_map", return_value={}):
            chunks = build_paper_chunks(PAPER)

    assert len(chunks) > 0, "Expected at least one chunk from the synthetic paper"

    save_chunks(chunks, stem)
    return chunks, stem


# Stage 1: chunking 

class TestChunkingStage:
    def test_chunks_produced(self, chunks_from_paper):
        chunks, _ = chunks_from_paper
        assert len(chunks) > 0

    def test_chunks_have_required_fields(self, chunks_from_paper):
        chunks, _ = chunks_from_paper
        required = {"chunk_uid", "embed_text", "paper_id_arxiv", "chunk_type"}
        for chunk in chunks:
            assert required.issubset(chunk.keys()), f"Missing keys in chunk: {chunk.keys()}"

    def test_paper_id_matches(self, chunks_from_paper):
        chunks, _ = chunks_from_paper
        for chunk in chunks:
            assert chunk["paper_id_arxiv"] == "1234.56789"


# Stage 2: save / load round-trip 

class TestChunkIOStage:
    def test_save_load_roundtrip(self, chunks_from_paper, patched_paths):
        chunks, stem = chunks_from_paper
        loaded = load_chunks(stem)
        assert len(loaded) == len(chunks)
        assert loaded[0]["chunk_uid"] == chunks[0]["chunk_uid"]

    def test_loaded_embed_text_preserved(self, chunks_from_paper, patched_paths):
        chunks, stem = chunks_from_paper
        loaded = load_chunks(stem)
        for orig, loaded_chunk in zip(chunks, loaded):
            assert orig["embed_text"] == loaded_chunk["embed_text"]


# Stage 3: indexing 

class TestIndexingStage:
    def test_upsert_called(self, chunks_from_paper):
        chunks, stem = chunks_from_paper
        dense_vecs = np.random.randn(len(chunks), 768).astype(np.float32)

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_embedder = MagicMock()
        mock_embedder.embed_passages.return_value = dense_vecs

        with patch("src.core.indexer.qdrant_client", return_value=mock_client), \
             patch("src.core.indexer.setup_collection"), \
             patch("src.core.indexer.build_points",
                   return_value=[MagicMock() for _ in chunks]), \
             patch("src.core.indexer.get_collection_info", return_value={
                 "collection_name": "unarxive_chunks",
                 "points_count": len(chunks),
                 "indexed_vectors_count": len(chunks),
                 "status": "green",
             }), \
             patch("src.core.indexer.chunks_file_exists", return_value=True), \
             patch("src.core.indexer.load_chunks", return_value=chunks), \
             patch("src.core.indexer.Embedder.from_model_key", return_value=mock_embedder):

            index_batches(
                batch_stems=[stem],
                model_key="e5-base-v2",
                recreate=False,
                profile=QDRANT_LAPTOP,
            )

        mock_client.upsert.assert_called()

    def test_embed_passages_called_with_texts(self, chunks_from_paper, patched_paths):
        chunks, stem = chunks_from_paper
        dense_vecs = np.random.randn(len(chunks), 768).astype(np.float32)

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_embedder = MagicMock()
        mock_embedder.embed_passages.return_value = dense_vecs

        with patch("src.core.indexer.qdrant_client", return_value=mock_client), \
             patch("src.core.indexer.setup_collection"), \
             patch("src.core.indexer.build_points",
                   return_value=[MagicMock() for _ in chunks]), \
             patch("src.core.indexer.get_collection_info", return_value={
                 "collection_name": "unarxive_chunks",
                 "points_count": len(chunks),
                 "indexed_vectors_count": len(chunks),
                 "status": "green",
             }), \
             patch("src.core.indexer.chunks_file_exists", return_value=True), \
             patch("src.core.indexer.load_chunks", return_value=chunks), \
             patch("src.core.indexer.Embedder.from_model_key", return_value=mock_embedder):

            index_batches(
                batch_stems=[stem],
                model_key="e5-base-v2",
                recreate=False,
                profile=QDRANT_LAPTOP,
            )

        mock_embedder.embed_passages.assert_called()
        texts_arg = mock_embedder.embed_passages.call_args[0][0]
        assert all(isinstance(t, str) for t in texts_arg)


# Stage 4: retrieval 

class TestRetrievalStage:
    def _mock_retriever(self, results):
        """Return a retriever whose client returns `results` on query_points."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = results
        mock_client.query_points.return_value = mock_response

        with patch("src.core.retriever.QdrantClient", return_value=mock_client):
            retriever = UniversalQueryRetriever()
        retriever.client = mock_client
        return retriever

    def _make_point(self, uid, score):
        p = MagicMock()
        p.score = score
        p.payload = {
            "chunk_uid": uid,
            "paper_id_arxiv": "1234.56789",
            "embed_text": f"text for {uid}",
            "section_title": "Introduction",
            "chunk_type": "paragraph",
        }
        return p

    def test_retrieve_returns_chunk_results(self):
        points = [self._make_point(f"uid_{i}", 1.0 - i * 0.1) for i in range(3)]
        retriever = self._mock_retriever(points)
        query_vec = [0.1] * 768
        results = retriever.retrieve(query_vec, "optimization methods", top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, ChunkResult) for r in results)

    def test_retrieve_results_have_paper_id(self):
        points = [self._make_point("uid_0", 0.95)]
        retriever = self._mock_retriever(points)
        results = retriever.retrieve([0.0] * 768, "query", top_k=1)
        assert results[0].paper_id == "1234.56789"

    def test_retrieve_scores_are_positive(self):
        points = [self._make_point(f"uid_{i}", 0.9 - i * 0.05) for i in range(5)]
        retriever = self._mock_retriever(points)
        results = retriever.retrieve([0.0] * 768, "test query", top_k=5)
        assert all(r.score >= 0 for r in results)


#  Full end-to-end smoke test 

class TestFullPipelineSmoke:
    """
    Single test that runs the entire mock pipeline from paper dict → ChunkResult.
    All external I/O (model, Qdrant) is mocked.
    """
    def test_paper_to_results(self):
        stem = "pipeline_smoke"

        with patch("src.core.chunker.get_tokenizer") as mock_tok:
            tokenizer = MagicMock()
            tokenizer.encode = lambda text, **kw: list(range(max(1, len(text) // 4)))
            mock_tok.return_value = tokenizer
            with patch("src.core.preprocessor.build_work_id_map", return_value={}):
                chunks = build_paper_chunks(PAPER)

        assert len(chunks) > 0

        save_chunks(chunks, stem)
        loaded = load_chunks(stem)
        assert len(loaded) == len(chunks)

        dense_vecs = np.random.randn(len(chunks), 768).astype(np.float32)
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_embedder = MagicMock()
        mock_embedder.embed_passages.return_value = dense_vecs

        with patch("src.core.indexer.qdrant_client", return_value=mock_client), \
             patch("src.core.indexer.setup_collection"), \
             patch("src.core.indexer.build_points",
                   return_value=[MagicMock() for _ in chunks]), \
             patch("src.core.indexer.get_collection_info", return_value={
                 "collection_name": "unarxive_chunks",
                 "points_count": len(chunks),
                 "indexed_vectors_count": len(chunks),
                 "status": "green",
             }), \
             patch("src.core.indexer.chunks_file_exists", return_value=True), \
             patch("src.core.indexer.load_chunks", return_value=chunks), \
             patch("src.core.indexer.Embedder.from_model_key", return_value=mock_embedder):

            index_batches(
                batch_stems=[stem],
                model_key="e5-base-v2",
                recreate=False,
                profile=QDRANT_LAPTOP,
            )

        mock_client.upsert.assert_called()

        retrieval_points = []
        for i, chunk in enumerate(chunks[:3]):
            p = MagicMock()
            p.score = 0.9 - i * 0.05
            p.payload = {
                "chunk_uid": chunk["chunk_uid"],
                "paper_id_arxiv": chunk["paper_id_arxiv"],
                "embed_text": chunk["embed_text"],
                "section_title": chunk.get("section_title"),
                "chunk_type": chunk["chunk_type"],
            }
            retrieval_points.append(p)

        mock_retrieval_client = MagicMock()
        mock_retrieval_client.query_points.return_value = MagicMock(
            points=retrieval_points
        )

        with patch("src.core.retriever.QdrantClient",
                   return_value=mock_retrieval_client):
            retriever = UniversalQueryRetriever()
        retriever.client = mock_retrieval_client

        query_vec = np.random.randn(768).tolist()
        results = retriever.retrieve(query_vec,
                                     "What optimization method is used?",
                                     top_k=5)

        assert len(results) > 0
        assert all(isinstance(r, ChunkResult) for r in results)
        assert results[0].paper_id == "1234.56789"
