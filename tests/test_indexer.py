import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.indexer import index_batches
from src.utils.qdrant import check_qdrant_alive as _check_qdrant_alive
from src.core.config import QDRANT_LAPTOP



class TestCheckQdrantAlive:
    def test_passes_when_get_collections_succeeds(self):
        client = MagicMock()
        client.get_collections.return_value = MagicMock(collections=[])
        # Should not raise
        _check_qdrant_alive(client)

    def test_raises_runtime_error_when_unreachable(self):
        client = MagicMock()
        client.get_collections.side_effect = ConnectionRefusedError("refused")
        with pytest.raises(RuntimeError, match="not reachable"):
            _check_qdrant_alive(client)



def _make_chunks(n=3):
    return [
        {
            "chunk_uid": "a" * 39 + str(i),
            "chunk_type": "abstract",
            "section_title": "Abstract",
            "embed_text": f"Text {i}",
            "paper_id_arxiv": "1234.56789",
        }
        for i in range(n)
    ]


def _mock_qdrant_client():
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    fake_info = MagicMock()
    fake_info.points_count = 3
    fake_info.indexed_vectors_count = 3
    fake_info.status = "green"
    client.get_collection.return_value = fake_info
    return client


class TestIndexBatches:
    def _run(self, batch_stems, chunks_exist=True, chunks=None, recreate=False):
        """Run index_batches with fully mocked dependencies."""
        if chunks is None:
            chunks = _make_chunks()
        dense_vecs = np.random.randn(len(chunks), 768).astype(np.float32)

        mock_client = _mock_qdrant_client()
        mock_embedder = MagicMock()
        mock_embedder.embed_passages.return_value = dense_vecs

        with patch("src.core.indexer.qdrant_client", return_value=mock_client), \
             patch("src.core.indexer.setup_collection") as mock_setup, \
             patch("src.core.indexer.build_points", return_value=[MagicMock() for _ in chunks]), \
             patch("src.core.indexer.get_collection_info", return_value={
                 "collection_name": "unarxive_chunks",
                 "points_count": len(chunks),
                 "indexed_vectors_count": len(chunks),
                 "status": "green",
             }), \
             patch("src.core.indexer.chunks_file_exists", return_value=chunks_exist), \
             patch("src.core.indexer.load_chunks", return_value=chunks), \
             patch("src.core.indexer.Embedder.from_model_key", return_value=mock_embedder):

            index_batches(
                batch_stems=batch_stems,
                model_key="e5-base-v2",
                recreate=recreate,
                profile=QDRANT_LAPTOP,
            )
            return mock_client, mock_setup, mock_embedder

    def test_setup_collection_is_called(self):
        _, mock_setup, _ = self._run(["batch_01"])
        mock_setup.assert_called_once()

    def test_upsert_called_for_each_batch(self):
        mock_client, _, _ = self._run(["batch_01"])
        mock_client.upsert.assert_called()

    def test_skips_missing_chunks_file(self):
        mock_client, _, mock_embedder = self._run(["batch_01"], chunks_exist=False)
        mock_embedder.embed_passages.assert_not_called()
        mock_client.upsert.assert_not_called()

    def test_skips_empty_chunks(self):
        mock_client, _, mock_embedder = self._run(["batch_01"], chunks=[])
        mock_embedder.embed_passages.assert_not_called()
        mock_client.upsert.assert_not_called()

    def test_recreate_flag_passed_to_setup(self):
        _, mock_setup, _ = self._run(["batch_01"], recreate=True)
        _, kwargs = mock_setup.call_args
        assert kwargs.get("recreate") is True

    def test_multiple_batch_stems(self):
        mock_client, _, mock_embedder = self._run(["batch_01", "batch_02"])
        # embed_passages should be called once per batch (one upsert-batch each)
        assert mock_embedder.embed_passages.call_count >= 2

    def test_all_shorthand_resolved(self):
        """'all' should resolve to whatever glob finds; here we mock it to one stem."""
        chunks = _make_chunks(2)
        dense_vecs = np.random.randn(len(chunks), 768).astype(np.float32)
        mock_client = _mock_qdrant_client()
        mock_embedder = MagicMock()
        mock_embedder.embed_passages.return_value = dense_vecs

        fake_glob_results = [Path("/fake/batch_01_chunks.jsonl")]

        with patch("src.core.indexer.qdrant_client", return_value=mock_client), \
             patch("src.core.indexer.setup_collection"), \
             patch("src.core.indexer.build_points", return_value=[MagicMock() for _ in chunks]), \
             patch("src.core.indexer.get_collection_info", return_value={
                 "collection_name": "unarxive_chunks",
                 "points_count": len(chunks),
                 "indexed_vectors_count": len(chunks),
                 "status": "green",
             }), \
             patch("src.core.indexer.chunks_file_exists", return_value=True), \
             patch("src.core.indexer.load_chunks", return_value=chunks), \
             patch("src.core.indexer.Embedder.from_model_key", return_value=mock_embedder), \
             patch("src.core.indexer.PATHS") as mock_paths:

            mock_paths.chunks.glob.return_value = fake_glob_results
            index_batches(batch_stems=["all"], model_key="e5-base-v2", profile=QDRANT_LAPTOP)

        mock_client.upsert.assert_called()
