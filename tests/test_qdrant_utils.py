import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.embedder import BGEOutput
from src.utils.qdrant import (
    build_points,
    get_collection_info,
    setup_collection,
    uid_to_uuid,
)
from src.core.config import QDRANT_LAPTOP


class TestUidToUuid:
    def test_returns_valid_uuid_string(self):
        result = uid_to_uuid("a" * 40)
        parsed = uuid.UUID(result)   # raises if invalid
        assert str(parsed) == result

    def test_deterministic(self):
        uid = "b" * 40
        assert uid_to_uuid(uid) == uid_to_uuid(uid)

    def test_different_inputs_different_output(self):
        assert uid_to_uuid("a" * 40) != uid_to_uuid("b" * 40)



def _mock_client(existing_collections=None):
    """Return a MagicMock that mimics QdrantClient."""
    client = MagicMock()
    names = existing_collections or []

    fake_collections = [MagicMock(name=n) for n in names]
    # Make each fake have .name attribute properly
    for fc, n in zip(fake_collections, names):
        fc.name = n

    client.get_collections.return_value = MagicMock(collections=fake_collections)

    # get_collection return value for get_collection_info
    fake_info = MagicMock()
    fake_info.points_count = 42
    fake_info.indexed_vectors_count = 10
    fake_info.status = "green"
    client.get_collection.return_value = fake_info

    return client


class TestSetupCollection:
    def test_creates_collection_when_not_existing(self):
        client = _mock_client(existing_collections=[])
        setup_collection(client, model_key="e5-base-v2", profile=QDRANT_LAPTOP)
        client.create_collection.assert_called_once()

    def test_skips_when_collection_exists_no_recreate(self):
        col_name = QDRANT_LAPTOP.collection_name
        client = _mock_client(existing_collections=[col_name])
        setup_collection(client, model_key="e5-base-v2", profile=QDRANT_LAPTOP, recreate=False)
        client.create_collection.assert_not_called()
        client.delete_collection.assert_not_called()

    def test_deletes_then_creates_on_recreate(self):
        col_name = QDRANT_LAPTOP.collection_name
        # _mock_client already sets .name as a proper attribute on each entry
        client = _mock_client(existing_collections=[col_name])
        setup_collection(client, model_key="e5-base-v2", profile=QDRANT_LAPTOP, recreate=True)
        client.delete_collection.assert_called_once_with(col_name)
        client.create_collection.assert_called_once()

    def test_payload_indexes_created(self):
        client = _mock_client(existing_collections=[])
        setup_collection(client, model_key="e5-base-v2", profile=QDRANT_LAPTOP)
        # create_payload_index should be called for each field + fulltext
        assert client.create_payload_index.call_count >= len(QDRANT_LAPTOP.payload_indexes)

    def test_collection_name_matches_profile(self):
        client = _mock_client(existing_collections=[])
        setup_collection(client, model_key="e5-base-v2", profile=QDRANT_LAPTOP)
        _, kwargs = client.create_collection.call_args
        assert kwargs.get("collection_name") == QDRANT_LAPTOP.collection_name



def _make_chunks(n: int) -> list[dict]:
    return [
        {
            "chunk_uid": "a" * 39 + str(i),
            "chunk_type": "abstract",
            "section_title": "Abstract",
            "embed_text": f"Text number {i}",
            "paper_id_arxiv": "1234.56789",
        }
        for i in range(n)
    ]


class TestBuildPointsDense:
    def test_returns_correct_number_of_points(self):
        n = 5
        chunks = _make_chunks(n)
        embeddings = np.random.randn(n, 768).astype(np.float32)
        points = build_points(chunks, embeddings, profile=QDRANT_LAPTOP, model_key="e5-base-v2")
        assert len(points) == n

    def test_point_ids_are_valid_uuids(self):
        chunks = _make_chunks(3)
        embeddings = np.random.randn(3, 768).astype(np.float32)
        points = build_points(chunks, embeddings, profile=QDRANT_LAPTOP, model_key="e5-base-v2")
        for p in points:
            parsed = uuid.UUID(str(p.id))
            assert str(parsed) == str(p.id)

    def test_point_vector_is_list_of_floats(self):
        chunks = _make_chunks(2)
        embeddings = np.random.randn(2, 768).astype(np.float32)
        points = build_points(chunks, embeddings, profile=QDRANT_LAPTOP, model_key="e5-base-v2")
        for p in points:
            assert isinstance(p.vector, list)
            assert all(isinstance(v, float) for v in p.vector[:5])

    def test_payload_contains_chunk_data(self):
        chunks = _make_chunks(2)
        embeddings = np.random.randn(2, 768).astype(np.float32)
        points = build_points(chunks, embeddings, profile=QDRANT_LAPTOP, model_key="e5-base-v2")
        for p, c in zip(points, chunks):
            assert p.payload["chunk_uid"] == c["chunk_uid"]

    def test_deterministic_ids_for_same_chunk(self):
        chunks = _make_chunks(1)
        e1 = np.random.randn(1, 768).astype(np.float32)
        e2 = np.random.randn(1, 768).astype(np.float32)   # different vectors
        p1 = build_points(chunks, e1, profile=QDRANT_LAPTOP, model_key="e5-base-v2")
        p2 = build_points(chunks, e2, profile=QDRANT_LAPTOP, model_key="e5-base-v2")
        # Same chunk → same id regardless of vector values
        assert str(p1[0].id) == str(p2[0].id)


class TestGetCollectionInfo:
    def test_returns_dict_with_expected_keys(self):
        client = _mock_client()
        info = get_collection_info(client, "my_collection")
        assert "collection_name" in info
        assert "points_count" in info
        assert "indexed_vectors_count" in info
        assert "status" in info

    def test_collection_name_in_result(self):
        client = _mock_client()
        info = get_collection_info(client, "papers")
        assert info["collection_name"] == "papers"

    def test_calls_get_collection(self):
        client = _mock_client()
        get_collection_info(client, "papers")
        client.get_collection.assert_called_once_with("papers")
