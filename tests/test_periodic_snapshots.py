from __future__ import annotations
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch
from indexing.ingestion import ingest_shards
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger(__name__)


class TestPeriodicSnapshotBehavior:

    @patch("indexing.ingestion.qdrant_client")
    @patch("indexing.ingestion.setup_collection")
    @patch("indexing.ingestion.get_collection_info")
    @patch("indexing.ingestion.get_qdrant_profile")
    @patch("indexing.ingestion.shard_artifacts")
    @patch("indexing.ingestion.read_jsonl")
    @patch("indexing.ingestion.append_jsonl")
    @patch("indexing.ingestion.build_points_from_shard_records")
    @patch("indexing.ingestion.create_collection_snapshot")
    def test_periodic_snapshots_created_at_interval(
        self,
        mock_create_snapshot,
        mock_build_points,
        mock_append_jsonl,
        mock_read_jsonl,
        mock_shard_artifacts,
        mock_profile_fn,
        mock_collection_info,
        mock_setup_collection,
        mock_client_fn,
    ):

        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client

        mock_profile = MagicMock(
            collection_name="test_collection", upsert_batch_size=1000
        )
        mock_profile_fn.return_value = mock_profile

        mock_build_points.return_value = [{"id": "chunk_0", "vector": [0.1] * 384}]

        mock_create_snapshot.side_effect = [f"snap-{i}" for i in range(10)]

        shard_stems = [f"shard_{i:04d}" for i in range(300)]

        mock_artifacts = MagicMock()
        mock_artifacts.records_path = MagicMock()
        mock_artifacts.records_path.exists.return_value = True
        mock_shard_artifacts.return_value = mock_artifacts

        mock_read_jsonl.return_value = [
            {"id": f"chunk_{j}", "embedding": [0.1] * 384} for j in range(100)
        ]

        ingest_shards(
            shard_stems=shard_stems,
            model_key="bge-small",
            profile_name="default",
            recreate_collection=False,
            resume=False,
            snapshot_interval=100,
        )

        assert mock_create_snapshot.call_count >= 3, (
            f"Expected at least 3 snapshots with 300 shards and interval 100, "
            f"got {mock_create_snapshot.call_count}"
        )

        manifest_calls = [
            c for c in mock_append_jsonl.call_args_list if "manifest" in str(c)
        ]
        assert len(manifest_calls) > 0, "Snapshot manifest should be appended to"

        logger.info(
            "✓ Periodic snapshots verified: %d snapshot calls", 
            mock_create_snapshot.call_count
        )

    @patch("indexing.ingestion.qdrant_client")
    @patch("indexing.ingestion.setup_collection")
    @patch("indexing.ingestion.get_collection_info")
    @patch("indexing.ingestion.get_qdrant_profile")
    @patch("indexing.ingestion.shard_artifacts")
    @patch("indexing.ingestion.read_jsonl")
    @patch("indexing.ingestion.append_jsonl")
    @patch("indexing.ingestion.build_points_from_shard_records")
    @patch("indexing.ingestion.create_collection_snapshot")
    def test_final_snapshot_created_always(
        self,
        mock_create_snapshot,
        mock_build_points,
        mock_append_jsonl,
        mock_read_jsonl,
        mock_shard_artifacts,
        mock_profile_fn,
        mock_collection_info,
        mock_setup_collection,
        mock_client_fn,
    ):      

        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client

        mock_profile = MagicMock(
            collection_name="test_collection", upsert_batch_size=1000
        )
        mock_profile_fn.return_value = mock_profile

        mock_build_points.return_value = []
        mock_create_snapshot.side_effect = ["snapshot-final"]

        shard_stems = [f"shard_{i:04d}" for i in range(50)]

        mock_artifacts = MagicMock()
        mock_artifacts.records_path = MagicMock()
        mock_artifacts.records_path.exists.return_value = True
        mock_shard_artifacts.return_value = mock_artifacts

        mock_read_jsonl.return_value = [
            {"id": f"chunk_{j}", "embedding": [0.1] * 384} for j in range(50)
        ]

        ingest_shards(
            shard_stems=shard_stems,
            model_key="bge-small",
            profile_name="default",
            recreate_collection=False,
            resume=False,
            snapshot_interval=100,
        )

        assert mock_create_snapshot.call_count >= 1, (
            "Final snapshot should be created"
        )

        logger.info("✓ Final snapshot created (call count: %d)", 
                    mock_create_snapshot.call_count)

    @patch("indexing.ingestion.qdrant_client")
    @patch("indexing.ingestion.setup_collection")
    @patch("indexing.ingestion.get_collection_info")
    @patch("indexing.ingestion.get_qdrant_profile")
    @patch("indexing.ingestion.shard_artifacts")
    @patch("indexing.ingestion.read_jsonl")
    @patch("indexing.ingestion.append_jsonl")
    @patch("indexing.ingestion.build_points_from_shard_records")
    @patch("indexing.ingestion.create_collection_snapshot")
    def test_custom_snapshot_interval_respected(
        self,
        mock_create_snapshot,
        mock_build_points,
        mock_append_jsonl,
        mock_read_jsonl,
        mock_shard_artifacts,
        mock_profile_fn,
        mock_collection_info,
        mock_setup_collection,
        mock_client_fn,
    ):

        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client

        mock_profile = MagicMock(
            collection_name="test_collection", upsert_batch_size=1000
        )
        mock_profile_fn.return_value = mock_profile

        mock_build_points.return_value = []
        mock_create_snapshot.side_effect = [f"snap-{i}" for i in range(10)]

        # 60 shards with interval of 20
        shard_stems = [f"shard_{i:04d}" for i in range(60)]

        mock_artifacts = MagicMock()
        mock_artifacts.records_path = MagicMock()
        mock_artifacts.records_path.exists.return_value = True
        mock_shard_artifacts.return_value = mock_artifacts

        mock_read_jsonl.return_value = [
            {"id": f"chunk_{j}", "embedding": [0.1] * 384} for j in range(50)
        ]

        ingest_shards(
            shard_stems=shard_stems,
            model_key="bge-small",
            profile_name="default",
            recreate_collection=False,
            resume=False,
            snapshot_interval=20,
        )

        assert mock_create_snapshot.call_count >= 3, (
            f"Expected at least 3 snapshots (at 20, 40, 60), "
            f"got {mock_create_snapshot.call_count}"
        )

        logger.info("✓ Custom interval=20 respected (calls: %d)", 
                    mock_create_snapshot.call_count)


class TestSnapshotMetadataTracking:

    @patch("indexing.ingestion.qdrant_client")
    @patch("indexing.ingestion.setup_collection")
    @patch("indexing.ingestion.get_collection_info")
    @patch("indexing.ingestion.get_qdrant_profile")
    @patch("indexing.ingestion.shard_artifacts")
    @patch("indexing.ingestion.read_jsonl")
    @patch("indexing.ingestion.append_jsonl")
    @patch("indexing.ingestion.build_points_from_shard_records")
    @patch("indexing.ingestion.create_collection_snapshot")
    def test_snapshot_metadata_records_shard_count(
        self,
        mock_create_snapshot,
        mock_build_points,
        mock_append_jsonl,
        mock_read_jsonl,
        mock_shard_artifacts,
        mock_profile_fn,
        mock_collection_info,
        mock_setup_collection,
        mock_client_fn,
    ):

        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client

        mock_profile = MagicMock(
            collection_name="test_collection", upsert_batch_size=1000
        )
        mock_profile_fn.return_value = mock_profile

        mock_build_points.return_value = []
        mock_create_snapshot.side_effect = [f"snap-{i}" for i in range(5)]

        shard_stems = [f"shard_{i:04d}" for i in range(15)]

        mock_artifacts = MagicMock()
        mock_artifacts.records_path = MagicMock()
        mock_artifacts.records_path.exists.return_value = True
        mock_shard_artifacts.return_value = mock_artifacts

        mock_read_jsonl.return_value = [
            {"id": f"chunk_{j}", "embedding": [0.1] * 384} for j in range(100)
        ]

        ingest_shards(
            shard_stems=shard_stems,
            model_key="bge-small",
            profile_name="default",
            recreate_collection=False,
            resume=False,
            snapshot_interval=5,
        )

        manifest_calls = [
            c for c in mock_append_jsonl.call_args_list if "manifest" in str(c)
        ]

        assert len(manifest_calls) > 0, "Manifest entries should exist"

        for call_obj in manifest_calls:
            args, kwargs = call_obj
            if len(args) >= 2:
                data = args[1]
                if isinstance(data, dict) and "shard_count" in data:
                    logger.info(f"✓ Snapshot metadata includes shard_count: {data}")
                    return

        pytest.fail("No manifest entry with shard_count found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
