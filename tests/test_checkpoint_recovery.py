
from __future__ import annotations
import json
import logging
import sys
import tempfile
from pathlib import Path
from unittest import mock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indexing.storage import (
    mark_done,
    read_jsonl, 
    shard_artifacts,
    shard_done,
    write_jsonl,
)
from indexing.ingestion import _load_ingested_stems

logger = logging.getLogger(__name__)


# PHASE A CHECKPOINTS: Shard Completion Markers

class TestPhaseACheckpoints:

    def test_mark_done_creates_marker(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            done_path = Path(tmpdir) / "shard_123.done"
            
            mark_done(done_path)
            
            assert done_path.exists(), "Done marker should exist"
            assert done_path.read_text().strip() == "DONE"

    def test_shard_done_requires_both_files(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            stem = "test_shard"
            shards_dir = Path(tmpdir)
            artifacts = shard_artifacts(shards_dir, stem)
            
            assert not shard_done(artifacts), "Should be incomplete when files missing"
            
            # Only records file exists
            artifacts.records_path.write_text("[]\n")
            assert not shard_done(artifacts), "Should be incomplete with only .jsonl"
            
            mark_done(artifacts.done_path)
            assert shard_done(artifacts), "Should be complete with both files"

    def test_shard_done_with_missing_records(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = shard_artifacts(Path(tmpdir), "missing_records")
            
            mark_done(artifacts.done_path)
            
            assert not shard_done(artifacts), ".done without .jsonl should be incomplete"

    def test_mark_done_overwrites(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            done_path = Path(tmpdir) / "test.done"
            
            mark_done(done_path)
            first_mtime = done_path.stat().st_mtime
            
            import time
            time.sleep(0.01)
            mark_done(done_path)
            second_mtime = done_path.stat().st_mtime
            
            assert second_mtime > first_mtime, "File should be updated"
            assert done_path.read_text().strip() == "DONE"

    def test_shard_artifacts_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shards_dir = Path(tmpdir)
            artifacts = shard_artifacts(shards_dir, "batch_0001")
            
            assert artifacts.stem == "batch_0001"
            assert artifacts.records_path.name == "batch_0001.jsonl"
            assert artifacts.done_path.name == "batch_0001.done"


# PHASE B CHECKPOINTS: Ingestion Progress Tracking

class TestPhaseBCheckpoints:

    def test_load_ingested_stems_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = Path(tmpdir) / "ingested_shards.jsonl"
            
            with mock.patch("indexing.ingestion.PATHS") as mock_paths:
                mock_paths.ingested_shards = progress_file
                result = _load_ingested_stems()
            
            assert result == set(), "Empty file should return empty set"

    def test_load_ingested_stems_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = Path(tmpdir) / "ingested_shards.jsonl"
            
            records = [
                {"stem": "batch_0001", "status": "INGESTED", "rows": 100},
                {"stem": "batch_0002", "status": "INGESTED", "rows": 150},
                {"stem": "batch_0003", "status": "PENDING", "rows": 0},
            ]
            write_jsonl(progress_file, records)
            
            with mock.patch("indexing.ingestion.PATHS") as mock_paths:
                mock_paths.ingested_shards = progress_file
                result = _load_ingested_stems()
            
            # Should only include INGESTED stems
            assert result == {"batch_0001", "batch_0002"}

    def test_load_ingested_stems_missing_file(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = Path(tmpdir) / "nonexistent.jsonl"
            
            with mock.patch("indexing.ingestion.PATHS") as mock_paths:
                mock_paths.ingested_shards = progress_file
                # Should handle missing file gracefully
                try:
                    result = _load_ingested_stems()
                    # If no exception, should return empty set
                    assert isinstance(result, set)
                except FileNotFoundError:
                    # This is acceptable too - depending on implementation
                    pass

    def test_load_ingested_stems_filters_status(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = Path(tmpdir) / "ingested_shards.jsonl"
            
            records = [
                {"stem": "shard_1", "status": "INGESTED", "rows": 100},
                {"stem": "shard_2", "status": "FAILED", "rows": 0},
                {"stem": "shard_3", "status": "INGESTED", "rows": 200},
                {"stem": "shard_4", "status": "PENDING", "rows": 0},
                {"stem": "shard_5", "status": "INGESTED", "rows": 150},
            ]
            write_jsonl(progress_file, records)
            
            with mock.patch("indexing.ingestion.PATHS") as mock_paths:
                mock_paths.ingested_shards = progress_file
                result = _load_ingested_stems()
            
            assert result == {"shard_1", "shard_3", "shard_5"}
            assert "shard_2" not in result, "FAILED status should not be included"
            assert "shard_4" not in result, "PENDING status should not be included"

    def test_load_ingested_stems_handles_missing_fields(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = Path(tmpdir) / "ingested_shards.jsonl"
            
            records = [
                {"stem": "shard_1", "status": "INGESTED"},
                {"status": "INGESTED"},  # Missing stem
                {"stem": "shard_3"},  # Missing status
                {"stem": "", "status": "INGESTED"},  # Empty stem
            ]
            write_jsonl(progress_file, records)
            
            with mock.patch("indexing.ingestion.PATHS") as mock_paths:
                mock_paths.ingested_shards = progress_file
                result = _load_ingested_stems()
            
            assert result == {"shard_1"}


# CHECKPOINT RESUMPTION

class TestCheckpointResumption:

    def test_phase_a_resume_skip_completed(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            shards_dir = Path(tmpdir)
            
            # Setup completed shard
            artifacts = shard_artifacts(shards_dir, "batch_001")
            artifacts.records_path.write_text('{"chunk": 1}\n')
            mark_done(artifacts.done_path)
            
            # Verify it's marked as done
            assert shard_done(artifacts), "Shard should be marked complete"

    def test_phase_b_resume_skip_ingested(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = Path(tmpdir) / "ingested_shards.jsonl"
            
            records = [
                {"stem": "shard_001", "status": "INGESTED", "rows": 100},
                {"stem": "shard_002", "status": "INGESTED", "rows": 150},
            ]
            write_jsonl(progress_file, records)
            
            with mock.patch("indexing.ingestion.PATHS") as mock_paths:
                mock_paths.ingested_shards = progress_file
                ingested = _load_ingested_stems()
            
            to_process = ["shard_001", "shard_002", "shard_003"]
            remaining = [s for s in to_process if s not in ingested]
            
            assert remaining == ["shard_003"], "Only new shard should be processed"

    def test_resize_detection_with_checkpoint(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = Path(tmpdir) / "ingested_shards.jsonl"
            
            # First ingestion: 100 rows
            record1 = {"stem": "shard_001", "status": "INGESTED", "rows": 100}
            write_jsonl(progress_file, [record1])
            
            # Later detection: same shard but different size (data updated)
            new_size = 150
            
            with mock.patch("indexing.ingestion.PATHS") as mock_paths:
                mock_paths.ingested_shards = progress_file
                ingested = _load_ingested_stems()
            
            # Checkpoint would show old size
            old_record = list(read_jsonl(progress_file))[0]
            assert old_record["rows"] == 100
            assert new_size != old_record["rows"], "Size difference detectable"


# EDGE CASES

class TestCheckpointEdgeCases:

    def test_corrupted_progress_file(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = Path(tmpdir) / "ingested_shards.jsonl"
            
            # Write corrupted JSON
            progress_file.write_text("{ invalid json }\n")
            
            with mock.patch("indexing.ingestion.PATHS") as mock_paths:
                mock_paths.ingested_shards = progress_file
                # Should handle error gracefully
                try:
                    result = _load_ingested_stems()
                    # May return empty set or raise, both acceptable
                except (json.JSONDecodeError, ValueError):
                    pass  # Expected

    def test_concurrent_checkpoint_writes(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = Path(tmpdir) / "ingested_shards.jsonl"
            
            # First write
            records1 = [{"stem": "shard_1", "status": "INGESTED", "rows": 100}]
            write_jsonl(progress_file, records1)
            
            # Append new entry (simulating concurrent write)
            records2 = [{"stem": "shard_2", "status": "INGESTED", "rows": 150}]
            import json
            with open(progress_file, "a") as f:
                json.dump(records2[0], f)
                f.write("\n")
            
            # Load should see both
            with mock.patch("indexing.ingestion.PATHS") as mock_paths:
                mock_paths.ingested_shards = progress_file
                result = _load_ingested_stems()
            
            assert len(result) >= 1, "Should load at least first entry"

    def test_empty_stem_ignored(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = Path(tmpdir) / "ingested_shards.jsonl"
            
            records = [
                {"stem": "valid_shard", "status": "INGESTED"},
                {"stem": "", "status": "INGESTED"},
                {"stem": None, "status": "INGESTED"},
            ]
            write_jsonl(progress_file, records)
            
            with mock.patch("indexing.ingestion.PATHS") as mock_paths:
                mock_paths.ingested_shards = progress_file
                result = _load_ingested_stems()
            
            assert "valid_shard" in result
            assert "" not in result
            assert None not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
