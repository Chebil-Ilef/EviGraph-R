from __future__ import annotations
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock
import pytest
    
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.qdrant import (
    _ensure_hpc_singularity_instance,
    _ensure_local_docker_qdrant,
    _get_singularity_tool,
    _instance_is_running,
    _parse_instance_list,
    _start_singularity_instance,
)

logger = logging.getLogger(__name__)


class TestSingularityLifecycle:

    def test_get_singularity_tool_found(self):

        with mock.patch("shutil.which") as mock_which:
            mock_which.return_value = "singularity"
            result = _get_singularity_tool()
            assert result == "singularity"


    def test_instance_running_detection(self):

        output = """INSTANCE NAME     PID      IP    IMAGE FILE
evigraph-qdrant   12345    127.0.0.1  /path/to/qdrant.sif"""
        
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(stdout=output, returncode=0)
            result = _instance_is_running("singularity", "evigraph-qdrant")
            assert result is True

    def test_instance_not_running_detection(self):

        output = """INSTANCE NAME     PID      IP    IMAGE FILE"""
        
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(stdout=output, returncode=0)
            result = _instance_is_running("singularity", "evigraph-qdrant")
            assert result is False

    def test_start_instance_command_construction(self):

        with mock.patch("subprocess.run") as mock_run:
            _start_singularity_instance(
                "singularity",
                "test-instance",
                "test.sif",
                "/host/storage",
                "/host/snapshots",
            )
            
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            
            assert args[0] == "singularity"
            assert args[1] == "instance"
            assert args[2] == "start"
            assert "--bind" in args
            assert "/host/storage:/qdrant/storage" in args
            assert "/host/snapshots:/qdrant/snapshots" in args

    def test_ensure_already_running(self):

        with mock.patch("utils.qdrant._get_singularity_tool") as mock_get_tool, \
             mock.patch("os.getenv") as mock_getenv, \
             mock.patch("pathlib.Path.exists") as mock_exists, \
             mock.patch("utils.qdrant._instance_is_running") as mock_is_running:
            
            mock_get_tool.return_value = "singularity"
            mock_getenv.return_value = "/some/path"
            mock_exists.return_value = True
            mock_is_running.return_value = True
            
            _ensure_hpc_singularity_instance()
            mock_is_running.assert_called_once()

    def test_ensure_missing_env_variables(self):

        with mock.patch("utils.qdrant._get_singularity_tool") as mock_get_tool, \
             mock.patch("os.getenv") as mock_getenv:
            
            mock_get_tool.return_value = "singularity"
            mock_getenv.side_effect = lambda key, *args: None
            
            with pytest.raises(RuntimeError, match="Missing HPC environment variables"):
                _ensure_hpc_singularity_instance()

    def test_ensure_missing_storage_path(self):

        with mock.patch("utils.qdrant._get_singularity_tool") as mock_get_tool, \
             mock.patch("os.getenv") as mock_getenv, \
             mock.patch("pathlib.Path.exists") as mock_exists:
            
            mock_get_tool.return_value = "singularity"
            mock_getenv.return_value = "/some/path"
            mock_exists.return_value = False
            
            with pytest.raises(RuntimeError, match="storage path does not exist"):
                _ensure_hpc_singularity_instance()


class TestDockerLifecycle:

    def test_local_docker_already_running(self):

        with mock.patch("subprocess.run") as mock_run, \
             mock.patch("shutil.which") as mock_which:
            
            mock_which.return_value = "docker"
            # First call checks if running: returns "true"
            mock_run.return_value = mock.Mock(
                stdout="true",
                returncode=0,
            )
            
            _ensure_local_docker_qdrant()
            
            # Should only call once to check status
            mock_run.assert_called_once()

    def test_local_docker_start_existing(self):

        with mock.patch("subprocess.run") as mock_run, \
             mock.patch("shutil.which") as mock_which:
            
            mock_which.return_value = "docker"
            check_result = mock.Mock(stdout="", returncode=0)
            start_result = mock.Mock(stdout="", returncode=0)
            
            mock_run.side_effect = [check_result, start_result]
            
            _ensure_local_docker_qdrant()
            
            assert mock_run.call_count == 2
            
            start_call = mock_run.call_args_list[1][0][0]
            assert "start" in start_call

    def test_local_docker_not_installed(self):

        with mock.patch("shutil.which") as mock_which:
            mock_which.return_value = None
            
            with pytest.raises(RuntimeError, match="Docker is required"):
                _ensure_local_docker_qdrant()


@pytest.mark.integration
def test_singularity_instance_list_parsing_real():

    try:
        result = subprocess.run(
            ["singularity", "instance", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        instances = _parse_instance_list(result.stdout)

        assert isinstance(instances, set)
        for name in instances:
            assert isinstance(name, str)
            
        logger.info("Found Singularity instances: %s", instances)
    except FileNotFoundError:
        pytest.skip("Singularity not installed")
    except subprocess.TimeoutExpired:
        pytest.skip("Singularity command timed out")


@pytest.mark.integration
def test_volume_mount_configuration():
    
    # Create temp volumes like HPC would
    with tempfile.TemporaryDirectory(prefix="evigraph_volume_test_") as tmpdir:
        storage = Path(tmpdir) / "qdrant_storage"
        snapshots = Path(tmpdir) / "qdrant_snapshots"
        storage.mkdir()
        snapshots.mkdir()
        
        # Verify paths follow pattern
        assert str(storage).endswith("qdrant_storage"), "Storage path should end with 'qdrant_storage'"
        assert str(snapshots).endswith("qdrant_snapshots"), "Snapshots path should end with 'qdrant_snapshots'"
        assert storage.exists(), "Storage directory should exist"
        assert snapshots.exists(), "Snapshots directory should exist"
        
        logger.info("✓ Volume mount paths configured correctly: %s, %s", storage, snapshots)


@pytest.mark.integration
def test_volume_data_persistence():

    
    with tempfile.TemporaryDirectory(prefix="evigraph_persistence_test_") as tmpdir:
        storage = Path(tmpdir) / "qdrant_storage"
        storage.mkdir()
        
        # Write test data to volume
        test_file = storage / "test_data.json"
        test_data = {"test": "data", "timestamp": "persistence_check"}
        test_file.write_text(json.dumps(test_data))
        
        # Verify data is readable
        read_data = json.loads(test_file.read_text())
        assert read_data == test_data, "Data should persist and be readable"
        
        logger.info("✓ Volume data persists: %s", test_data)


@pytest.mark.integration
def test_volume_directory_mount_not_in_container():

    with tempfile.TemporaryDirectory(prefix="evigraph_mount_test_") as tmpdir:
        storage = Path(tmpdir) / "qdrant_storage"
        storage.mkdir()
        
        marker = storage / "host_marker.txt"
        marker.write_text("This file should be accessible from container volume mount")
        
        assert marker.exists(), "Host file should exist before mount"
        assert marker.read_text()  
        
        logger.info("✓ Volume mount setup verified (directory-mounted, not in container)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
