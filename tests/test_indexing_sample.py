
import logging
import sys
import subprocess
from pathlib import Path
import os
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config.settings import QDRANT_ACTIVE, PATHS, QDRANT_RUNTIME
from utils.qdrant import (
    ensure_qdrant_runtime,
    qdrant_client,
    setup_collection,
    get_collection_info,
    _get_singularity_tool,
    _instance_is_running,
    _get_running_instances,
)

logger = logging.getLogger(__name__)


class TestQdrantRuntimeSetup:

    def test_singularity_available(self):

        try:
            tool = _get_singularity_tool()
            assert tool in ["singularity"], f"Expected 'singularity', got {tool}"
            logger.info("✓ Singularity tool found: %s", tool)
        except RuntimeError as e:
            pytest.skip(f"Singularity not available: {e}")

    def test_qdrant_paths_exist(self):

        assert PATHS.qdrant_storage.exists(), f"Storage missing: {PATHS.qdrant_storage}"
        assert PATHS.qdrant_snapshots.exists(), f"Snapshots missing: {PATHS.qdrant_snapshots}"
        logger.info("✓ Storage: %s", PATHS.qdrant_storage)
        logger.info("✓ Snapshots: %s", PATHS.qdrant_snapshots)

    @pytest.mark.hpc
    def test_environment_variables_set(self):

        required = ["SINGULARITY_CACHEDIR", "SINGULARITY_TMPDIR"]
        for var in required:
            assert os.getenv(var), f"Missing env var: {var}"
            logger.info("✓ %s=%s", var, os.getenv(var))


class TestQdrantInstanceManagement:

    @pytest.mark.hpc
    def test_ensure_qdrant_runtime_starts_instance(self):

        logger.info("Ensuring Qdrant runtime (profile=%s)...", QDRANT_ACTIVE.profile)
        ensure_qdrant_runtime(QDRANT_ACTIVE.profile, startup_timeout=60)
        logger.info("✓ Qdrant runtime established")

    @pytest.mark.hpc
    def test_singularity_instance_is_running(self):

        try:
            tool = _get_singularity_tool()
            instance_name = QDRANT_RUNTIME.hpc_instance_name
            
            running = _instance_is_running(tool, instance_name)
            assert running, f"Instance {instance_name} not running"
            logger.info("✓ Instance running: %s", instance_name)
            
            # List all instances
            all_instances = _get_running_instances(tool)
            logger.info("  Available instances: %s", all_instances)
        except RuntimeError:
            pytest.skip("Singularity not available")

    @pytest.mark.integration
    def test_qdrant_client_connection(self):

        client = qdrant_client()
        collections = client.get_collections()
        logger.info("✓ Connected to Qdrant, collections count: %d", len(collections.collections))


class TestCollectionCreation:

    @pytest.mark.integration
    def test_setup_collection_creates_or_reuses(self):

        client = qdrant_client()
        model_key = "e5-base-v2"
        
        logger.info("Setting up collection (model=%s)...", model_key)
        setup_collection(client, model_key=model_key, profile=QDRANT_ACTIVE, recreate=False)
        logger.info("✓ Collection ready")

    @pytest.mark.integration
    def test_get_collection_info(self):

        client = qdrant_client()
        collection_name = QDRANT_ACTIVE.collection_name
        
        info = get_collection_info(client, collection_name)
        logger.info("Collection info: %s", info)
        
        assert info["collection_name"] == collection_name
        assert "points_count" in info
        assert "status" in info
        logger.info("✓ Collection info retrieved")


class TestIndexingPipeline:

    @pytest.mark.integration
    def test_run_indexing_on_sample(self):
        
        repo_dir = Path(__file__).resolve().parent.parent
        script = repo_dir / "scripts/run_indexing_capella.sh"
        
        if not script.exists():
            pytest.skip(f"Script not found: {script}")
        
        env = os.environ.copy()
        env["SAMPLE_SIZE"] = "1000"
        env["PHASE"] = "run"
        env["RESUME_FLAG"] = "--resume"
        
        logger.info("Running indexing pipeline (SAMPLE_SIZE=1000)...")
        logger.info("ℹRun this on HPC: SAMPLE_SIZE=1000 sbatch scripts/run_indexing_capella.sh")


class TestCollectionValidation:

    @pytest.mark.integration
    def test_collection_has_points(self):
        client = qdrant_client()
        collection_name = QDRANT_ACTIVE.collection_name
        
        info = get_collection_info(client, collection_name)
        points_count = info.get("points_count", 0)
        
        logger.info("Collection '%s' has %d points", collection_name, points_count)
        
        if points_count == 0:
            logger.warning("  Collection is empty - run indexing first!")
        else:
            logger.info("  ✓ Collection populated")

    @pytest.mark.integration
    def test_collection_schema_valid(self):

        client = qdrant_client()
        collection_name = QDRANT_ACTIVE.collection_name
        
        collection = client.get_collection(collection_name)
        logger.info("Collection schema:")
        logger.info("  Status: %s", collection.status)
        logger.info("  Vector count: %d", getattr(collection, "indexed_vectors_count", 0))
        
        # Check config
        if hasattr(collection, "config"):
            cfg = collection.config
            if hasattr(cfg, "vectors"):
                logger.info("  Vectors config: %s", cfg.vectors)
            if hasattr(cfg, "sparse_vectors"):
                logger.info("  Sparse vectors: %s", cfg.sparse_vectors)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    pytest.main([__file__, "-v", "-s", "--tb=short"])
