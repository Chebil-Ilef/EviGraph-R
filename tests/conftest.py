import pytest
from qdrant_client import QdrantClient


def pytest_collection_modifyitems(config, items):
    markexpr = getattr(config.option, "markexpr", "") or ""

    # Skip @pytest.mark.hpc tests unless the user explicitly targets them.
    if "hpc" not in markexpr:
        skip_hpc = pytest.mark.skip(
            reason="HPC tests require a Singularity compute node — run with: pytest -m hpc"
        )
        for item in items:
            if item.get_closest_marker("hpc"):
                item.add_marker(skip_hpc)

    # Skip @pytest.mark.integration tests when Qdrant is not reachable.
    try:
        client = QdrantClient(host="localhost", port=6333, timeout=2)
        client.get_collections()
        qdrant_up = True
    except Exception:
        qdrant_up = False

    if not qdrant_up:
        skip_integration = pytest.mark.skip(
            reason="Qdrant not reachable at localhost:6333 — start Docker locally or run in CI"
        )
        for item in items:
            if item.get_closest_marker("integration"):
                item.add_marker(skip_integration)
