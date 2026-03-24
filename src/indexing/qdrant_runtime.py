from __future__ import annotations

import logging
import os
import shutil
import subprocess

try:
    from config.settings import PATHS, QDRANT_CONNECTION, QDRANT_RUNTIME
    from utils.qdrant import check_qdrant_alive, qdrant_client
except ModuleNotFoundError:
    from src.config.settings import PATHS, QDRANT_CONNECTION, QDRANT_RUNTIME
    from src.utils.qdrant import check_qdrant_alive, qdrant_client

logger = logging.getLogger(__name__)


def ensure_qdrant_runtime(profile: str) -> None:
    if profile == "local":
        _ensure_local_docker_qdrant()
    else:
        _validate_hpc_runtime()

    client = qdrant_client()
    check_qdrant_alive(client, profile=profile)


def _ensure_local_docker_qdrant() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for local Qdrant runs but is not installed.")

    container_name = QDRANT_RUNTIME.local_container_name
    image = QDRANT_RUNTIME.local_image

    status = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode == 0 and status.stdout.strip() == "true":
        return

    if status.returncode == 0:
        logger.info("Starting existing Qdrant container %s", container_name)
        subprocess.run(["docker", "start", container_name], check=True)
        return

    logger.info("Creating local Qdrant container %s", container_name)
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{QDRANT_CONNECTION.port}:6333",
            "-p",
            f"{QDRANT_CONNECTION.grpc_port}:6334",
            "-v",
            f"{PATHS.qdrant_storage.resolve()}:/qdrant/storage",
            "-v",
            f"{PATHS.qdrant_snapshots.resolve()}:/qdrant/snapshots",
            image,
        ],
        check=True,
    )


def _validate_hpc_runtime() -> None:
    container_tool = shutil.which("apptainer") or shutil.which("singularity")
    if container_tool is None:
        raise RuntimeError(
            "HPC profile requires Apptainer or Singularity on PATH. "
            "See documentation/hpc.md for the expected run command."
        )

    required_env = {
        "SINGULARITY_CACHEDIR": os.getenv("SINGULARITY_CACHEDIR"),
        "SINGULARITY_TMPDIR": os.getenv("SINGULARITY_TMPDIR"),
    }
    missing = [key for key, value in required_env.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing HPC container environment variables: "
            + ", ".join(missing)
            + ". See documentation/hpc.md for setup."
        )

    if not PATHS.qdrant_storage.exists():
        raise RuntimeError(f"Qdrant storage path does not exist: {PATHS.qdrant_storage}")
    if not PATHS.qdrant_snapshots.exists():
        raise RuntimeError(f"Qdrant snapshot path does not exist: {PATHS.qdrant_snapshots}")

    logger.info(
        "Validated HPC runtime. Start Qdrant with: %s run --bind %s:/qdrant/storage "
        "--bind %s:/qdrant/snapshots %s",
        container_tool,
        PATHS.qdrant_storage,
        PATHS.qdrant_snapshots,
        QDRANT_RUNTIME.hpc_sif_path,
    )
