from __future__ import annotations
import json
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Iterable
from config.settings import HF_DATASET, PATHS
from indexing.utils.models import PipelineRunConfig, PreparedBatch
from indexing.utils.storage import ensure_directory, list_prepared_batches

logger = logging.getLogger(__name__)


def ensure_prepared_batches(config: PipelineRunConfig) -> list[PreparedBatch]:
    existing = list_prepared_batches(PATHS.batches, config.batches)
    if existing:
        logger.info("Using %d prepared batch file(s) from %s", len(existing), PATHS.batches)
        return _limit_batches(existing, config.sample_size)

    ensure_directory(PATHS.batches)
    logger.info(
        "Prepared batches not found. Materializing dataset=%s split=%s mode=%s",
        HF_DATASET.dataset_id,
        HF_DATASET.split,
        config.dataset_mode,
    )

    rows = _load_source_rows(config)
    count = materialize_batches(
        rows=rows,
        out_dir=PATHS.batches,
        batch_size=config.batch_size,
        sample_size=config.sample_size,
    )
    logger.info("Prepared %d paper(s) under %s", count, PATHS.batches)
    return list_prepared_batches(PATHS.batches, config.batches)


def materialize_batches(
    *,
    rows: Iterable[dict],
    out_dir: Path,
    batch_size: int,
    sample_size: int | None,
) -> int:
    ensure_directory(out_dir)
    written = 0
    current_index = -1
    handle = None

    try:
        for row in rows:
            if sample_size is not None and written >= sample_size:
                break

            paper = _extract_paper(row)
            if paper is None:
                continue

            batch_index = written // batch_size
            if batch_index != current_index:
                if handle is not None:
                    handle.close()
                current_index = batch_index
                stem = _batch_stem(batch_index)
                handle = (out_dir / f"{stem}.jsonl").open("w", encoding="utf-8")

            assert handle is not None
            handle.write(json.dumps(paper, ensure_ascii=False) + "\n")
            written += 1
    finally:
        if handle is not None:
            handle.close()

    return written


def _load_source_rows(config: PipelineRunConfig) -> Iterable[dict]:
    local_tar = Path(PATHS.data) / "unarxive_2024_full" / "unarXive_2024.tar.gz"
    if not local_tar.exists():
        _download_tar_from_hf(local_tar)
    logger.info("Loading from local tar file: %s", local_tar)
    return _load_from_tar(local_tar)


def _download_tar_from_hf(dest: Path) -> None:
    from huggingface_hub import hf_hub_download, list_repo_files

    ensure_directory(dest.parent)
    repo_id = HF_DATASET.dataset_id
    logger.info("Local tar not found — downloading from HuggingFace repo %s", repo_id)

    all_files = sorted(list_repo_files(repo_id, repo_type="dataset"))
    parts = [f for f in all_files if ".tar.gz.part_" in f]
    if not parts:
        # Repo stores a single tar
        parts = [f for f in all_files if f.endswith(".tar.gz")]
    if not parts:
        raise RuntimeError(f"No tar file(s) found in HF repo {repo_id}")

    part_paths = []
    for part in parts:
        logger.info("Downloading %s …", part)
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=part,
            repo_type="dataset",
            cache_dir=str(PATHS.hf_dataset_cache),
        )
        part_paths.append(cached)

    logger.info("Assembling %d part(s) → %s", len(part_paths), dest)
    tmp = dest.with_suffix(".tmp")
    try:
        with tmp.open("wb") as out:
            for p in part_paths:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out)
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    logger.info("✓ Dataset tar ready: %s (%.1f GB)", dest, dest.stat().st_size / 1e9)


def _load_from_tar(tar_path: Path) -> Iterable[dict]:

    with tarfile.open(tar_path, "r|gz") as tar:
        yielded = 0
        for member in tar:
            if not member.isfile() or not member.name.endswith(".jsonl"):
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            logger.info("Streaming %s from %s", member.name, tar_path.name)
            for line in extracted:
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue
                try:
                    yield json.loads(line_str)
                    yielded += 1
                    if yielded % 1000 == 0:
                        logger.info("Read %d paper(s) from %s", yielded, tar_path.name)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON in %s", member.name)


def _limit_batches(
    batches: list[PreparedBatch],
    sample_size: int | None,
) -> list[PreparedBatch]:
    if sample_size is None:
        return batches
    needed_batches = max(1, (sample_size + HF_DATASET.batch_size - 1) // HF_DATASET.batch_size)
    return batches[:needed_batches]


def _batch_stem(index: int) -> str:
    return f"batch_{index + 1:04d}"


def _extract_paper(row: dict) -> dict | None:
    # When loading from tar, rows are already JSON dicts - just pass through
    if isinstance(row, dict) and (row.get("paper_id") or row.get("abstract") or row.get("title") or row.get("content")):
        return row
    
    # When loading from HF datasets, extract from jsonl field
    if "jsonl" in row:
        first_line = next((line for line in str(row["jsonl"]).splitlines() if line.strip()), None)
        if not first_line:
            return None
        try:
            return json.loads(first_line)
        except json.JSONDecodeError:
            logger.warning("Failed to parse jsonl field")
            return None
    
    return row if isinstance(row, dict) and len(row) > 0 else None
