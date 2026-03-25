from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS, PATHS, get_qdrant_profile
from indexing.models import PipelineRunConfig
from indexing.storage import (
    append_jsonl,
    iter_papers,
    list_shard_stems,
    mark_done,
    read_jsonl,
    shard_artifacts,
    shard_done,
    write_json,
    write_jsonl,
)
from indexing.ingestion import ingest_shards, write_snapshot_metadata
from utils.qdrant import ensure_qdrant_runtime

logger = logging.getLogger(__name__)


def run_pipeline(config: PipelineRunConfig) -> None:
    _write_run_metadata(config)

    prepared_batches = []
    if config.phase in {"prepare-dataset", "chunk", "embed", "run"}:
        ensure_prepared_batches = _load_dataset_preparer()
        prepared_batches = ensure_prepared_batches(config)

    if config.phase == "prepare-dataset":
        return

    if config.phase in {"chunk", "embed", "run"}:
        build_embedding_shards(config, prepared_batches)

    if config.phase in {"ingest", "run"}:
        shard_stems = _resolve_ingest_stems(config, prepared_batches)
        if not shard_stems:
            raise RuntimeError(
                f"No shard files found in {PATHS.shards}. "
                "Run the chunk phase first or point EVI_SHARDS_DIR at an existing shard directory."
            )
        ensure_qdrant_runtime(config.profile)
        ingest_shards(
            shard_stems=shard_stems,
            model_key=config.model_key,
            profile_name=config.profile,
            recreate_collection=config.recreate_collection,
            resume=config.resume,
        )

    if config.phase in {"snapshot", "run"}:
        ensure_qdrant_runtime(config.profile)
        snapshot_name = write_snapshot_metadata(config.profile)
        logger.info("Snapshot created: %s", snapshot_name)


def build_embedding_shards(config: PipelineRunConfig, prepared_batches) -> None:
    
    Embedder, build_paper_chunks, embed_result_to_serializable = _load_phase_a_ops()
    embedder = Embedder.from_model_key(config.model_key)
    model_cfg = EMBEDDING_MODELS[config.model_key]
    logger.info(
        "Embedder ready: %s dim=%d sparse=%s device=%s",
        config.model_key,
        model_cfg.dim,
        model_cfg.bge_produces_sparse,
        model_cfg.device,
    )

    remaining = config.sample_size
    for batch in prepared_batches:
        artifacts = shard_artifacts(PATHS.shards, batch.stem)
        if config.resume and shard_done(artifacts):
            logger.info("Skipping completed shard %s", batch.stem)
            if remaining is not None:
                remaining = max(0, remaining - _count_rows(batch.path))
                if remaining == 0:
                    break
            continue

        papers = list(iter_papers(batch))
        if remaining is not None:
            papers = papers[:remaining]
        if not papers:
            break

        logger.info("Building shard %s from %d paper(s)", batch.stem, len(papers))
        shard_records: list[dict] = []
        for index in range(0, len(papers), config.shard_batch_size):
            paper_batch = papers[index:index + config.shard_batch_size]
            chunks: list[dict] = []
            for paper in paper_batch:
                chunks.extend(build_paper_chunks(paper, model_key=config.model_key))

            if not chunks:
                continue

            vectors = embed_result_to_serializable(
                embedder.embed_passages([chunk["embed_text"] for chunk in chunks])
            )
            for chunk, vector_payload in zip(chunks, vectors):
                shard_records.append(
                    {
                        "chunk_uid": chunk["chunk_uid"],
                        "payload": chunk,
                        "vectors": vector_payload,
                    }
                )

        write_jsonl(artifacts.records_path, shard_records)
        mark_done(artifacts.done_path)
        append_jsonl(
            PATHS.shard_manifest,
            {
                "stem": batch.stem,
                "status": "DONE",
                "rows": len(shard_records),
                "source_papers": len(papers),
            },
        )

        if remaining is not None:
            remaining -= len(papers)
            if remaining <= 0:
                break


def _write_run_metadata(config: PipelineRunConfig) -> None:
    profile = get_qdrant_profile(config.profile)
    write_json(
        PATHS.run_metadata,
        {
            "phase": config.phase,
            "profile": config.profile,
            "dataset_mode": config.dataset_mode,
            "model_key": config.model_key,
            "sample_size": config.sample_size,
            "recreate_collection": config.recreate_collection,
            "resume": config.resume,
            "collection_name": profile.collection_name,
        },
    )


def _count_rows(path: Path) -> int:
    return sum(1 for _ in read_jsonl(path))


def _resolve_ingest_stems(config: PipelineRunConfig, prepared_batches) -> list[str]:
    if prepared_batches:
        return [batch.stem for batch in prepared_batches]

    shard_stems = list_shard_stems(PATHS.shards, config.batches)
    if config.sample_size is None:
        return shard_stems

    needed_shards = max(1, (config.sample_size + config.batch_size - 1) // config.batch_size)
    return shard_stems[:needed_shards]


def _load_dataset_preparer():
    try:
        from indexing.dataset import ensure_prepared_batches
    except ModuleNotFoundError:
        from src.indexing.dataset import ensure_prepared_batches
    return ensure_prepared_batches


def _load_phase_a_ops():
    try:
        from indexing.index_builder import build_paper_chunks
        from retrieval.embedder import Embedder
        from utils.qdrant import embed_result_to_serializable
    except ModuleNotFoundError:
        from src.indexing.index_builder import build_paper_chunks
        from src.retrieval.embedder import Embedder
        from src.utils.qdrant import embed_result_to_serializable
    return Embedder, build_paper_chunks, embed_result_to_serializable


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Restartable scholarly indexing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        default=["all"],
        metavar="STEM",
        help='Prepared batch stems to process, e.g. batch_0001 batch_0002, or "all".',
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        choices=list(EMBEDDING_MODELS),
        help="Embedding model key.",
    )
    parser.add_argument(
        "--phase",
        default="run",
        choices=["prepare-dataset", "chunk", "embed", "ingest", "snapshot", "run"],
        help="Pipeline phase to execute.",
    )
    parser.add_argument(
        "--dataset-mode",
        default="stream",
        choices=["stream", "mirror"],
        help="How to fetch the Hugging Face dataset when prepared batches are missing.",
    )
    parser.add_argument(
        "--profile",
        default=get_qdrant_profile(os.getenv("INDEXING_PROFILE", "local")).profile,
        choices=["local", "hpc"],
        help="Runtime profile for Qdrant and storage behavior.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional cap on the total number of papers to process.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing shards and ingestion checkpoints.")
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Drop and rebuild the Qdrant collection before ingestion.",
    )
    args = parser.parse_args()

    run_pipeline(
        PipelineRunConfig(
            phase=args.phase,
            profile=args.profile,
            dataset_mode=args.dataset_mode,
            model_key=args.model,
            sample_size=args.sample_size,
            recreate_collection=args.recreate_collection,
            resume=args.resume,
            batch_size=PATHS.dataset_batch_size,
            shard_batch_size=PATHS.shard_batch_size,
            batches=args.batches,
        )
    )
