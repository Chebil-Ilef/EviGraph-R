from __future__ import annotations

import logging
from datetime import datetime, timezone

from config.settings import HF_INDEX_EXPORT, PATHS, get_qdrant_profile
from indexing.utils.storage import read_jsonl, shard_artifacts, write_json

logger = logging.getLogger(__name__)


def require_hf_index_export_config() -> None:
    missing: list[str] = []
    if not HF_INDEX_EXPORT.username:
        missing.append("INDEXING_HF_USERNAME")
    if not HF_INDEX_EXPORT.dense_dataset:
        missing.append("INDEXING_HF_DENSE_DATASET")
    if not HF_INDEX_EXPORT.sparse_dataset:
        missing.append("INDEXING_HF_SPARSE_DATASET")
    if not HF_INDEX_EXPORT.token:
        missing.append("HF_TOKEN")

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Missing required Hugging Face indexing settings in .env: "
            f"{joined}. These are mandatory for the indexing pipeline."
        )


def export_shard_indexes_to_hf(
    *,
    shard_stems: list[str],
    model_key: str,
    profile_name: str,
) -> dict[str, object]:
    require_hf_index_export_config()

    stats = summarize_shards(shard_stems)
    if stats["dense_rows"] == 0:
        raise RuntimeError("No dense vectors found in shard records; aborting Hugging Face export.")
    if stats["sparse_rows"] == 0:
        raise RuntimeError(
            "No sparse vectors found in shard records; cannot publish the sparse index dataset. "
            "Use a sparse-capable embedding model such as `bge-m3`."
        )

    dense_repo_id = resolve_repo_id(HF_INDEX_EXPORT.username, HF_INDEX_EXPORT.dense_dataset)
    sparse_repo_id = resolve_repo_id(HF_INDEX_EXPORT.username, HF_INDEX_EXPORT.sparse_dataset)
    profile = get_qdrant_profile(profile_name)
    generated_at = _now_iso()

    _push_dataset(
        repo_id=dense_repo_id,
        rows_fn=lambda: iter_dense_rows(shard_stems),
        card_text=build_dataset_card(
            repo_id=dense_repo_id,
            vector_kind="dense",
            model_key=model_key,
            profile_name=profile.profile,
            collection_name=profile.collection_name,
            shard_count=stats["shard_count"],
            row_count=stats["dense_rows"],
            generated_at=generated_at,
        ),
    )
    _push_dataset(
        repo_id=sparse_repo_id,
        rows_fn=lambda: iter_sparse_rows(shard_stems),
        card_text=build_dataset_card(
            repo_id=sparse_repo_id,
            vector_kind="sparse",
            model_key=model_key,
            profile_name=profile.profile,
            collection_name=profile.collection_name,
            shard_count=stats["shard_count"],
            row_count=stats["sparse_rows"],
            generated_at=generated_at,
        ),
    )

    metadata = {
        "generated_at": generated_at,
        "profile": profile.profile,
        "collection_name": profile.collection_name,
        "model_key": model_key,
        "split": HF_INDEX_EXPORT.split,
        "shard_count": stats["shard_count"],
        "dense_rows": stats["dense_rows"],
        "sparse_rows": stats["sparse_rows"],
        "dense_repo_id": dense_repo_id,
        "sparse_repo_id": sparse_repo_id,
    }
    write_json(PATHS.hf_export_metadata, metadata)
    logger.info("Published shard indexes to Hugging Face: %s", metadata)
    return metadata


def summarize_shards(shard_stems: list[str]) -> dict[str, int]:
    dense_rows = 0
    sparse_rows = 0

    for stem in shard_stems:
        for record in _iter_shard_records(stem):
            vectors = record.get("vectors") or {}
            if vectors.get("dense"):
                dense_rows += 1

            sparse = vectors.get("sparse") or {}
            if sparse.get("indices") and sparse.get("values"):
                sparse_rows += 1

    return {
        "shard_count": len(shard_stems),
        "dense_rows": dense_rows,
        "sparse_rows": sparse_rows,
    }


def iter_dense_rows(shard_stems: list[str]):
    for stem in shard_stems:
        for record in _iter_shard_records(stem):
            vectors = record.get("vectors") or {}
            dense = vectors.get("dense")
            if not dense:
                continue

            yield {
                **(record.get("payload") or {}),
                "chunk_uid": record.get("chunk_uid"),
                "dense_vector": dense,
            }


def iter_sparse_rows(shard_stems: list[str]):
    for stem in shard_stems:
        for record in _iter_shard_records(stem):
            vectors = record.get("vectors") or {}
            sparse = vectors.get("sparse") or {}
            indices = sparse.get("indices") or []
            values = sparse.get("values") or []
            if not indices or not values:
                continue

            yield {
                **(record.get("payload") or {}),
                "chunk_uid": record.get("chunk_uid"),
                "sparse_indices": indices,
                "sparse_values": values,
            }


def build_dataset_card(
    *,
    repo_id: str,
    vector_kind: str,
    model_key: str,
    profile_name: str,
    collection_name: str,
    shard_count: int,
    row_count: int,
    generated_at: str,
) -> str:
    vector_label = "Dense" if vector_kind == "dense" else "Sparse"
    vector_column = "dense_vector" if vector_kind == "dense" else "sparse_indices / sparse_values"
    tags = "dense-embeddings" if vector_kind == "dense" else "sparse-vectors"
    return f"""---
pretty_name: EviGraph-R {vector_label} Index
task_categories:
- feature-extraction
- text-retrieval
tags:
- evigraph-r
- scientific-literature
- qdrant
- {tags}
---

# EviGraph-R {vector_label} Index

This dataset contains the {vector_kind} retrieval index generated by the EviGraph-R indexing pipeline.
It is exported from the finalized shard records after the collection has been written to Qdrant, so the Hub copy matches the indexed corpus that was prepared for retrieval.

## What is inside

- One row per indexed chunk.
- Original chunk payload metadata used by retrieval and analysis.
- Vector columns: `{vector_column}`.
- Source collection: `{collection_name}`.
- Embedding model key: `{model_key}`.
- Runtime profile: `{profile_name}`.

## Build summary

- Repository: `{repo_id}`
- Split: `{HF_INDEX_EXPORT.split}`
- Shards exported: `{shard_count}`
- Rows exported: `{row_count}`
- Generated at: `{generated_at}`

## Suggested use

Use this dataset as a portable snapshot of the EviGraph-R retrieval index for reproducible experiments, offline analysis, or mirroring the vector store outside Qdrant.
"""


def resolve_repo_id(username: str, dataset_name: str) -> str:
    if "/" in dataset_name:
        return dataset_name
    return f"{username}/{dataset_name}"


def _push_dataset(*, repo_id: str, rows_fn, card_text: str) -> None:
    from datasets import Dataset
    from huggingface_hub import HfApi

    logger.info("Publishing Hugging Face dataset %s", repo_id)
    api = HfApi(token=HF_INDEX_EXPORT.token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

    dataset = Dataset.from_generator(rows_fn)
    dataset.push_to_hub(
        repo_id=repo_id,
        split=HF_INDEX_EXPORT.split,
        token=HF_INDEX_EXPORT.token,
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="README.md",
        path_or_fileobj=card_text.encode("utf-8"),
        commit_message="Update dataset card",
    )


def _iter_shard_records(stem: str):
    artifacts = shard_artifacts(PATHS.shards, stem)
    yield from read_jsonl(artifacts.records_path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
