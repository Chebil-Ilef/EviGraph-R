from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


DatasetMode = Literal["stream", "mirror"]
PhaseName = Literal["prepare-dataset", "chunk", "embed", "ingest", "snapshot", "run"]
ProfileName = Literal["local", "hpc"]


@dataclass(frozen=True)
class NormalizedSection:
    title: str
    text: str


@dataclass(frozen=True)
class NormalizedPaper:
    paper_id: str
    paper_doi: str
    meta: dict
    abstract_text: str
    sections: list[NormalizedSection]
    citations_by_ref: dict[str, dict]
    ref_captions: dict[str, str]


@dataclass(frozen=True)
class ChunkWindow:
    text: str
    cite_spans: list[dict]
    section_title: str
    chunk_type: str


@dataclass(frozen=True)
class PipelineRunConfig:
    phase: PhaseName
    profile: ProfileName
    dataset_mode: DatasetMode
    model_key: str
    sample_size: Optional[int]
    recreate_collection: bool
    resume: bool
    batch_size: int
    shard_batch_size: int
    batches: list[str] = field(default_factory=lambda: ["all"])


@dataclass(frozen=True)
class PreparedBatch:
    stem: str
    path: Path


@dataclass(frozen=True)
class ShardArtifacts:
    stem: str
    records_path: Path
    done_path: Path
