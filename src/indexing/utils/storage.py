from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Optional
from indexing.utils.models import PreparedBatch, ShardArtifacts

def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_directory(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    ensure_directory(path.parent)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def read_json(path: Path, default: Optional[dict] = None) -> dict:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def list_prepared_batches(batches_dir: Path, stems: list[str]) -> list[PreparedBatch]:
    if stems == ["all"]:
        paths = sorted(batches_dir.glob("batch_*.jsonl"))
    else:
        paths = [batches_dir / f"{stem}.jsonl" for stem in stems]
    return [
        PreparedBatch(stem=path.stem, path=path)
        for path in paths
        if path.exists()
    ]


def list_shard_stems(shards_dir: Path, stems: list[str]) -> list[str]:
    if stems == ["all"]:
        paths = sorted(shards_dir.glob("*.jsonl"))
        return [path.stem for path in paths if path.is_file()]

    return [
        stem
        for stem in stems
        if (shards_dir / f"{stem}.jsonl").exists()
    ]


def iter_papers(batch: PreparedBatch) -> Iterator[dict]:
    yield from read_jsonl(batch.path)


def shard_artifacts(shards_dir: Path, stem: str) -> ShardArtifacts:
    ensure_directory(shards_dir)
    return ShardArtifacts(
        stem=stem,
        records_path=shards_dir / f"{stem}.jsonl",
        done_path=shards_dir / f"{stem}.done",
    )


def shard_done(artifacts: ShardArtifacts) -> bool:
    return artifacts.records_path.exists() and artifacts.done_path.exists()


def mark_done(done_path: Path) -> None:
    ensure_directory(done_path.parent)
    done_path.write_text("DONE\n", encoding="utf-8")
