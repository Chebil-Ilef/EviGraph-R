from __future__ import annotations
import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_REPO = Path("/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R")

# uv run scripts/backup_snapshots_to_s3.py --no-dry-run

@dataclass(frozen=True)
class S3Target:
    bucket: str
    prefix: str

    @property
    def uri(self) -> str:
        suffix = f"/{self.prefix}" if self.prefix else ""
        return f"s3://{self.bucket}{suffix}"

    def key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name


@dataclass(frozen=True)
class Config:
    repo: Path
    collection: str
    snapshots_dir: Path
    keep: int
    dry_run: bool
    env_file: Path
    target: S3Target
    endpoint_url: str | None
    profile: str | None


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_s3_uri(uri: str, collection: str) -> S3Target:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"S3_URI must look like s3://bucket/path, got: {uri!r}")
    base_prefix = parsed.path.strip("/")
    prefix = f"{base_prefix}/{collection}" if base_prefix else collection
    return S3Target(bucket=parsed.netloc, prefix=prefix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=None, help="Env file to load before reading settings.")
    parser.add_argument("--repo", default=None, help="Repository root.")
    parser.add_argument("--s3-uri", default=None, help="Base S3 URI, e.g. s3://bucket/evigraph/snapshots.")
    parser.add_argument("--collection", default=None, help="Qdrant collection name.")
    parser.add_argument("--snapshots-dir", default=None, help="Directory containing collection snapshot folders.")
    parser.add_argument("--keep", type=int, default=None, help="Number of newest S3 snapshots to retain.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without S3 writes.")
    parser.add_argument("--no-dry-run", action="store_true", help="Force real upload even if DRY_RUN=1 in env.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    repo_for_env = Path(args.repo or os.getenv("REPO", DEFAULT_REPO)).expanduser()
    env_file = Path(args.env_file or os.getenv("S3_ENV_FILE", repo_for_env / ".env.s3")).expanduser()
    load_env_file(env_file)

    repo = Path(args.repo or os.getenv("REPO", DEFAULT_REPO)).expanduser()
    collection = args.collection or os.getenv("COLLECTION", "unarxive_chunks")
    snapshots_dir = Path(args.snapshots_dir or os.getenv("SNAPSHOTS_DIR", repo / "snapshots")).expanduser()
    keep = args.keep if args.keep is not None else int(os.getenv("KEEP", "5"))
    dry_run = env_bool("DRY_RUN", False)
    if args.dry_run:
        dry_run = True
    if args.no_dry_run:
        dry_run = False

    s3_uri = args.s3_uri or os.getenv("S3_URI")
    if not s3_uri:
        raise ValueError("S3_URI is required. Set it in .env.s3 or pass --s3-uri.")
    if keep < 1:
        raise ValueError(f"KEEP must be at least 1, got: {keep}")

    return Config(
        repo=repo,
        collection=collection,
        snapshots_dir=snapshots_dir,
        keep=keep,
        dry_run=dry_run,
        env_file=env_file,
        target=parse_s3_uri(s3_uri, collection),
        endpoint_url=os.getenv("S3_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT_URL_S3") or None,
        profile=os.getenv("AWS_PROFILE") or None,
    )


def local_snapshots(collection_dir: Path) -> list[Path]:
    return sorted(
        collection_dir.glob("*.snapshot"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def require_boto3():
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for real S3 uploads. Install/sync dependencies with "
            "`uv sync`, or run with DRY_RUN=1 to preview."
        ) from exc
    return boto3, ClientError


def make_s3_client(config: Config):
    boto3, _ = require_boto3()
    session = boto3.Session(profile_name=config.profile) if config.profile else boto3.Session()
    return session.client("s3", endpoint_url=config.endpoint_url)


def object_exists(s3, client_error_type, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except client_error_type as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def upload_missing(config: Config, s3, client_error_type, snapshot_files: list[Path]) -> tuple[int, int]:
    uploaded = 0
    skipped = 0
    for snapshot_path in snapshot_files[: config.keep]:
        checksum_path = snapshot_path.with_name(f"{snapshot_path.name}.checksum")
        for path in [snapshot_path, checksum_path]:
            if not path.exists():
                continue
            key = config.target.key(path.name)
            uri = f"s3://{config.target.bucket}/{key}"
            if config.dry_run:
                print(f"+ upload {path} -> {uri}")
                if path == snapshot_path:
                    uploaded += 1
                continue
            if object_exists(s3, client_error_type, config.target.bucket, key):
                print(f"  exists: {path.name}")
                if path == snapshot_path:
                    skipped += 1
                continue
            print(f"+ upload {path} -> {uri}")
            s3.upload_file(str(path), config.target.bucket, key)
            if path == snapshot_path:
                uploaded += 1
    return uploaded, skipped


def list_remote_snapshots(config: Config, s3) -> list[dict]:
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=config.target.bucket, Prefix=f"{config.target.prefix}/")
    snapshots = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".snapshot"):
                snapshots.append({"key": key, "last_modified": obj["LastModified"]})
    return sorted(snapshots, key=lambda obj: obj["last_modified"], reverse=True)


def apply_retention(config: Config, s3) -> None:
    if config.dry_run:
        print("DRY_RUN=1: skipping remote listing/removal.")
        return

    remote_snapshots = list_remote_snapshots(config, s3)
    if len(remote_snapshots) <= config.keep:
        print(f"Remote already has {len(remote_snapshots)} snapshot(s); nothing to prune.")
        return

    for obj in remote_snapshots[config.keep :]:
        key = obj["key"]
        checksum_key = f"{key}.checksum"
        print(f"+ delete s3://{config.target.bucket}/{key}")
        s3.delete_object(Bucket=config.target.bucket, Key=key)
        print(f"+ delete s3://{config.target.bucket}/{checksum_key}")
        s3.delete_object(Bucket=config.target.bucket, Key=checksum_key)


def main() -> int:
    try:
        config = build_config(parse_args())
        collection_dir = config.snapshots_dir / config.collection
        if not collection_dir.is_dir():
            raise RuntimeError(f"Snapshot collection directory not found: {collection_dir}")

        snapshot_files = local_snapshots(collection_dir)
        print(f"Repository     : {config.repo}")
        print(f"Env file       : {config.env_file}")
        print(f"Local snapshots: {collection_dir}")
        print(f"S3 target      : {config.target.uri}")
        print(f"Retention      : newest {config.keep} snapshot(s)")
        print(f"Dry run        : {int(config.dry_run)}")
        print("")

        if not snapshot_files:
            print(f"No local .snapshot files found in {collection_dir}")
            return 0

        s3 = None
        client_error_type = Exception
        if not config.dry_run:
            s3 = make_s3_client(config)
            _, client_error_type = require_boto3()

        print("Uploading missing local snapshots...")
        uploaded, skipped = upload_missing(config, s3, client_error_type, snapshot_files)

        print("")
        print("Applying S3 retention...")
        apply_retention(config, s3)

        print("")
        print(f"S3 snapshot backup complete: uploaded={uploaded}, skipped_existing={skipped}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
