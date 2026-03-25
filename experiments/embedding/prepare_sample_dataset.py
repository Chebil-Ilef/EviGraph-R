import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.qdrant import qdrant_client


def export_qdrant_to_jsonl(
    collection_name: str,
    output_path: Path,
    host: str = "localhost",
    port: int = 6333,
) -> int:

    client = qdrant_client()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Scroll all points from collection
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=100000,  # large batch
    )
    
    logger.info("Exporting %d points from collection '%s'...", len(points), collection_name)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for point in points:
            chunk_uid = point.payload.get("chunk_uid", str(point.id))
            chunk_dict = {
                "paper_id": point.payload.get("paper_id_arxiv", ""),
                "uid": chunk_uid,
                "chunk_uid": chunk_uid,
                "section_title": point.payload.get("section_title", ""),
                "embed_text": point.payload.get("embed_text", ""),
            }
            f.write(json.dumps(chunk_dict) + "\n")
    
    logger.info("✓ Exported %d chunks to %s", len(points), output_path)
    return len(points)


def main():
    p = argparse.ArgumentParser(
        description="Export sample chunks from Qdrant for evaluation. "
        "First, run: uv run python -m src.indexing.indexing_pipeline --phase run --sample-size 1000"
    )
    p.add_argument(
        "--collection", default="ml_papers_chunks",
        help="Qdrant collection name.",
    )
    p.add_argument(
        "--output", type=Path, default=Path("experiments/embedding/data/chunks_sample.jsonl"),
        help="Output JSONL path.",
    )
    p.add_argument(
        "--host", default="localhost",
        help="Qdrant host.",
    )
    p.add_argument(
        "--port", type=int, default=6333,
        help="Qdrant port.",
    )
    args = p.parse_args()

    export_qdrant_to_jsonl(args.collection, args.output, args.host, args.port)
    logger.info("✓ Ready for evaluation!")


if __name__ == "__main__":
    main()
