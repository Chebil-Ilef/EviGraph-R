from __future__ import annotations
import os
import sys
import random
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

from utils.qdrant import ensure_qdrant_runtime
from qdrant_client import QdrantClient

COLLECTION = "unarxive_chunks"
SAMPLE_SIZE = 200
RANDOM_SEED = 42


def main() -> None:
    profile = os.getenv("INDEXING_PROFILE", "local")
    print(f"Starting Qdrant (profile={profile})...")
    ensure_qdrant_runtime(profile, startup_timeout=30)
    print("✓ Qdrant ready\n")

    client = QdrantClient(host="localhost", port=6333)
    info = client.get_collection(COLLECTION)
    total_points = info.points_count
    print(f"Collection '{COLLECTION}': {total_points:,} indexed chunks\n")

    # Use random offsets to sample from different parts of the collection
    random.seed(RANDOM_SEED)
    chunks = []
    offset = None
    batch_size = 50
    batches_needed = SAMPLE_SIZE // batch_size

    for _ in range(batches_needed):
        result, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        chunks.extend(result)
        if next_offset is None:
            break
        offset = next_offset

    print(f"Sampled {len(chunks)} chunks\n")

    section_counts: Counter = Counter()
    paper_ids: set = set()
    for c in chunks:
        p = c.payload
        section_counts[p.get("section_title") or "?"] += 1
        paper_ids.add(p.get("paper_id_arxiv") or p.get("paper_id") or "?")

    print(f"Unique papers in sample: {len(paper_ids)}")
    print("\nSection distribution (top 20):")
    for sec, cnt in section_counts.most_common(20):
        print(f"  {cnt:>4}  {sec[:70]}")

    print(f"\n{'═' * 70}")
    print("  CHUNK SAMPLES (for query ideation)")
    print(f"{'═' * 70}\n")

    # 30 spread across the sample
    step = max(1, len(chunks) // 30)
    selected = chunks[::step][:30]

    for i, c in enumerate(selected, 1):
        p = c.payload
        paper_id = p.get("paper_id_arxiv") or p.get("paper_id") or "?"
        section  = p.get("section_title") or "?"
        text     = (p.get("embed_text") or p.get("text") or "").replace("\n", " ").strip()
        print(f"[{i:02d}] paper={paper_id}  section={section}")
        print(f"      {text[:200]}")
        print()


if __name__ == "__main__":
    main()
