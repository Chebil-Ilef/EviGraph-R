from __future__ import annotations
import logging
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qdrant_client.models import Filter, FieldCondition, MatchValue

from evaluation.config import (
    CATEGORY_TARGETS,
    DOMAINS,
    MIN_EMBED_TEXT_LEN,
    SCROLL_LIMIT,
    arxiv_category_to_domain,
    domain_quota,
)
from evaluation.samplers.base import ContextGroup, QdrantSamplerBase

logger = logging.getLogger(__name__)


class Cat1Sampler(QdrantSamplerBase):

    def __init__(self, seed: int = 42, collection_name: str | None = None) -> None:
        super().__init__(collection_name=collection_name)
        self._rng = random.Random(seed)

    def sample(self, target: int | None = None) -> list[ContextGroup]:
        target = target or CATEGORY_TARGETS[1]
        quota = max(1, target // len(DOMAINS))
        logger.info("[CAT1] target=%d  quota_per_domain=%d", target, quota)

        global_seen_papers: set[str] = set()
        groups: list[ContextGroup] = []
        for domain in DOMAINS:
            domain_groups = self._sample_domain(domain, quota, global_seen_papers)
            groups.extend(domain_groups)
            logger.info("[CAT1] domain=%s  collected=%d", domain, len(domain_groups))

        self._rng.shuffle(groups)
        logger.info("[CAT1] Total groups: %d", len(groups))
        return groups

    def _sample_domain(
        self, domain: str, quota: int, global_seen_papers: set[str] | None = None
    ) -> list[ContextGroup]:

        from evaluation.config import DOMAIN_GROUPS
        prefixes = DOMAIN_GROUPS[domain]

        # Scroll subsection chunks — no per-category Qdrant filter exists, so
        # we fetch by chunk_type and filter domain in Python.
        chunk_filter = Filter(
            must=[FieldCondition(key="chunk_type", match=MatchValue(value="subsection"))]
        )

        seen_papers: set[str] = global_seen_papers if global_seen_papers is not None else set()
        paper_chunks: dict[str, list[dict]] = {}

        offset = None
        collected_groups: list[ContextGroup] = []

        while len(collected_groups) < quota:
            batch, next_offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=chunk_filter,
                limit=SCROLL_LIMIT,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not batch:
                break

            for point in batch:
                p = point.payload or {}
                cats = p.get("categories") or []
                if not self._matches_domain(cats, prefixes):
                    continue

                pid = p.get("paper_id_arxiv") or ""
                if not pid or pid in seen_papers:
                    continue

                text = p.get("embed_text", "")
                if len(text) < MIN_EMBED_TEXT_LEN or not self.is_usable_chunk(text):
                    continue

                paper_chunks.setdefault(pid, []).append({
                    "chunk_uid": p.get("chunk_uid", ""),
                    "embed_text": text,
                    "chunk_index": p.get("chunk_index") or 0,
                    "total_chunks": p.get("total_chunks") or 1,
                    "categories": cats,
                })

            # Try to build groups from any paper that now has ≥ 3 chunks
            for pid, chunks in list(paper_chunks.items()):
                if pid in seen_papers:
                    continue
                if len(chunks) < 3:
                    continue
                group = self._build_group(pid, chunks, domain)
                if group is not None:
                    collected_groups.append(group)
                    seen_papers.add(pid)
                    del paper_chunks[pid]
                    if len(collected_groups) >= quota:
                        break

            if next_offset is None:
                break
            offset = next_offset

        return collected_groups[:quota]

    def _build_group(
        self, paper_id: str, chunks: list[dict], domain: str
    ) -> ContextGroup | None:

        chunks_sorted = sorted(chunks, key=lambda c: c["chunk_index"])
        n = len(chunks_sorted)
        if n < 3:
            return None

        # Tercile boundaries
        lo_end = max(1, n // 3)
        hi_start = n - max(1, n // 3)

        lo_chunks = chunks_sorted[:lo_end]
        mid_chunks = chunks_sorted[lo_end:hi_start]
        hi_chunks = chunks_sorted[hi_start:]

        if not mid_chunks:
            mid_chunks = [chunks_sorted[n // 2]]

        chosen = [
            self._rng.choice(lo_chunks),
            self._rng.choice(mid_chunks),
            self._rng.choice(hi_chunks),
        ]

        # Ensure 3 distinct chunks
        if len({c["chunk_uid"] for c in chosen}) < 3:
            return None

        return ContextGroup(
            category=1,
            domain=domain,
            paper_ids=[paper_id],
            chunk_ids=[c["chunk_uid"] for c in chosen],
            texts=[c["embed_text"] for c in chosen],
            metadata={
                "chunk_indices": [c["chunk_index"] for c in chosen],
                "total_chunks": chosen[0].get("total_chunks"),
            },
        )

    @staticmethod
    def _matches_domain(categories: list[str], prefixes: list[str]) -> bool:
        for cat in categories:
            for pfx in prefixes:
                if cat.lower().startswith(pfx.lower()):
                    return True
        return False


def main() -> None:
    import argparse, json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(description="Sample Category-1 context groups")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--target", type=int, default=CATEGORY_TARGETS[1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sampler = Cat1Sampler(seed=args.seed)
    groups = sampler.sample(target=args.target)
    sampler.close()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for g in groups:
            f.write(json.dumps({
                "category": g.category,
                "domain": g.domain,
                "paper_ids": g.paper_ids,
                "chunk_ids": g.chunk_ids,
                "texts": g.texts,
                "metadata": g.metadata,
            }) + "\n")
    print(f"Wrote {len(groups)} groups to {out}")


if __name__ == "__main__":
    main()
