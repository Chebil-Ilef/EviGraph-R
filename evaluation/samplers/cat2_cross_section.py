from __future__ import annotations
import logging
import random
import sys
from collections import defaultdict
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
    EXCLUDED_SECTION_LABELS,
    MIN_EMBED_TEXT_LEN,
    SCROLL_LIMIT,
    is_valid_imrad_pair,
    section_label,
)
from evaluation.samplers.base import ContextGroup, QdrantSamplerBase

logger = logging.getLogger(__name__)


class Cat2Sampler(QdrantSamplerBase):

    def __init__(self, seed: int = 42, collection_name: str | None = None) -> None:
        super().__init__(collection_name=collection_name)
        self._rng = random.Random(seed)


    def sample(self, target: int | None = None) -> list[ContextGroup]:
        target = target or CATEGORY_TARGETS[2]
        quota = max(1, target // len(DOMAINS))
        logger.info("[CAT2] target=%d  quota_per_domain=%d", target, quota)

        global_seen_papers: set[str] = set()
        groups: list[ContextGroup] = []
        for domain in DOMAINS:
            domain_groups = self._sample_domain(domain, quota, global_seen_papers)
            groups.extend(domain_groups)
            logger.info("[CAT2] domain=%s  collected=%d", domain, len(domain_groups))

        self._rng.shuffle(groups)
        logger.info("[CAT2] Total groups: %d", len(groups))
        return groups

 
    def _sample_domain(
        self, domain: str, quota: int, global_seen_papers: set[str] | None = None
    ) -> list[ContextGroup]:
        from evaluation.config import DOMAIN_GROUPS
        prefixes = DOMAIN_GROUPS[domain]

        chunk_filter = Filter(
            must=[FieldCondition(key="chunk_type", match=MatchValue(value="subsection"))]
        )

        # paper_id → {section_label → [chunk_payload]}
        paper_sections: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        seen_papers: set[str] = global_seen_papers if global_seen_papers is not None else set()
        collected: list[ContextGroup] = []
        # Track how many groups of each pair type have been collected so far.
        # Cap each pair type at ceil(quota / num_pair_types) to force diversity.
        pair_type_counts: dict[frozenset, int] = defaultdict(int)
        from evaluation.config import VALID_IMRAD_PAIRS
        pair_cap = max(1, -(-quota // max(1, len(VALID_IMRAD_PAIRS))))  # ceil division

        offset = None
        while len(collected) < quota:
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

                sec = section_label(p)
                if sec in EXCLUDED_SECTION_LABELS:
                    continue

                text = p.get("embed_text", "")
                if len(text) < MIN_EMBED_TEXT_LEN or not self.is_usable_chunk(text):
                    continue

                paper_sections[pid][sec].append({
                    "chunk_uid": p.get("chunk_uid", ""),
                    "embed_text": text,
                    "section_label": sec,
                })

            # Try to build groups from papers that now have ≥ 2 valid sections
            for pid, sections in list(paper_sections.items()):
                if pid in seen_papers:
                    continue
                group = self._build_group(pid, sections, domain, pair_type_counts, pair_cap)
                if group is not None:
                    pair_key = frozenset({group.metadata["section_a"], group.metadata["section_b"]})
                    pair_type_counts[pair_key] += 1
                    collected.append(group)
                    seen_papers.add(pid)
                    del paper_sections[pid]
                    if len(collected) >= quota:
                        break

            if next_offset is None:
                break
            offset = next_offset

        return collected[:quota]

    def _build_group(
        self,
        paper_id: str,
        sections: dict[str, list[dict]],
        domain: str,
        pair_type_counts: dict,
        pair_cap: int,
    ) -> ContextGroup | None:
        sec_labels = list(sections.keys())

        # Find all valid section pairs that haven't hit their cap yet
        valid_pairs: list[tuple[str, str]] = []
        for i, a in enumerate(sec_labels):
            for b in sec_labels[i + 1:]:
                if not is_valid_imrad_pair(a, b):
                    continue
                if pair_type_counts[frozenset({a, b})] < pair_cap:
                    valid_pairs.append((a, b))

        if not valid_pairs:
            # Fall back to any valid pair if all caps are exhausted
            for i, a in enumerate(sec_labels):
                for b in sec_labels[i + 1:]:
                    if is_valid_imrad_pair(a, b):
                        valid_pairs.append((a, b))

        if not valid_pairs:
            return None

        self._rng.shuffle(valid_pairs)
        sec_a, sec_b = valid_pairs[0]
        chunk_a = self._rng.choice(sections[sec_a])
        chunk_b = self._rng.choice(sections[sec_b])

        if chunk_a["chunk_uid"] == chunk_b["chunk_uid"]:
            return None

        return ContextGroup(
            category=2,
            domain=domain,
            paper_ids=[paper_id],
            chunk_ids=[chunk_a["chunk_uid"], chunk_b["chunk_uid"]],
            texts=[chunk_a["embed_text"], chunk_b["embed_text"]],
            metadata={
                "section_a": sec_a,
                "section_b": sec_b,
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

    parser = argparse.ArgumentParser(description="Sample Category-2 context groups")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=int, default=CATEGORY_TARGETS[2])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sampler = Cat2Sampler(seed=args.seed)
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
