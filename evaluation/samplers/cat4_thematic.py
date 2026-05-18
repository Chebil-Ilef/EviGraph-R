from __future__ import annotations
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from evaluation.config import (
    CAT4_SIM_MAX,
    CAT4_SIM_MIN,
    CATEGORY_TARGETS,
    DOMAINS,
    MIN_EMBED_TEXT_LEN,
    SCROLL_LIMIT,
    VECTOR_SEARCH_LIMIT,
    arxiv_category_to_domain,
)
from evaluation.samplers.base import ContextGroup, QdrantSamplerBase

logger = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


class Cat4Sampler(QdrantSamplerBase):

    def __init__(
        self,
        seed: int = 42,
        collection_name: str | None = None,
        dense_vector_name: str = "dense",
    ) -> None:
        super().__init__(collection_name=collection_name)
        self._rng = random.Random(seed)
        self._dense_vector_name = dense_vector_name

    def sample(self, target: int | None = None) -> list[ContextGroup]:
        target = target or CATEGORY_TARGETS[4]
        quota = max(1, target // len(DOMAINS))
        logger.info("[CAT4] target=%d  quota_per_domain=%d", target, quota)

        groups: list[ContextGroup] = []
        for domain in DOMAINS:
            domain_groups = self._sample_domain(domain, quota)
            groups.extend(domain_groups)
            logger.info("[CAT4] domain=%s  collected=%d", domain, len(domain_groups))

        self._rng.shuffle(groups)
        logger.info("[CAT4] Total groups: %d", len(groups))
        return groups

    def _sample_domain(self, domain: str, quota: int) -> list[ContextGroup]:
        from evaluation.config import DOMAIN_GROUPS
        prefixes = DOMAIN_GROUPS[domain]

        chunk_filter = Filter(
            must=[FieldCondition(key="chunk_type", match=MatchValue(value="subsection"))]
        )

        seen_seeds: set[str] = set()
        used_papers: set[str] = set()
        collected: list[ContextGroup] = []
        offset = None

        while len(collected) < quota:
            batch, next_offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=chunk_filter,
                limit=SCROLL_LIMIT,
                offset=offset,
                with_payload=True,
                with_vectors=True,  # need vector for similarity search
            )
            if not batch:
                break

            batch_list = list(batch)
            self._rng.shuffle(batch_list)

            for point in batch_list:
                if len(collected) >= quota:
                    break

                p = point.payload or {}
                cats = p.get("categories") or []
                if not self._matches_domain(cats, prefixes):
                    continue

                text_seed = p.get("embed_text", "")
                if len(text_seed) < MIN_EMBED_TEXT_LEN:
                    continue

                paper_seed = p.get("paper_id_arxiv") or ""
                chunk_seed_uid = p.get("chunk_uid") or str(point.id)

                if chunk_seed_uid in seen_seeds:
                    continue
                seen_seeds.add(chunk_seed_uid)

                # Get seed dense vector
                vec_seed = point.vector
                if isinstance(vec_seed, dict):
                    vec_seed = vec_seed.get(self._dense_vector_name)
                if vec_seed is None:
                    continue

                group = self._expand_seed(
                    paper_seed=paper_seed,
                    chunk_seed_uid=chunk_seed_uid,
                    text_seed=text_seed,
                    vec_seed=vec_seed,
                    domain=domain,
                    used_papers=used_papers,
                )
                if group is not None:
                    collected.append(group)
                    # Mark all three papers as used to ensure paper diversity
                    for pid in group.paper_ids:
                        used_papers.add(pid)

            if next_offset is None:
                break
            offset = next_offset

        return collected[:quota]

    def _expand_seed(
        self,
        paper_seed: str,
        chunk_seed_uid: str,
        text_seed: str,
        vec_seed: list[float],
        domain: str,
        used_papers: set[str],
    ) -> Optional[ContextGroup]:
    
        try:
            results = self._client.query_points(
                collection_name=self._collection,
                query=vec_seed,
                using=self._dense_vector_name,
                limit=VECTOR_SEARCH_LIMIT,
                with_payload=True,
                with_vectors=True,
            )
            neighbours = results.points
        except Exception as exc:
            logger.debug("[CAT4] Vector search failed for seed %s: %s", chunk_seed_uid, exc)
            return None

        from evaluation.config import DOMAIN_GROUPS
        prefixes = DOMAIN_GROUPS[domain]

        qualifying: list[dict] = []
        seen_papers_local: set[str] = {paper_seed}

        for nb in neighbours:
            if len(qualifying) >= 2:
                break

            nb_p = nb.payload or {}
            nb_pid = nb_p.get("paper_id_arxiv") or ""
            if not nb_pid or nb_pid in seen_papers_local or nb_pid in used_papers:
                continue

            nb_cats = nb_p.get("categories") or []
            if not self._matches_domain(nb_cats, prefixes):
                continue

            nb_text = nb_p.get("embed_text", "")
            if len(nb_text) < MIN_EMBED_TEXT_LEN:
                continue

            nb_vec = nb.vector
            if isinstance(nb_vec, dict):
                nb_vec = nb_vec.get(self._dense_vector_name)
            if nb_vec is None:
                continue

            sim = _cosine(vec_seed, nb_vec)
            if not (CAT4_SIM_MIN <= sim <= CAT4_SIM_MAX):
                continue

            qualifying.append({
                "paper_id": nb_pid,
                "chunk_uid": nb_p.get("chunk_uid") or str(nb.id),
                "embed_text": nb_text,
                "sim": sim,
            })
            seen_papers_local.add(nb_pid)

        if len(qualifying) < 2:
            return None

        nb1, nb2 = qualifying[0], qualifying[1]
        return ContextGroup(
            category=4,
            domain=domain,
            paper_ids=[paper_seed, nb1["paper_id"], nb2["paper_id"]],
            chunk_ids=[chunk_seed_uid, nb1["chunk_uid"], nb2["chunk_uid"]],
            texts=[text_seed, nb1["embed_text"], nb2["embed_text"]],
            metadata={
                "sim_seed_nb1": round(nb1["sim"], 4),
                "sim_seed_nb2": round(nb2["sim"], 4),
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

    parser = argparse.ArgumentParser(description="Sample Category-4 context groups")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=int, default=CATEGORY_TARGETS[4])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from config.settings import QDRANT_ACTIVE
    sampler = Cat4Sampler(
        seed=args.seed,
        dense_vector_name=QDRANT_ACTIVE.dense_vector_name,
    )
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
