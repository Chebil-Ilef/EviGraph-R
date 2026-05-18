from __future__ import annotations
import logging
import math
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
    CAT3_SIM_MAX,
    CAT3_SIM_MIN,
    CATEGORY_TARGETS,
    DOMAINS,
    MIN_EMBED_TEXT_LEN,
    SCROLL_LIMIT,
    VECTOR_SEARCH_LIMIT,
)
from evaluation.samplers.base import ContextGroup, QdrantSamplerBase

logger = logging.getLogger(__name__)

# Preferred sections to pick chunk B from (ordered by priority)
_PREFERRED_B_SECTIONS = [
    "methods", "methodology", "results", "experiments",
    "approach", "model", "system", "evaluation", "discussion",
]


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


class Cat3Sampler(QdrantSamplerBase):
   
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
        target = target or CATEGORY_TARGETS[3]
        quota = max(1, target // len(DOMAINS))
        logger.info("[CAT3] target=%d  quota_per_domain=%d", target, quota)

        groups: list[ContextGroup] = []
        for domain in DOMAINS:
            domain_groups = self._sample_domain(domain, quota)
            groups.extend(domain_groups)
            logger.info("[CAT3] domain=%s  collected=%d", domain, len(domain_groups))

        self._rng.shuffle(groups)
        logger.info("[CAT3] Total groups: %d", len(groups))
        return groups

    
    def _sample_domain(self, domain: str, quota: int) -> list[ContextGroup]:
        from evaluation.config import DOMAIN_GROUPS
        prefixes = DOMAIN_GROUPS[domain]

        # Scroll subsection chunks — filter by chunk_type only; domain check in Python
        chunk_filter = Filter(
            must=[FieldCondition(key="chunk_type", match=MatchValue(value="subsection"))]
        )

        seen_pairs: set[tuple[str, str]] = set()
        collected: list[ContextGroup] = []
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

            # Shuffle batch so we don't always bias toward the same papers
            batch_list = list(batch)
            self._rng.shuffle(batch_list)

            for point in batch_list:
                if len(collected) >= quota:
                    break

                p = point.payload or {}
                cats = p.get("categories") or []
                if not self._matches_domain(cats, prefixes):
                    continue

                text_a = p.get("embed_text", "")
                if len(text_a) < MIN_EMBED_TEXT_LEN:
                    continue

                paper_a = p.get("paper_id_arxiv") or ""
                chunk_a_uid = p.get("chunk_uid") or str(point.id)

                # Extract resolved arXiv cite_spans
                cited_arxiv_ids = self._extract_cited_arxiv(p)
                if not cited_arxiv_ids:
                    continue

                self._rng.shuffle(cited_arxiv_ids)
                for paper_b in cited_arxiv_ids:
                    if paper_b == paper_a:
                        continue
                    pair_key = (chunk_a_uid, paper_b)
                    if pair_key in seen_pairs:
                        continue

                    group = self._try_build_group(
                        paper_a=paper_a,
                        chunk_a_uid=chunk_a_uid,
                        text_a=text_a,
                        paper_b=paper_b,
                        domain=domain,
                        point_id=point.id,
                    )
                    seen_pairs.add(pair_key)

                    if group is not None:
                        collected.append(group)
                        break  # one group per chunk A

            if next_offset is None:
                break
            offset = next_offset

        return collected[:quota]

    def _try_build_group(
        self,
        paper_a: str,
        chunk_a_uid: str,
        text_a: str,
        paper_b: str,
        domain: str,
        point_id,
    ) -> Optional[ContextGroup]:
      
        # 1. Fetch chunk A's dense vector
        vec_a = self._fetch_point_vector(point_id)
        if vec_a is None:
            return None

        # 2. Find chunk B: dense search within paper B, ranked by query similarity to chunk A
        chunks_b = self._fetch_paper_chunks_with_vectors(paper_b)
        if not chunks_b:
            return None

        # Prefer Methods/Results sections; fall back to any
        chunks_b_sorted = self._sort_by_section_preference(chunks_b)

        for chunk_b in chunks_b_sorted:
            vec_b = chunk_b.get("vector")
            if vec_b is None:
                continue
            text_b = chunk_b.get("embed_text", "")
            if len(text_b) < MIN_EMBED_TEXT_LEN:
                continue

            sim = _cosine(vec_a, vec_b)
            if not (CAT3_SIM_MIN <= sim <= CAT3_SIM_MAX):
                continue

            return ContextGroup(
                category=3,
                domain=domain,
                paper_ids=[paper_a, paper_b],
                chunk_ids=[chunk_a_uid, chunk_b["chunk_uid"]],
                texts=[text_a, text_b],
                metadata={
                    "cosine_sim": round(sim, 4),
                    "section_b": chunk_b.get("section_title", ""),
                },
            )

        return None

    def _fetch_point_vector(self, point_id) -> Optional[list[float]]:
        try:
            result = self._client.retrieve(
                collection_name=self._collection,
                ids=[point_id],
                with_vectors=True,
                with_payload=False,
            )
            if not result:
                return None
            vectors = result[0].vector
            if isinstance(vectors, dict):
                return vectors.get(self._dense_vector_name)
            return vectors
        except Exception as exc:
            logger.debug("[CAT3] Vector fetch failed for id=%s: %s", point_id, exc)
            return None

    def _fetch_paper_chunks_with_vectors(self, paper_id: str) -> list[dict]:
        try:
            paper_filter = Filter(
                must=[
                    FieldCondition(key="paper_id_arxiv", match=MatchValue(value=paper_id)),
                    FieldCondition(key="chunk_type", match=MatchValue(value="subsection")),
                ]
            )
            points, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=paper_filter,
                limit=50,
                with_payload=True,
                with_vectors=True,
            )
            results = []
            for pt in points:
                p = pt.payload or {}
                text = p.get("embed_text", "")
                if len(text) < MIN_EMBED_TEXT_LEN:
                    continue
                vec = pt.vector
                if isinstance(vec, dict):
                    vec = vec.get(self._dense_vector_name)
                results.append({
                    "chunk_uid": p.get("chunk_uid") or str(pt.id),
                    "embed_text": text,
                    "section_title": (
                        p.get("imrad_section_title") or p.get("section_title") or ""
                    ).lower(),
                    "vector": vec,
                })
            return results
        except Exception as exc:
            logger.debug("[CAT3] Paper chunk fetch failed for %s: %s", paper_id, exc)
            return []

    @staticmethod
    def _sort_by_section_preference(chunks: list[dict]) -> list[dict]:
        def priority(c: dict) -> int:
            sec = c.get("section_title", "").lower()
            for i, pref in enumerate(_PREFERRED_B_SECTIONS):
                if pref in sec:
                    return i
            return len(_PREFERRED_B_SECTIONS)

        return sorted(chunks, key=priority)

    @staticmethod
    def _extract_cited_arxiv(payload: dict) -> list[str]:
        spans_blob = payload.get("spans") or {}
        cite_spans = spans_blob.get("cite_spans") or []
        ids: list[str] = []
        for cs in cite_spans:
            if not isinstance(cs, dict):
                continue
            arxiv_id = cs.get("paper_id_arxiv") or cs.get("arxiv_id") or ""
            if arxiv_id:
                ids.append(arxiv_id)
        return list(dict.fromkeys(ids))  # deduplicate, preserve order

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

    parser = argparse.ArgumentParser(description="Sample Category-3 context groups")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=int, default=CATEGORY_TARGETS[3])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from config.settings import QDRANT_ACTIVE
    sampler = Cat3Sampler(
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
