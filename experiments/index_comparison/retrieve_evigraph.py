from __future__ import annotations
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@dataclass
class RetrievalResult:
    rank: int
    paper_id: str
    title: str
    score: float
    section_title: Optional[str] = None  # only available for EviGraph-R


class EviGraphRetriever:

    def __init__(self, top_k: int = 100):
        self.top_k = top_k
        self._retriever = None
        self._embedder = None

    def _load(self) -> None:
        import os
        from retrieval.retriever import HybridQueryRetriever
        from retrieval.embedder import Embedder
        from utils.qdrant import ensure_qdrant_runtime

        profile = os.getenv("INDEXING_PROFILE", "local").lower()
        logger.info("Ensuring Qdrant runtime (profile=%s) …", profile)
        ensure_qdrant_runtime(profile, startup_timeout=1800)

        logger.info("Loading EviGraph-R retriever (BGE-M3 + Qdrant) …")
        self._embedder = Embedder.from_model_key("bge-m3")
        self._retriever = HybridQueryRetriever(model_key="bge-m3")
        logger.info("  EviGraph-R retriever ready")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievalResult]:
        if self._retriever is None:
            self._load()

        k = top_k or self.top_k

        # BGE-M3 produces both dense and sparse in one forward pass
        bge_output = self._embedder.embed_query(query)
        dense = bge_output.dense.tolist()
        sparse = bge_output.sparse[0] if bge_output.sparse else {}

        chunks = self._retriever.retrieve(
            embeddings=dense,
            query_text=query,
            top_k=k,
            sparse_embeddings=sparse,
        )

        # Deduplicate by paper_id, keeping highest-scoring chunk per paper.
        # Qdrant returns chunks; without this the same paper can appear multiple
        # times in top-k, inflating Precision/MAP/NDCG.
        seen: dict[str, RetrievalResult] = {}
        for chunk in chunks:
            pid = chunk.paper_id
            if pid not in seen or chunk.score > seen[pid].score:
                seen[pid] = RetrievalResult(
                    rank=0,  # assigned below after dedup
                    paper_id=pid,
                    title=chunk.paper_title or "",
                    score=chunk.score,
                    section_title=chunk.section_title,
                )

        deduped = sorted(seen.values(), key=lambda r: r.score, reverse=True)
        for rank, r in enumerate(deduped):
            r.rank = rank + 1

        return deduped
