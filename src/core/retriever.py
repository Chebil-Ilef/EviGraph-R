import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, Document

from src.core.config import QDRANT_ACTIVE, QDRANT_CONNECTION, BENCHMARK

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:

    chunk_uid: str
    paper_id: str
    score: float
    embed_text: str
    section_title: Optional[str] = None
    chunk_type: Optional[str] = None


class UniversalQueryRetriever:
    """
    Universal retriever using Qdrant's query_points API.

    Combines dense (E5) and sparse (BM25) vectors via:
    - Parallel prefetch: Both searches run concurrently
    - Reciprocal Rank Fusion: Merges ranked results mathematically
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        profile=None,
    ):
        self.profile = profile or QDRANT_ACTIVE
        conn = QDRANT_CONNECTION
        if conn.url:
            self.client = QdrantClient(url=conn.url, api_key=conn.api_key)
        else:
            self.client = QdrantClient(
                host=conn.host,
                port=conn.port,
                grpc_port=conn.grpc_port,
                prefer_grpc=conn.prefer_grpc,
            )
        self.collection_name = collection_name or self.profile.collection_name
        logger.info("Initialized UniversalQueryRetriever → collection '%s'", self.collection_name)

    def retrieve(
        self,
        embeddings: List[float],
        query_text: str,
        top_k: int = 5,
    ) -> List[ChunkResult]:
        """
        Retrieve top-k chunks using hybrid dense + sparse search with RRF fusion.

        Args:
            embeddings: Query embedding vector (for dense search)
            query_text: Query text (for sparse BM25 search)
            top_k: Number of top results to return

        Returns:
            List of ChunkResult objects, ranked by RRF score
        """
        try:
            if self.profile.enable_sparse:
                # Hybrid: parallel dense + sparse prefetch with RRF fusion
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        Prefetch(
                            query=embeddings,
                            using=self.profile.dense_vector_name,
                            limit=BENCHMARK.dense_top_k,
                        ),
                        Prefetch(
                            query=Document(
                                text=query_text,
                                model=self.profile.bm25_model,
                            ),
                            using=self.profile.sparse_vector_name,
                            limit=BENCHMARK.bm25_top_k,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=top_k,
                )
            else:
                # Dense-only: single unnamed vector (no named-vector schema)
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=embeddings,
                    limit=top_k,
                )

            results = []
            for point in response.points:
                payload = point.payload
                results.append(ChunkResult(
                    chunk_uid=payload.get("chunk_uid"),
                    paper_id=payload.get("paper_id_arxiv"),
                    score=point.score,
                    embed_text=payload.get("embed_text", ""),
                    section_title=payload.get("section_title"),
                    chunk_type=payload.get("chunk_type"),
                ))

            logger.debug(f"Retrieved {len(results)} results (RRF fused)")
            return results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            logger.error(f"  Collection: {self.collection_name}")
            logger.error(f"  Endpoint: {getattr(self.client, 'url', getattr(self.client, '_host', 'unknown'))}")
            return []