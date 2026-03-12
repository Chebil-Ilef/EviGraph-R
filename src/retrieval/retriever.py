import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, Document, SparseVector
from config.settings import (
    QDRANT_ACTIVE, QDRANT_CONNECTION, BENCHMARK, 
    DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:

    chunk_uid: str
    paper_id: str
    score: float
    embed_text: str
    section_title: Optional[str] = None
    chunk_type: Optional[str] = None
    chunk_index: Optional[int] = None     
    total_chunks: Optional[int] = None     
    spans: Optional[dict] = None          


class HybridQueryRetriever:
    """
    Hybrid retriever for two hybrid retrieval modes:
    
    Mode A — Dense + BM25 (normal embedding models):
        Uses dense embeddings + BM25 keyword search, fused with RRF.
        Applied to: e5, jina, qwen, etc.
    
    Mode B — Dense + Sparse Embeddings (BGE-M3 only):
        Uses dense + sparse embeddings from BGE-M3 model, fused with RRF.
        Applied to: bge-m3
    
    The mode is automatically determined from the model configuration.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        profile=None,
        model_key: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.profile = profile or QDRANT_ACTIVE
        self.model_key = model_key
        self.model_cfg = EMBEDDING_MODELS[model_key]
        self.use_bge_sparse = self.model_cfg.bge_produces_sparse
        
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
        logger.info(
            "Initialized HybridQueryRetriever → collection '%s' | model='%s' | mode='%s'",
            self.collection_name,
            self.model_key,
            "dense+sparse (BGE-M3)" if self.use_bge_sparse else "dense+bm25"
        )

    def retrieve(
        self,
        embeddings: List[float],
        query_text: str,
        top_k: int = 5,
        sparse_embeddings: Optional[dict] = None,
    ) -> List[ChunkResult]:
        """
        Retrieve top-k chunks using hybrid retrieval with RRF fusion.
        
        Returns:
            List of ChunkResult objects, ranked by RRF score
        """
        try:
            if self.use_bge_sparse:
                # Mode B — BGE-M3: Dense + Sparse Embeddings (from the model)
                if sparse_embeddings is None:
                    logger.warning(
                        "BGE-M3 requires sparse_embeddings but got None. "
                        "Falling back to dense-only search."
                    )
                    return self._retrieve_dense_only(embeddings, top_k)
                
                # Convert sparse dict to SparseVector format
                indices = list(sparse_embeddings.keys())
                values = [sparse_embeddings[i] for i in indices]
                sparse_vector = SparseVector(indices=indices, values=values)
                
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
                            query=sparse_vector,
                            using=self.profile.sparse_vector_name,
                            limit=BENCHMARK.sparse_top_k,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=top_k,
                )
                logger.debug("Retrieved via Dense+Sparse (BGE-M3) with RRF fusion")
            else:
                # Mode A — Normal models: Dense + BM25 (Reciprocal Rank Fusion)
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        Prefetch(
                            query=embeddings,
                            using=self.profile.dense_vector_name,  # named dense vector
                            limit=BENCHMARK.dense_top_k,
                        ),
                        Prefetch(
                            query=Document(
                                text=query_text,
                                model=self.profile.bm25_model,
                            ),
                            using=self.profile.sparse_vector_name,  # named sparse vector for BM25
                            limit=BENCHMARK.bm25_top_k,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=top_k,
                )
                logger.debug("Retrieved via Dense+BM25 with RRF fusion")

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
                    chunk_index=payload.get("chunk_index"),
                    total_chunks=payload.get("total_chunks"),
                    spans=payload.get("spans"),
                ))

            logger.debug(f"Retrieved {len(results)} results (RRF fused)")
            return results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            logger.error(f"  Collection: {self.collection_name}")
            logger.error(f"  Model: {self.model_key}")
            logger.error(f"  Mode: {'dense+sparse' if self.use_bge_sparse else 'dense+bm25'}")
            logger.error(f"  Endpoint: {getattr(self.client, 'url', getattr(self.client, '_host', 'unknown'))}")
            return []

    def _retrieve_dense_only(
        self, embeddings: List[float], top_k: int
    ) -> List[ChunkResult]:
        """Fallback: dense-only search (used only if sparse retrieval fails)."""
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=embeddings,
                limit=top_k,
                using=self.profile.dense_vector_name,
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
                    chunk_index=payload.get("chunk_index"),
                    total_chunks=payload.get("total_chunks"),
                    spans=payload.get("spans"),
                ))
            logger.debug(f"Retrieved {len(results)} results (dense-only fallback)")
            return results
        except Exception as e:
            logger.error(f"Dense-only fallback retrieval failed: {e}")
            return []