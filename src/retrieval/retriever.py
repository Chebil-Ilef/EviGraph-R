import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, Document, SparseVector, Filter, FieldCondition, MatchAny, MatchText
from config.settings import (
    QDRANT_ACTIVE, QDRANT_CONNECTION, RETRIEVAL,
    DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS, RERANKER
)
from retrieval.embedder import BGEOutput

logger = logging.getLogger(__name__)

@dataclass
class ChunkResult:
    chunk_uid: str
    paper_id: str
    score: float
    embed_text: str
    section_title: Optional[str] = None
    paper_title: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    cite_spans: Optional[dict] = None



class HybridQueryRetriever:
    """
    Mode A : Dense + BM25 (normal embedding models):
        Uses dense embeddings + BM25 keyword search, fused with RRF.
        Applied to: e5, jina, qwen, etc.

    Mode B : Dense + Sparse Embeddings (BGE-M3 only):
        Uses dense + sparse embeddings from BGE-M3 model, fused with RRF.
        Applied to: bge-m3

    After reranking an optional section boost nudges chunks whose
    section_title matches the sub-query's target sections.  The boost
    magnitude is top_score * RETRIEVAL.section_boost_gamma (default 0.10).
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
            self.client = QdrantClient(url=conn.url, api_key=conn.api_key, timeout=conn.timeout)
        else:
            self.client = QdrantClient(
                host=conn.host,
                port=conn.port,
                grpc_port=conn.grpc_port,
                prefer_grpc=conn.prefer_grpc,
                timeout=conn.timeout,
            )
        self.collection_name = collection_name or self.profile.collection_name

        # cross-encoder reranker
        self.reranker = None
        if RERANKER.enabled:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(RERANKER.model_id, device=RERANKER.device)
                logger.info("[RETRIEVER] Loaded cross-encoder: %s", RERANKER.model_id)
            except Exception as e:
                logger.warning("[RETRIEVER] Cross-encoder init failed: %s. Reranking disabled.", e)

        logger.info(
            "[RETRIEVER] Initialized HybridQueryRetriever → collection '%s' | model='%s' | mode='%s' | reranker=%s",
            self.collection_name,
            self.model_key,
            "dense+sparse (BGE-M3)" if self.use_bge_sparse else "dense+bm25",
            "enabled" if self.reranker else "disabled"
        )

    def retrieve(
        self,
        embeddings: List[float],
        query_text: str,
        top_k: int = 5,
        sparse_embeddings: Optional[dict] = None,
        target_sections: Optional[List[str]] = None,
    ) -> List[ChunkResult]:
        # fetch more candidates if reranking is enabled
        fetch_limit = RERANKER.top_n if self.reranker else top_k

        try:
            if self.use_bge_sparse:
                # Mode B : BGE-M3: Dense + Sparse Embeddings (from the model)
                if sparse_embeddings is None:
                    logger.warning(
                        "BGE-M3 requires sparse_embeddings but got None. "
                        "Falling back to dense-only search."
                    )
                    return self._retrieve_dense_only(embeddings, top_k)

                indices = list(sparse_embeddings.keys())
                values = [sparse_embeddings[i] for i in indices]
                sparse_vector = SparseVector(indices=indices, values=values)

                # hybrid: parallel dense + sparse prefetch with RRF fusion
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        Prefetch(
                            query=embeddings,
                            using=self.profile.dense_vector_name,
                            limit=RETRIEVAL.dense_top_k,
                        ),
                        Prefetch(
                            query=sparse_vector,
                            using=self.profile.sparse_vector_name,
                            limit=RETRIEVAL.sparse_top_k,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=fetch_limit,
                )
                logger.debug("[RETRIEVER] Retrieved via Dense+Sparse (BGE-M3) with RRF fusion")
            else:
                # Mode A : Normal models: Dense + BM25 (Reciprocal Rank Fusion)
                try:
                    response = self.client.query_points(
                        collection_name=self.collection_name,
                        prefetch=[
                            Prefetch(
                                query=embeddings,
                                using=self.profile.dense_vector_name,
                                limit=RETRIEVAL.dense_top_k,
                            ),
                            Prefetch(
                                query=Document(
                                    text=query_text,
                                    model=self.profile.bm25_model,
                                ),
                                using=self.profile.sparse_vector_name,
                                limit=RETRIEVAL.bm25_top_k,
                            ),
                        ],
                        query=FusionQuery(fusion=Fusion.RRF),
                        limit=fetch_limit,
                    )
                    logger.debug("[RETRIEVER] Retrieved via Dense+BM25 with RRF fusion")
                except Exception as bm25_exc:
                    if "Not existing vector name" in str(bm25_exc):
                        logger.debug(
                            "[RETRIEVER] No sparse vector in collection for model '%s'; "
                            "using dense-only retrieval.",
                            self.model_key,
                        )
                        return self._retrieve_dense_only(embeddings, top_k)
                    raise

            results = []
            for point in response.points:
                payload = point.payload or {}
                results.append(ChunkResult(
                    chunk_uid=payload.get("chunk_uid") or "",
                    paper_id=payload.get("paper_id_arxiv") or "",
                    score=point.score,
                    embed_text=payload.get("embed_text", ""),
                    section_title=payload.get("imrad_section_title") or payload.get("section_title"),
                    paper_title=" ".join((payload.get("title") or "").split()) or None,
                    chunk_index=payload.get("chunk_index"),
                    total_chunks=payload.get("total_chunks"),
                    cite_spans=payload.get("spans"),  # map spans → cite_spans
                ))

            logger.debug(
                "[RETRIEVER][RRF] %d candidates before reranking (target_sections=%s)",
                len(results), target_sections
            )

            # cross-encoder reranking if enabled
            if self.reranker and len(results) > 0:
                results = self._rerank(query_text, results, top_k)
            else:
                results = results[:top_k]

            # soft section boost after reranking (does not filter, only nudges scores)
            if target_sections and RETRIEVAL.section_boost_gamma > 0.0:
                results = self._apply_section_boost(results, target_sections)

            return results

        except Exception as e:
            logger.error(f"[RETRIEVER] Retrieval failed: {e}")
            logger.error(f"  Collection: {self.collection_name}")
            logger.error(f"  Model: {self.model_key}")
            logger.error(f"  Mode: {'dense+sparse' if self.use_bge_sparse else 'dense+bm25'}")
            logger.error(f"  Endpoint: {getattr(self.client, 'url', getattr(self.client, '_host', 'unknown'))}")
            return []

    def _retrieve_dense_only(
        self, embeddings: List[float], top_k: int, query_filter: Optional[Filter] = None
    ) -> List[ChunkResult]:

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=embeddings,
                limit=top_k,
                using=self.profile.dense_vector_name,
                query_filter=query_filter,
            )
            results = []
            for point in response.points:
                payload = point.payload or {}
                results.append(ChunkResult(
                    chunk_uid=payload.get("chunk_uid") or "",
                    paper_id=payload.get("paper_id_arxiv") or "",
                    score=point.score,
                    embed_text=payload.get("embed_text", ""),
                    section_title=payload.get("imrad_section_title") or payload.get("section_title"),
                    paper_title=" ".join((payload.get("title") or "").split()) or None,
                    chunk_index=payload.get("chunk_index"),
                    total_chunks=payload.get("total_chunks"),
                    cite_spans=payload.get("spans"),
                ))
            logger.debug(f"[RETRIEVER] Retrieved {len(results)} results (dense-only fallback)")
            return results
        except Exception as e:
            logger.error(f"[RETRIEVER] Dense-only fallback retrieval failed: {e}")
            return []

    def _rerank(self, query: str, results: List[ChunkResult], top_k: int) -> List[ChunkResult]:

        if not results or self.reranker is None:
            return results

        # query-document pairs for cross-encoder
        pairs = [[query, r.embed_text] for r in results]

        # score with cross-encoder in batches
        scores = self.reranker.predict(pairs, batch_size=RERANKER.batch_size, show_progress_bar=False)

        # update scores and re-sort
        for i, score in enumerate(scores):
            results[i].score = float(score)

        # filter by minimum score threshold
        filtered_results = [r for r in results if r.score >= RERANKER.min_score_threshold]
        filtered_results.sort(key=lambda x: x.score, reverse=True)

        logger.debug(
            "[RETRIEVER][RERANK] cross-encoder scored %d → kept %d (threshold=%.2f) → top %d",
            len(results), len(filtered_results), RERANKER.min_score_threshold, top_k
        )

        return filtered_results[:top_k]

    def retrieve_hop_chunks(
        self,
        arxiv_id: str = "",
        doi: str = "",
        openalex_id: str = "",
        look_for: str = "",
        embedder=None,
        top_k: int = 2,
        citation_raw: str = "",
    ) -> List[ChunkResult]:
   

        if not look_for or not embedder:
            return []

        # At least one indexable ID must be provided (openalex_id is NOT stored in chunk payload)
        if not arxiv_id and not doi:
            logger.warning(
                "[RETRIEVER][HOP] No indexable paper ID provided (arxiv=%s doi=%s openalex=%s). "
                "Skipping hop — cannot filter without arxiv_id or doi.",
                arxiv_id or "None", doi or "None", openalex_id or "None",
            )
            return []

        try:
            query_vector = embedder.embed_query(look_for)
        except Exception as exc:
            logger.warning("[RETRIEVER][HOP] Embed failed for look_for=%r: %s", look_for, exc)
            return []

        # Handle BGEOutput from BGE embedder (multi-vector retrieval)
        if isinstance(query_vector, BGEOutput):
            query_vector = query_vector.dense

        # Convert numpy array to list if needed
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        # Build ID filter using only fields stored in chunk payload:
        #   paper_id_arxiv → from paper_id in unarXive schema
        #   paper_doi      → from metadata.doi in unarXive schema
        # NOTE: openalex_id is NOT stored in chunk payloads.
        if arxiv_id and doi:
            id_filter = Filter(
                should=[
                    FieldCondition(key="paper_id_arxiv", match=MatchAny(any=[arxiv_id])),
                    FieldCondition(key="paper_doi", match=MatchAny(any=[doi])),
                ]
            )
        elif arxiv_id:
            id_filter = Filter(
                must=[FieldCondition(key="paper_id_arxiv", match=MatchAny(any=[arxiv_id]))]
            )
        else:
            id_filter = Filter(
                must=[FieldCondition(key="paper_doi", match=MatchAny(any=[doi]))]
            )

        logger.info(
            "[RETRIEVER][HOP] ID-filter search: arxiv=%s doi=%s look_for=%r top_k=%d",
            arxiv_id or "None", doi or "None", look_for, top_k,
        )
        results = self._retrieve_dense_only(query_vector, top_k, query_filter=id_filter)

        if results:
            logger.info(
                "[RETRIEVER][HOP] ID-filter hit: arxiv=%s doi=%s → %d chunk(s)",
                arxiv_id or "None", doi or "None", len(results),
            )
            return results

        # ID filter returned 0 results: run diagnostics then try title fallback 

        paper_exists = False
        paper_chunk_count = 0
        try:
            field_map = [("paper_id_arxiv", arxiv_id), ("paper_doi", doi)]
            for field, val in field_map:
                if not val:
                    continue
                check_filter = Filter(must=[FieldCondition(key=field, match=MatchAny(any=[val]))])
                check_resp = self.client.count(
                    collection_name=self.collection_name,
                    count_filter=check_filter,
                    exact=False,
                )
                paper_chunk_count = check_resp.count
                if paper_chunk_count > 0:
                    paper_exists = True
                    logger.warning(
                        "[RETRIEVER][HOP] ID-filter: paper EXISTS (%s=%r → %d chunk(s)) "
                        "but dense similarity returned 0 results for look_for=%r. "
                        "Possible cause: look_for query semantically misaligned with paper content.",
                        field, val, paper_chunk_count, look_for,
                    )
                    break
            if not paper_exists:
                logger.warning(
                    "[RETRIEVER][HOP] ID-filter: paper NOT FOUND in collection "
                    "(arxiv=%s doi=%s). Will attempt title fallback if citation_raw available.",
                    arxiv_id or "None", doi or "None",
                )
        except Exception as diag_exc:
            logger.warning(
                "[RETRIEVER][HOP] Diagnostic count query failed (arxiv=%s doi=%s): %s",
                arxiv_id or "None", doi or "None", diag_exc,
            )

        # If paper exists but similarity was too low, no point in title fallback
        if paper_exists:
            logger.info(
                "[RETRIEVER][HOP] Skipping title fallback — paper IS indexed (%d chunks) "
                "but look_for query didn't match. Consider rephrasing look_for.",
                paper_chunk_count,
            )
            return []

        # Title fallback: paper not found by ID → search by title keyword
        if not citation_raw:
            logger.warning(
                "[RETRIEVER][HOP] Paper not found by ID and no citation_raw provided — "
                "cannot attempt title fallback.",
            )
            return []

        title_results = self._retrieve_by_title_fallback(
            query_vector=query_vector,
            citation_raw=citation_raw,
            look_for=look_for,
            top_k=top_k,
        )
        if title_results:
            logger.info(
                "[RETRIEVER][HOP] Title fallback SUCCESS: citation_raw=%r → %d chunk(s) "
                "(paper_id=%s)",
                citation_raw[:80], len(title_results),
                title_results[0].paper_id if title_results else "?",
            )
        else:
            logger.warning(
                "[RETRIEVER][HOP] Title fallback returned 0 results for citation_raw=%r. "
                "Paper is likely not in the indexed corpus.",
                citation_raw[:80],
            )
        return title_results

    def _retrieve_by_title_fallback(
        self,
        query_vector: List[float],
        citation_raw: str,
        look_for: str,
        top_k: int,
    ) -> List[ChunkResult]:
       
        # Use citation_raw directly as a title hint — strip author prefix heuristically:
        # format is often "Author et al. Title. Venue Year." — take everything after first period
        parts = citation_raw.split(".", 1)
        title_hint = (parts[1].strip() if len(parts) > 1 else citation_raw).strip()
        # Trim trailing venue/year (last segment after final period)
        title_parts = title_hint.rsplit(".", 1)
        title_hint = title_parts[0].strip() if len(title_parts) > 1 else title_hint
        # Cap length for Qdrant text match
        title_hint = title_hint[:120]

        if not title_hint:
            logger.warning("[RETRIEVER][HOP][TITLE] Empty title_hint extracted from citation_raw=%r", citation_raw[:80])
            return []

        logger.info(
            "[RETRIEVER][HOP][TITLE] Trying title fallback: title_hint=%r look_for=%r",
            title_hint, look_for,
        )

        try:
            title_filter = Filter(
                must=[FieldCondition(key="title", match=MatchText(text=title_hint))]
            )
            results = self._retrieve_dense_only(query_vector, top_k, query_filter=title_filter)
            logger.info(
                "[RETRIEVER][HOP][TITLE] title=%r → %d chunk(s)",
                title_hint, len(results),
            )
            return results
        except Exception as exc:
            logger.warning(
                "[RETRIEVER][HOP][TITLE] Title filter search failed for title_hint=%r: %s",
                title_hint, exc,
            )
            return []

    @staticmethod
    def deduplicate_chunks(chunks: List[ChunkResult]) -> List[ChunkResult]:

        best: dict[tuple, ChunkResult] = {}
        for chunk in chunks:
            key = (chunk.paper_id, chunk.chunk_uid)
            if key not in best or chunk.score > best[key].score:
                best[key] = chunk
        return list(best.values())

    def _apply_section_boost(
        self, results: List[ChunkResult], target_sections: List[str]
    ) -> List[ChunkResult]:
        
        # Boost scores of chunks whose section_title is in target_sections.
        # Boost = top_score * RETRIEVAL.section_boost_gamma.
        if not results:
            return results

        gamma = RETRIEVAL.section_boost_gamma
        top_score = max(r.score for r in results)
        boost = top_score * gamma

        target_set = set(target_sections)
        boosted_count = 0
        for chunk in results:
            if chunk.section_title in target_set:
                chunk.score += boost
                boosted_count += 1

        results.sort(key=lambda x: x.score, reverse=True)

        logger.debug(
            "[RETRIEVER][SECTION BOOST] gamma=%.2f boost=%.4f | %d/%d chunks boosted (sections=%s)",
            gamma, boost, boosted_count, len(results), target_sections
        )

        return results
