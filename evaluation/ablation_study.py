from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class DecomposerAblation:
    skip: bool = False                 # A1.1: pass raw query as single sub-query
    equal_budget_weights: bool = False # A1.2: all sub-queries get weight 1/n


@dataclass
class RetrieverAblation:
    dense_only: bool = False           # R1: BGE-M3 sparse path disabled
    sparse_only: bool = False          # R2: BGE-M3 dense path disabled
    disable_section_boost: bool = False # R3: gamma = 0


@dataclass
class EvidenceGraphAblation:
    flat_chunks: bool = False          # G1: skip graph, pass flat chunk nodes
    disable_citation_hop: bool = False # G2: enable_hop = False


@dataclass
class JudgeAblation:
    skip: bool = False                 # J1: all claims pass unverified
    nli_only: bool = False             # J2: NLI for every claim, no LLM fallback
    llm_only: bool = False             # J3: LLM for every claim, skip NLI


@dataclass
class VariantConfig:
    name: str
    description: str
    state_overrides: dict[str, Any] = field(default_factory=dict)
    decomposer: DecomposerAblation = field(default_factory=DecomposerAblation)
    retriever: RetrieverAblation = field(default_factory=RetrieverAblation)
    evidence_graph: EvidenceGraphAblation = field(default_factory=EvidenceGraphAblation)
    judge: JudgeAblation = field(default_factory=JudgeAblation)

    @property
    def skip_decomposer(self) -> bool:
        return self.decomposer.skip

    @property
    def equal_budget_weights(self) -> bool:
        return self.decomposer.equal_budget_weights

    @property
    def disable_bm25(self) -> bool:
        return self.retriever.dense_only

    @property
    def disable_dense(self) -> bool:
        return self.retriever.sparse_only

    @property
    def disable_section_boost(self) -> bool:
        return self.retriever.disable_section_boost

    @property
    def skip_evidence_graph(self) -> bool:
        return self.evidence_graph.flat_chunks

    @property
    def disable_hop(self) -> bool:
        return self.evidence_graph.disable_citation_hop

    @property
    def skip_judge(self) -> bool:
        return self.judge.skip

    @property
    def nli_only(self) -> bool:
        return self.judge.nli_only

    @property
    def llm_only(self) -> bool:
        return self.judge.llm_only



class DecomposerVariants:
    NO_DECOMPOSITION = VariantConfig(
        name="A1.1",
        description="No decomposition — raw query sent directly to retriever",
        decomposer=DecomposerAblation(skip=True),
    )
    NO_BUDGET_WEIGHTS = VariantConfig(
        name="A1.2",
        description="No budget weights — all sub-queries receive equal weight",
        decomposer=DecomposerAblation(equal_budget_weights=True),
    )


class RetrieverVariants:
    DENSE_ONLY = VariantConfig(
        name="R1",
        description="Dense only — BGE-M3 sparse path disabled",
        retriever=RetrieverAblation(dense_only=True),
    )
    SPARSE_ONLY = VariantConfig(
        name="R2",
        description="Sparse only — BGE-M3 dense path disabled",
        retriever=RetrieverAblation(sparse_only=True),
    )
    NO_SECTION_BOOST = VariantConfig(
        name="R3",
        description="No section-aware boosting (gamma = 0)",
        retriever=RetrieverAblation(disable_section_boost=True),
    )


class EvidenceGraphVariants:
    FLAT_CHUNKS = VariantConfig(
        name="G1",
        description="Flat chunks — evidence graph replaced by ordered chunk list",
        evidence_graph=EvidenceGraphAblation(flat_chunks=True),
    )
    NO_CITATION_EXPANSION = VariantConfig(
        name="G2",
        description="No citation expansion (hop disabled)",
        evidence_graph=EvidenceGraphAblation(disable_citation_hop=True),
        state_overrides={"enable_hop": False},
    )


class JudgeVariants:
    NO_JUDGE = VariantConfig(
        name="J1",
        description="No judge — all claims pass directly to answer generator",
        judge=JudgeAblation(skip=True),
    )
    NLI_ONLY = VariantConfig(
        name="J2",
        description="NLI only — LLM judge disabled, NLI used for all claims",
        judge=JudgeAblation(nli_only=True),
    )
    LLM_ONLY = VariantConfig(
        name="J3",
        description="LLM only — NLI routing disabled, LLM judge for all claims",
        judge=JudgeAblation(llm_only=True),
    )


class FullSystemVariants:
    FULL = VariantConfig(
        name="full",
        description="EviGraph-R full system — all components enabled",
    )


VARIANTS: dict[str, VariantConfig] = {
    "A1.1": DecomposerVariants.NO_DECOMPOSITION,
    "A1.2": DecomposerVariants.NO_BUDGET_WEIGHTS,
    "R1": RetrieverVariants.DENSE_ONLY,
    "R2": RetrieverVariants.SPARSE_ONLY,
    "R3": RetrieverVariants.NO_SECTION_BOOST,
    "G1": EvidenceGraphVariants.FLAT_CHUNKS,
    "G2": EvidenceGraphVariants.NO_CITATION_EXPANSION,
    "J1": JudgeVariants.NO_JUDGE,
    "J2": JudgeVariants.NLI_ONLY,
    "J3": JudgeVariants.LLM_ONLY,
    "full": FullSystemVariants.FULL,
}


def load_shared_resources():
    """Load heavy models once and return them for reuse across variants."""
    import sys
    from pathlib import Path

    _SRC = Path(__file__).resolve().parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from evigraph.retrieval.embedder import Embedder
    from evigraph.retrieval.retriever import HybridQueryRetriever
    from evigraph.utils.llm import get_llm_client

    llm = get_llm_client()
    embedder = Embedder.from_model_key()
    retriever = HybridQueryRetriever()  # loads cross-encoder once
    return llm, embedder, retriever


def build_variant_services(variant: VariantConfig, shared=None):
    import sys
    from pathlib import Path

    _SRC = Path(__file__).resolve().parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from evigraph.retrieval.embedder import Embedder
    from evigraph.agents.answer_generator import AnswerGeneratorAgent
    from evigraph.workflow.graph import WorkflowServices
    from evigraph.utils.llm import get_llm_client

    if shared is not None:
        llm, embedder, base_retriever = shared
    else:
        from evigraph.retrieval.retriever import HybridQueryRetriever
        llm = get_llm_client()
        embedder = Embedder.from_model_key()
        base_retriever = HybridQueryRetriever()

    retriever = _build_retriever(variant, embedder, base_retriever)
    decomposer = _build_decomposer(variant, llm)
    egb = _build_egb(variant, retriever, embedder, llm)
    judge = _build_judge(variant, llm)
    answer_gen = AnswerGeneratorAgent(llm_client=llm)

    return WorkflowServices(
        decomposer=decomposer,
        embedder=embedder,
        retriever=retriever,
        evidence_graph_builder=egb,
        judge=judge,
        answer_generator=answer_gen,
    )


def _build_retriever(variant: VariantConfig, embedder, base=None):
    from evigraph.retrieval.retriever import HybridQueryRetriever, ChunkResult
    import logging

    _log = logging.getLogger(__name__)

    if base is None:
        base = HybridQueryRetriever()

    if not (variant.disable_bm25 or variant.disable_dense or variant.disable_section_boost):
        return base

    class _PatchedRetriever(base.__class__):
        def retrieve(self, embeddings, query_text, top_k=5, sparse_embeddings=None, target_sections=None):
            if variant.disable_section_boost:
                target_sections = None  # prevent section boost from firing

            if variant.disable_bm25:
                # Dense-only: bypass BM25 leg
                return self._retrieve_dense_only(embeddings, top_k)

            if variant.disable_dense:
                # BM25/sparse-only: call parent but override to skip dense prefetch
                return self._retrieve_bm25_only(query_text, top_k)

            return super().retrieve(
                embeddings, query_text, top_k, sparse_embeddings, target_sections
            )

        def _retrieve_bm25_only(self, query_text: str, top_k: int):
            from qdrant_client.models import Document

            try:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=Document(text=query_text, model=self.profile.bm25_model),
                    using=self.profile.sparse_vector_name,
                    limit=top_k,
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
                        paper_title=payload.get("title"),
                        chunk_index=payload.get("chunk_index"),
                        total_chunks=payload.get("total_chunks"),
                        cite_spans=payload.get("spans"),
                    ))
                return results
            except Exception as exc:
                _log.error("[R2 BM25-only] retrieval failed: %s", exc)
                return []

    # Monkey-patch the instance to avoid double-loading the cross-encoder
    patched = _PatchedRetriever.__new__(_PatchedRetriever)
    patched.__dict__.update(base.__dict__)
    return patched


def _build_decomposer(variant: VariantConfig, llm):
    from evigraph.agents.decomposer import DecomposerAgent
    from evigraph.schemas.objects import SubQuery, IMRaDSection

    base = DecomposerAgent(llm_client=llm)

    if not (variant.skip_decomposer or variant.equal_budget_weights):
        return base

    class _PatchedDecomposer(DecomposerAgent):
        def decompose(self, query: str):
            if variant.skip_decomposer:
                return [SubQuery(
                    text=query,
                    sections=[IMRaDSection.INTRODUCTION],
                    budget_weight=1.0,
                )]
            sub_queries = super().decompose(query)
            if variant.equal_budget_weights and sub_queries:
                n = len(sub_queries)
                equal_w = round(1.0 / n, 6)
                for sq in sub_queries:
                    sq.__dict__["budget_weight"] = equal_w
            return sub_queries

    patched = _PatchedDecomposer.__new__(_PatchedDecomposer)
    patched.__dict__.update(base.__dict__)
    return patched


def _build_egb(variant: VariantConfig, retriever, embedder, llm):
    from evigraph.agents.evidence_graph_builder import EvidenceGraphBuilderAgent

    if variant.skip_evidence_graph:
        # G1: return a stub that converts flat chunk list into a minimal graph
        return _FlatChunkGraphBuilder()

    egb = EvidenceGraphBuilderAgent(
        llm_client=llm,
        retriever=retriever,
        embedder=embedder,
    )

    if variant.disable_hop:
        # G2 is handled via state.enable_hop=False which EvidenceGraphBuilder already reads
        pass

    return egb


class _FlatChunkGraphBuilder:

    def build(self, query: str, sub_queries, documents, enable_hop: bool = False, **kwargs):
        from evigraph.schemas.objects import (
            EvidenceGraph, EvidenceNode, NodeType,
        )
        nodes = []
        for doc in documents:
            # Use CLAIM node type so answer_generator._collect_claims picks them up.
            # Mark verdict=Supported so they pass the verdict filter.
            nodes.append(EvidenceNode(
                node_id=doc.chunk_id,
                node_type=NodeType.CLAIM,
                text=doc.content,
                chunk_id=doc.chunk_id,
                paper_id=doc.doc_id,
                metadata={"score": doc.score, "verdict": "Supported"},
            ))
        return EvidenceGraph(nodes=nodes, edges=[]), {}

    def render_after(self, evidence_graph) -> None:
        pass


def _build_judge(variant: VariantConfig, llm):
    from evigraph.agents.judge import JudgeAgent

    base = JudgeAgent(llm_client=llm)

    if not (variant.skip_judge or variant.nli_only or variant.llm_only):
        return base

    class _PatchedJudge(JudgeAgent):
        def filter(self, query, evidence_graph, documents):
            if variant.skip_judge:
                # Pass all claims unverified with a synthetic "supported" verdict
                return self._pass_through(evidence_graph)
            if variant.nli_only:
                return self._nli_only_filter(query, evidence_graph, documents)
            if variant.llm_only:
                return self._llm_only_filter(query, evidence_graph, documents)
            return super().filter(query, evidence_graph, documents)

        def _pass_through(self, evidence_graph):
            from evigraph.schemas.objects import JudgementResult, NodeType, VerdictDetail
            verdict_details = {}
            for node in (evidence_graph.nodes if evidence_graph else []):
                if node.node_type.value == "claim":
                    verdict_details[node.node_id] = VerdictDetail(
                        verdict="Supported",
                        verifier_used="pass_through",
                        evidence_trail=[],
                    )
            return JudgementResult(
                evidence_graph=evidence_graph,
                verdict_details=verdict_details,
            )

        def _nli_only_filter(self, query, evidence_graph, documents):
            # Force NLI path by temporarily disabling the LLM judge escalation.
            # We monkey-patch _llm_judge to always return "inconclusive" so
            # the NLI verdict is never escalated.
            original_llm_judge = self._llm_judge

            def _force_nli_verdict(claim_id, claim_text, dag, **kwargs):
                from evigraph.schemas.objects import VerdictType
                return {
                    "verdict": VerdictType.INCONCLUSIVE.value,
                    "verifier_used": "nli_only_stub",
                    "evidence_trail": [],
                }

            self._llm_judge = _force_nli_verdict
            try:
                result = super().filter(query, evidence_graph, documents)
            finally:
                self._llm_judge = original_llm_judge
            return result

        def _llm_only_filter(self, query, evidence_graph, documents):
            # Force LLM path for every claim by bypassing the NLI phase entirely.
            # The routing in JudgeAgent.filter() buckets claims by hop_depth/contradiction,
            # not claim_type, so patching _classify_claim_type has no effect on routing.
            # Instead we override filter() directly: skip NLI, send all claims to LLM.
            import time
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from evigraph.schemas.objects import (
                ClaimType, EvidenceGraph, HopDepth, JudgementResult, NodeType, VerdictDetail,
            )
            from evigraph.utils.graph import project_dag, compute_hop_depth

            if not evidence_graph or not evidence_graph.nodes:
                return JudgementResult(
                    evidence_graph=evidence_graph.model_copy(deep=True) if evidence_graph else EvidenceGraph(),
                    verdict_details={},
                )

            G = self._to_networkx(evidence_graph)
            dag = project_dag(G)

            claim_nodes = [
                n for n, d in dag.nodes(data=True)
                if d.get("node_type") == NodeType.CLAIM.value
            ]

            hop_depth_cache = {cid: compute_hop_depth(cid, dag) for cid in claim_nodes}

            verdict_details: dict = {}

            def _judge_one(cid: str):
                claim_text = dag.nodes[cid].get("text", "")
                if not claim_text:
                    return cid, None
                hop_depth = hop_depth_cache[cid]
                has_contradiction = bool(dag.nodes[cid].get("contradicts"))
                vd = self._llm_judge(cid, claim_text, dag, verifier_label="llm_only")
                vd["claim_type"] = ClaimType.INFERENTIAL.value
                vd["hop_depth"] = hop_depth.value
                if dag.nodes[cid].get("hop_fail_reason"):
                    vd["hop_attempted"] = True
                    vd["hop_fail_reason"] = dag.nodes[cid]["hop_fail_reason"]
                return cid, vd

            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=min(len(claim_nodes), 8)) as pool:
                futures = {pool.submit(_judge_one, cid): cid for cid in claim_nodes}
                for future in as_completed(futures):
                    cid, vd = future.result()
                    if vd is not None:
                        verdict_details[cid] = vd
            import logging
            logging.getLogger(__name__).info(
                "[JUDGE][J3] LLM-only: %.3fs | %d claims", time.perf_counter() - t0, len(claim_nodes)
            )

            verdict_details_obj = {cid: VerdictDetail(**vd) for cid, vd in verdict_details.items()}
            judged_graph = evidence_graph.model_copy(deep=True)
            self._merge_verdicts_into_graph(judged_graph, verdict_details)
            judged_graph.edges.extend(self._build_judged_edges(judged_graph, verdict_details))
            return JudgementResult(evidence_graph=judged_graph, verdict_details=verdict_details_obj)

    patched = _PatchedJudge.__new__(_PatchedJudge)
    patched.__dict__.update(base.__dict__)
    return patched
