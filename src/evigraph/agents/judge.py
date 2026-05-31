from __future__ import annotations
import json
import logging
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any
import networkx as nx
from evigraph.config.prompts import JUDGE_SYSTEM_PROMPT, build_llm_judge_user_prompt
from evigraph.config.settings import AGENT_MODELS
from evigraph.schemas.objects import (
    ClaimType,
    EvidenceEdge,
    EvidenceGraph,
    HopDepth,
    JUDGE_EDGE_PREFIX,
    JudgementResult,
    NodeType,
    RetrievedDocument,
    VerdictDetail,
    VerdictType,
)
from evigraph.utils.llm import LLMClient, get_llm_client
from evigraph.utils.graph import project_dag, backwards_traverse, compute_hop_depth
from evigraph.utils.nli import nli_verify, nli_verify_batch
logger = logging.getLogger(__name__)

# Regex: detect atomic-factual signals (numbers, percentages, dates, named metrics)
_ATOMIC_PATTERN = re.compile(
    r"\b\d+(\.\d+)?(%|x\b|×)?"        # numbers / percentages / multipliers
    r"|\b(19|20)\d{2}\b"               # years
    r"|\b[A-Z]{2}[A-Z0-9\-]*\b"       # true acronyms ≥2 caps (BERT, GPT, SQuAD, MMLU)
    r"|\b[A-Z][a-z]+-\d+\b"           # versioned names: GPT-3, GPT-4, Llama-2
    r"|\b\d+\s*(ms|s|mb|gb|tb|fps)\b", # measurements
)


class JudgeAgent:

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.config = AGENT_MODELS["judge"]

    def _verdict_dict(self, verdict: VerdictType, verifier: str, evidence_trail: list[dict], error_stage: str | None = None, reason: str | None = None) -> dict[str, Any]:
        return {
            "verdict": verdict.value,
            "verifier_used": verifier,
            "evidence_trail": evidence_trail,
            "error_stage": error_stage,
            "reason": reason,
        }

    @staticmethod
    def _merge_verdicts_into_graph(evidence_graph: EvidenceGraph, verdict_details: dict[str, dict]) -> None:
        for node in evidence_graph.nodes:
            if node.node_id in verdict_details:
                vd = verdict_details[node.node_id]
                node.metadata["verdict"] = vd.get("verdict")
                node.metadata["verifier_used"] = vd.get("verifier_used")
                node.metadata["claim_type"] = vd.get("claim_type")
                node.metadata["hop_depth"] = vd.get("hop_depth")
                node.metadata["reason"] = vd.get("reason")
                node.metadata["hop_attempted"] = vd.get("hop_attempted")
                node.metadata["hop_fail_reason"] = vd.get("hop_fail_reason")

    @classmethod
    def _judge_relation(cls, verdict: str) -> str:
        normalized = verdict.lower().replace("-", "_").replace(" ", "_")
        return f"{JUDGE_EDGE_PREFIX}{normalized}"

    @classmethod
    def _build_judged_edges(cls, judged_graph: EvidenceGraph, verdict_details: dict[str, dict]) -> list[EvidenceEdge]:
        claim_to_chunk: dict[str, str] = {}
        for node in judged_graph.nodes:
            if node.node_type == NodeType.CLAIM and node.chunk_id:
                claim_to_chunk[node.node_id] = node.chunk_id

        judged_edges: list[EvidenceEdge] = []
        for claim_id, vd in verdict_details.items():
            chunk_id = claim_to_chunk.get(claim_id)
            if not chunk_id:
                continue
            judged_edges.append(
                EvidenceEdge(
                    source=claim_id,
                    target=chunk_id,
                    relation=cls._judge_relation(vd["verdict"]),
                    score=0.0,
                    metadata={
                        "verdict": vd["verdict"],
                        "verifier": vd["verifier_used"],
                        "claim_type": vd.get("claim_type"),
                        "hop_depth": vd.get("hop_depth"),
                    },
                )
            )
        return judged_edges

    def filter(
        self,
        query: str,
        evidence_graph: EvidenceGraph,
        documents: list[RetrievedDocument],
    ) -> JudgementResult:

        if not evidence_graph or not evidence_graph.nodes:
            return JudgementResult(
                evidence_graph=evidence_graph.model_copy(deep=True) if evidence_graph else EvidenceGraph(),
                verdict_details={},
            )

        t_start = time.perf_counter()

        t0 = time.perf_counter()
        G = self._to_networkx(evidence_graph)
        dag = project_dag(G)
        t1 = time.perf_counter()
        logger.info("[JUDGE] Graph projection: %.3fs", t1 - t0)

        t0 = time.perf_counter()
        claim_nodes = [
            n for n, d in dag.nodes(data=True)
            if d.get("node_type") == NodeType.CLAIM.value
        ]
        t1 = time.perf_counter()
        logger.info("[JUDGE] Found %d claim nodes: %.3fs", len(claim_nodes), t1 - t0)

        # Cache hop depths and trails to avoid recomputation
        t0 = time.perf_counter()
        hop_depth_cache: dict[str, HopDepth] = {}
        for cid in claim_nodes:
            hop_depth_cache[cid] = compute_hop_depth(cid, dag)
        t1 = time.perf_counter()
        logger.info("[JUDGE] Precomputed hop depths for all claims: %.3fs", t1 - t0)

        verdict_details: dict[str, dict] = {}
        skipped_count: int = 0

        # Classify all claims upfront (pure Python, instant).
        claim_meta: dict[str, tuple[ClaimType, HopDepth, bool]] = {}
        for cid in claim_nodes:
            claim_text = dag.nodes[cid].get("text", "")
            if not claim_text:
                continue
            claim_meta[cid] = (
                self._classify_claim_type(claim_text),
                hop_depth_cache[cid],
                bool(dag.nodes[cid].get("contradicts")),
            )

        # Split claims into two buckets:
        #   nli_claims  — single-hop, no contradiction → run through batched NLI first
        #   llm_direct  — contradiction or multi-hop   → go straight to LLM (parallel)
        nli_claims: list[str] = []
        llm_direct: list[str] = []
        for cid in claim_nodes:
            if cid not in claim_meta:
                skipped_count += 1
                continue
            _, hop_depth, has_contradiction = claim_meta[cid]
            if has_contradiction or hop_depth == HopDepth.MULTI:
                llm_direct.append(cid)
            else:
                nli_claims.append(cid)

        # Phase A-NLI: single batched forward pass for all NLI-eligible claims.
        t0 = time.perf_counter()
        nli_inputs = [
            (dag.nodes[cid].get("text", ""), self._collect_evidence_chunks(cid, dag))
            for cid in nli_claims
        ]
        nli_results = nli_verify_batch(nli_inputs)
        t_nli = time.perf_counter() - t0
        logger.info("[JUDGE] Phase A-NLI (batched): %.3fs | %d claims", t_nli, len(nli_claims))

        escalation_queue: list[tuple[str, ClaimType, HopDepth]] = []
        for cid, nli_result in zip(nli_claims, nli_results):
            claim_type, hop_depth, _ = claim_meta[cid]
            nli_verdict = nli_result["verdict"]
            if nli_verdict in (VerdictType.SUPPORTED.value, VerdictType.CONTRADICTED.value):
                verdict_type = VerdictType.SUPPORTED if nli_verdict == VerdictType.SUPPORTED.value else VerdictType.CONTRADICTED
                vd = self._verdict_dict(
                    verdict_type, "nli",
                    nli_result["evidence_trail"],
                    nli_result.get("error_stage"),
                    reason=nli_result.get("reason"),
                )
                vd["claim_type"] = claim_type.value
                vd["hop_depth"] = hop_depth.value
                hop_fail_reason = dag.nodes[cid].get("hop_fail_reason")
                if hop_fail_reason:
                    vd["hop_attempted"] = True
                    vd["hop_fail_reason"] = hop_fail_reason
                verdict_details[cid] = vd
            else:
                escalation_queue.append((cid, claim_type, hop_depth))

        def _make_vd(claim_id: str, vd: dict) -> dict:
            claim_type, hop_depth, _ = claim_meta[claim_id]
            vd["claim_type"] = claim_type.value
            vd["hop_depth"] = hop_depth.value
            hop_fail_reason = dag.nodes[claim_id].get("hop_fail_reason")
            if hop_fail_reason:
                vd["hop_attempted"] = True
                vd["hop_fail_reason"] = hop_fail_reason
            return vd

        # Phase A-LLM: direct-LLM routes (contradiction / multi-hop) in parallel threads.
        if llm_direct:
            t0 = time.perf_counter()

            def _direct_llm_one(claim_id: str):
                claim_text = dag.nodes[claim_id].get("text", "")
                claim_type, hop_depth, has_contradiction = claim_meta[claim_id]
                vd = self._route_and_verify(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    claim_type=claim_type,
                    hop_depth=hop_depth,
                    has_contradiction=has_contradiction,
                    dag=dag,
                    trail_cache=None,
                )
                return claim_id, _make_vd(claim_id, vd)

            with ThreadPoolExecutor(max_workers=min(len(llm_direct), 8)) as pool:
                futures = {pool.submit(_direct_llm_one, cid): cid for cid in llm_direct}
                for future in as_completed(futures):
                    claim_id, vd = future.result()
                    verdict_details[claim_id] = vd

            logger.info(
                "[JUDGE] Phase A-LLM (direct): %.3fs | %d claims",
                time.perf_counter() - t0, len(llm_direct),
            )

        # Phase B: fire all NLI→LLM escalations in parallel (batched, not inline).
        if escalation_queue:
            t0 = time.perf_counter()

            def _escalate_one(args: tuple[str, ClaimType, HopDepth]):
                claim_id = args[0]
                vd = self._llm_judge(
                    claim_id,
                    dag.nodes[claim_id].get("text", ""),
                    dag,
                    verifier_label="nli→llm",
                )
                return claim_id, _make_vd(claim_id, vd)

            with ThreadPoolExecutor(max_workers=min(len(escalation_queue), 8)) as pool:
                escalation_futures: dict[
                    Future[tuple[str, dict[str, Any]]],
                    tuple[str, ClaimType, HopDepth],
                ] = {
                    pool.submit(_escalate_one, args): args for args in escalation_queue
                }
                for future in as_completed(escalation_futures):
                    claim_id, vd = future.result()
                    verdict_details[claim_id] = vd

            logger.info(
                "[JUDGE] Phase B (NLI→LLM batched): %.3fs | %d claims escalated",
                time.perf_counter() - t0, len(escalation_queue),
            )

        t1 = time.perf_counter()

        n_nli = sum(1 for v in verdict_details.values() if v.get("verifier_used") == "nli")
        n_nli_llm = sum(1 for v in verdict_details.values() if v.get("verifier_used") == "nli→llm")
        n_llm = sum(1 for v in verdict_details.values() if v.get("verifier_used") == "llm_judge")
        n_other = len(verdict_details) - n_nli - n_nli_llm - n_llm
        logger.info(
            "[JUDGE] Verified %d/%d claims (%d skipped — empty text): %.3fs"
            " | verifier breakdown: NLI=%d  NLI→LLM=%d  LLM=%d  other=%d",
            len(verdict_details), len(claim_nodes), skipped_count, t1 - t0,
            n_nli, n_nli_llm, n_llm, n_other,
        )

        t_end = time.perf_counter()
        n_supported = sum(1 for v in verdict_details.values() if v["verdict"] == VerdictType.SUPPORTED.value)
        logger.info(
            "[JUDGE] TOTAL: %.3fs | %d/%d claims verified supported (of %d total)",
            t_end - t_start,
            n_supported,
            len(verdict_details),
            len(claim_nodes),
        )

        verdict_details_obj = {
            claim_id: VerdictDetail(**vd)
            for claim_id, vd in verdict_details.items()
        }

        judged_graph = evidence_graph.model_copy(deep=True)
        self._merge_verdicts_into_graph(judged_graph, verdict_details)
        judged_graph.edges.extend(self._build_judged_edges(judged_graph, verdict_details))

        return JudgementResult(
            evidence_graph=judged_graph,
            verdict_details=verdict_details_obj,
        )


    def _classify_claim_type(self, claim_text: str) -> ClaimType:
        return (
            ClaimType.ATOMIC_FACTUAL
            if _ATOMIC_PATTERN.search(claim_text)
            else ClaimType.INFERENTIAL
        )

    def _route_and_verify(
        self,
        *,
        claim_id: str,
        claim_text: str,
        claim_type: ClaimType,
        hop_depth: HopDepth,
        has_contradiction: bool,
        dag,
        trail_cache: dict[str, list[dict]] | None = None,
    ) -> dict:

        # Cycle guard
        if dag.nodes[claim_id].get("cycle_detected"):
            return self._verdict_dict(VerdictType.INCONCLUSIVE, "none", [], "cycle_detected")

        evidence_chunks = self._collect_evidence_chunks(claim_id, dag)

        # Contradiction override → always LLM (needs semantic reasoning)
        if has_contradiction:
            return self._llm_judge(
                claim_id, claim_text, dag,
                trail_cache=trail_cache,
                error_stage="contradiction_flagged",
            )

        # Multi-hop → always LLM (needs chain reasoning)
        if hop_depth == HopDepth.MULTI:
            return self._llm_judge(claim_id, claim_text, dag, trail_cache=trail_cache)

        # Single-hop: NLI semantic check
        t0 = time.perf_counter()
        nli_result = nli_verify(claim_text, evidence_chunks)
        t1 = time.perf_counter()
        nli_verdict = nli_result["verdict"]
        logger.debug("[JUDGE][NLI] %s: %.3fs (%s)", claim_id[:30], t1 - t0, nli_verdict)

        if nli_verdict == VerdictType.SUPPORTED.value:
            return self._verdict_dict(
                VerdictType.SUPPORTED,
                "nli",
                nli_result["evidence_trail"],
                nli_result.get("error_stage"),
                reason=nli_result.get("reason"),
            )

        if nli_verdict == VerdictType.CONTRADICTED.value:
            return self._verdict_dict(
                VerdictType.CONTRADICTED,
                "nli",
                nli_result["evidence_trail"],
                nli_result.get("error_stage"),
                reason=nli_result.get("reason"),
            )

        # Step 3: Neutral / unavailable NLI result → LLM semantic reasoning
        return self._llm_judge(
            claim_id,
            claim_text,
            dag,
            trail_cache=trail_cache,
            verifier_label="nli→llm",
            error_stage=nli_result.get("error_stage"),
        )

    def _llm_judge(
        self,
        claim_id: str,
        claim_text: str,
        dag,
        *,
        trail_cache: dict[str, list[dict]] | None = None,
        verifier_label: str = "llm_judge",
        error_stage: str | None = None,
    ) -> dict:

        # Use cached trail if available
        if trail_cache is None:
            trail_cache = {}

        if claim_id not in trail_cache:
            t0 = time.perf_counter()
            trail_cache[claim_id] = backwards_traverse(claim_id, dag)
            t1 = time.perf_counter()
            logger.debug("[JUDGE][TRAVERSE] %s: %.3fs", claim_id[:30], t1 - t0)
        trail = trail_cache[claim_id]

        if not trail:
            return self._verdict_dict(VerdictType.NOT_SUPPORTED, verifier_label, [], error_stage or "no_evidence")

        try:
            t0 = time.perf_counter()
            raw = self.llm_client.chat_text(
                model=self.config.model,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=build_llm_judge_user_prompt(claim_text, trail),
                temperature=self.config.temperature,
                timeout=self.config.timeout_seconds,
                max_tokens=self.config.max_tokens,
            )
            t1 = time.perf_counter()
            logger.debug("[JUDGE][LLM] %s: %.3fs", claim_id[:30], t1 - t0)
        except Exception as exc:
            logger.warning("[JUDGE][LLM] Verdict request failed for %s: %s", claim_id, exc)
            verdict = VerdictType.INCONCLUSIVE
            return self._verdict_dict(verdict, verifier_label, trail, error_stage)

        payload = self._parse_verdict_json(raw)
        verdict_str = payload.get("verdict", VerdictType.INCONCLUSIVE.value)
        reasoning = payload.get("reasoning") or None

        if not payload:
            logger.warning("[JUDGE][LLM] Verdict response was not valid JSON for %s", claim_id)

        try:
            verdict = VerdictType(verdict_str)
        except ValueError:
            logger.warning(
                "[JUDGE][LLM] Verdict response had invalid verdict for %s: %r",
                claim_id,
                verdict_str,
            )
            verdict = VerdictType.INCONCLUSIVE

        return self._verdict_dict(verdict, verifier_label, trail, error_stage, reason=reasoning)

    def _collect_evidence_chunks(self, claim_id: str, dag) -> list[str]:

        chunks: list[str] = []
        for neighbor in dag.successors(claim_id):
            node_type = dag.nodes[neighbor].get("node_type", "")
            if node_type == "chunk":
                text = dag.nodes[neighbor].get("text", "")
                if text:
                    chunks.append(text)
        return chunks

    @staticmethod
    def _to_networkx(evidence_graph: EvidenceGraph):
        G: nx.DiGraph = nx.DiGraph()
        for node in evidence_graph.nodes:
            G.add_node(
                node.node_id,
                node_type=node.node_type.value,
                text=node.text,
                doc_id=node.doc_id or "",
                chunk_id=node.chunk_id or "",
                **node.metadata,
            )
        for edge in evidence_graph.edges:
            G.add_edge(
                edge.source,
                edge.target,
                relation=edge.relation,
                score=edge.score,
                **edge.metadata,
            )
        return G

    @staticmethod
    def _parse_verdict_json(raw: str) -> dict:
        from evigraph.utils.latex_sanitizer import safe_json_loads
        
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        
        try:
            return safe_json_loads(text)
        except json.JSONDecodeError as e:
            # Log the failure but return minimal valid verdict to allow pipeline to continue
            logger.error(f"[JUDGE] Failed to parse verdict JSON: {e}. Raw was: {text[:200]}")
            # Return inconclusive verdict with error info
            return {"verdict": "Inconclusive", "reasoning": f"JSON parse error: {str(e)[:100]}"}
