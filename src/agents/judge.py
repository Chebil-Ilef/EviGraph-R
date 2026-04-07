from __future__ import annotations

import json
import logging
import re
from typing import Any

import networkx as nx

from config.prompts import JUDGE_SYSTEM_PROMPT, build_llm_judge_user_prompt
from config.settings import AGENT_MODELS
from schemas.objects import (
    ClaimType,
    EvidenceEdge,
    EvidenceGraph,
    HopDepth,
    JudgementResult,
    RetrievedDocument,
    VerdictDetail,
    VerdictType,
)
from utils.llm import LLMClient, get_llm_client
from utils.graph import project_dag, backwards_traverse, compute_hop_depth
from utils.npm import npm_verify
from utils.nli import nli_verify

logger = logging.getLogger(__name__)

# Regex: detect atomic-factual signals (numbers, percentages, dates, named metrics)
_ATOMIC_PATTERN = re.compile(
    r"\b\d+(\.\d+)?(%|x\b|×)?"        # numbers / percentages / multipliers
    r"|\b(19|20)\d{2}\b"               # years
    r"|\b[A-Z]{2}[A-Z0-9\-]*\b"       # true acronyms ≥2 caps (BERT, GPT, SQuAD, MMLU)
    r"|\b[A-Z][a-z]+-\d+\b"           # versioned names: GPT-3, GPT-4, Llama-2
    r"|\b\d+\s*(ms|s|mb|gb|tb|fps)\b", # measurements
)


def _verdict_dict(
    verdict: VerdictType,
    verifier: str,
    evidence_trail: list[dict],
    error_stage: str | None = None,
) -> dict[str, Any]:

    return {
        "verdict": verdict.value,
        "verifier_used": verifier,
        "evidence_trail": evidence_trail,
        "error_stage": error_stage,
    }


class JudgeAgent:

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.config = AGENT_MODELS["judge"]

    def filter(
        self,
        query: str,
        evidence_graph: EvidenceGraph,
        documents: list[RetrievedDocument],
    ) -> JudgementResult:

        if not evidence_graph or not evidence_graph.nodes:
            return JudgementResult(
                filtered_documents=list(documents),
                judged_relations=[],
                verdict_details={},
            )

        G = self._to_networkx(evidence_graph)
        dag = project_dag(G)

        claim_nodes = [
            n for n, d in dag.nodes(data=True)
            if d.get("node_type") in ("claim", "concept")
        ]

        verdict_details: dict[str, dict] = {}
        supported_chunk_ids: set[str] = set()
        judged_edges: list[EvidenceEdge] = []

        for claim_id in claim_nodes:
            claim_text = dag.nodes[claim_id].get("text", "")
            if not claim_text:
                continue

            claim_type, hop_depth = self._classify_claim(claim_id, claim_text, dag)
            has_contradiction = bool(dag.nodes[claim_id].get("contradicts"))

            vd = self._route_and_verify(
                claim_id=claim_id,
                claim_text=claim_text,
                claim_type=claim_type,
                hop_depth=hop_depth,
                has_contradiction=has_contradiction,
                dag=dag,
            )
            # Add claim type and hop depth to verdict details for schema
            vd["claim_type"] = claim_type.value
            vd["hop_depth"] = hop_depth.value
            verdict_details[claim_id] = vd

            if vd["verdict"] == VerdictType.SUPPORTED.value:
                # Collect chunk IDs that back this supported claim
                chunk_id = dag.nodes[claim_id].get("chunk_id") or dag.nodes[claim_id].get("source_chunk_id")
                if chunk_id:
                    supported_chunk_ids.add(chunk_id)
                judged_edges.append(
                    EvidenceEdge(
                        source=claim_id,
                        target=chunk_id or "",
                        relation="verified",
                        score=1.0,
                        metadata={"verifier": vd["verifier_used"]},
                    )
                )

        filtered_docs = [d for d in documents if d.chunk_id in supported_chunk_ids]
        # Fallback: if nothing survived verification, forward all (safe degradation)
        if not filtered_docs:
            logger.warning("[JUDGE] No claims survived verification — forwarding all documents")
            filtered_docs = list(documents)

        logger.info(
            "[JUDGE] %d/%d claims supported; %d/%d docs forwarded",
            sum(1 for v in verdict_details.values() if v["verdict"] == VerdictType.SUPPORTED.value),
            len(verdict_details),
            len(filtered_docs),
            len(documents),
        )

        return JudgementResult(
            filtered_documents=filtered_docs,
            judged_relations=judged_edges,
            verdict_details={
                claim_id: VerdictDetail(**vd)
                for claim_id, vd in verdict_details.items()
            },
        )


    def _classify_claim(
        self, claim_id: str, claim_text: str, dag
    ) -> tuple[ClaimType, HopDepth]:

        claim_type = (
            ClaimType.ATOMIC_FACTUAL
            if _ATOMIC_PATTERN.search(claim_text)
            else ClaimType.INFERENTIAL
        )

        hop_depth = compute_hop_depth(claim_id, dag)

        return claim_type, hop_depth

    def _route_and_verify(
        self,
        *,
        claim_id: str,
        claim_text: str,
        claim_type: ClaimType,
        hop_depth: HopDepth,
        has_contradiction: bool,
        dag,
    ) -> dict:

        # Cycle guard
        if dag.nodes[claim_id].get("cycle_detected"):
            return _verdict_dict(VerdictType.INCONCLUSIVE, "none", [], "cycle_detected")

        evidence_chunks = self._collect_evidence_chunks(claim_id, dag)

        # Contradiction override → always LLM judge
        if has_contradiction:
            return self._llm_judge(claim_id, claim_text, dag, error_stage="contradiction_flagged")

        # Multi-hop → LLM judge
        if hop_depth == HopDepth.MULTI:
            return self._llm_judge(claim_id, claim_text, dag)

        # Single-hop atomic → NPM
        if claim_type == ClaimType.ATOMIC_FACTUAL:
            result = npm_verify(claim_text, evidence_chunks)
            return _verdict_dict(
                VerdictType(result["verdict"]),
                result["verifier_used"],
                result["evidence_trail"],
                result.get("error_stage"),
            )

        # Single-hop inferential → NLI (escalates to LLM if Neutral)
        result = nli_verify(claim_text, evidence_chunks)
        verdict = result["verdict"]
        
        # NLI returns "Neutral" when it can't decide → escalate to LLM judge
        if verdict == "Neutral":
            return self._llm_judge(
                claim_id,
                claim_text,
                dag,
                verifier_label="nli→llm",
                error_stage=result.get("error_stage"),
            )
        
        return _verdict_dict(
            VerdictType(verdict),
            result["verifier_used"],
            result["evidence_trail"],
            result.get("error_stage"),
        )

    def _llm_judge(
        self,
        claim_id: str,
        claim_text: str,
        dag,
        *,
        verifier_label: str = "llm_judge",
        error_stage: str | None = None,
    ) -> dict:

        trail = backwards_traverse(claim_id, dag)
        if not trail:
            return _verdict_dict(VerdictType.NOT_SUPPORTED, verifier_label, [], error_stage or "no_evidence")

        try:
            raw = self.llm_client.chat_text(
                model=self.config.model,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=build_llm_judge_user_prompt(claim_text, trail),
                temperature=self.config.temperature,
                timeout=self.config.timeout_seconds,
            )
        except Exception as exc:
            logger.warning("[JUDGE][LLM] Verdict request failed for %s: %s", claim_id, exc)
            verdict = VerdictType.INCONCLUSIVE
            return _verdict_dict(verdict, verifier_label, trail, error_stage)

        payload = self._parse_verdict_json(raw)
        verdict_str = payload.get("verdict", VerdictType.INCONCLUSIVE.value)

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

        return _verdict_dict(verdict, verifier_label, trail, error_stage)

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
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and start < end:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return {}
