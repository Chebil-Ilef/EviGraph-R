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
    RetrievedDocument,
    VerdictType,
)
from utils.llm import LLMClient, get_llm_client

logger = logging.getLogger(__name__)

# Relations that form valid evidence edges (source → claim direction)
_EVIDENCE_RELATIONS = {"extracted_from", "supports"}

# Max backward traversal depth for LLM judge
_MAX_TRAVERSAL_DEPTH = 5

# NPM: fraction of key claim tokens that must appear in evidence
_NPM_THRESHOLD = 0.70

# NLI entail/contradict confidence threshold
_NLI_THRESHOLD = 0.70

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


class _NLIModel:
    _instance: "_NLIModel | None" = None

    def __init__(self) -> None:
        from transformers import pipeline  
        self._pipe = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-small",
        )

    @classmethod
    def get(cls) -> "_NLIModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def classify(self, claim: str, evidence: str) -> dict[str, float]:

        result = self._pipe(
            evidence,
            candidate_labels=["entails", "contradicts", "neutral"],
            hypothesis_template="{}",
        )
        scores: dict[str, float] = {}
        for label, score in zip(result["labels"], result["scores"]):
            scores[label] = score
        return scores


class JudgeAgent:

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.config = AGENT_MODELS["judge"]

    def filter(
        self,
        query: str,
        evidence_graph: EvidenceGraph,
        documents: list[RetrievedDocument],
    ) -> dict:

        if not evidence_graph or not evidence_graph.nodes:
            return {
                "filtered_documents": list(documents),
                "judged_relations": [],
                "verdict_details": {},
            }

        G = self._to_networkx(evidence_graph)
        dag = self._project_dag(G)

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

        return {
            "filtered_documents": filtered_docs,
            "judged_relations": judged_edges,
            "verdict_details": verdict_details,
        }


    def _project_dag(self, G) -> "nx.DiGraph":

        dag: nx.DiGraph = nx.DiGraph()

        for node_id, data in G.nodes(data=True):
            dag.add_node(node_id, **data)

        for src, tgt, data in G.edges(data=True):
            relation = (data.get("relation") or "").lower()
            if relation in _EVIDENCE_RELATIONS:
                dag.add_edge(src, tgt, **data)

        # Detect and log cycles (should not exist; Agent 2 bug if they do)
        try:
            cycles = list(nx.simple_cycles(dag))
            if cycles:
                logger.warning("[JUDGE] Cycle(s) detected in DAG: %d cycle(s); affected claims marked inconclusive", len(cycles))
                cyclic_nodes: set[str] = {n for cycle in cycles for n in cycle}
                for n in cyclic_nodes:
                    dag.nodes[n]["cycle_detected"] = True
        except Exception:
            pass

        return dag


    def _classify_claim(
        self, claim_id: str, claim_text: str, dag
    ) -> tuple[ClaimType, HopDepth]:

        claim_type = (
            ClaimType.ATOMIC_FACTUAL
            if _ATOMIC_PATTERN.search(claim_text)
            else ClaimType.INFERENTIAL
        )

        hop_depth = self._compute_hop_depth(claim_id, dag)

        return claim_type, hop_depth

    def _compute_hop_depth(self, claim_id: str, dag) -> HopDepth:

        successors = list(dag.successors(claim_id))
        if not successors:
            return HopDepth.SINGLE

        for neighbor in successors:
            edge_data = dag.edges[claim_id, neighbor]
            relation = (edge_data.get("relation") or "").lower()
            # SciCite boundary labels indicate multi-hop
            if relation not in ("extracted_from", "supports", "belongs_to", ""):
                return HopDepth.MULTI

            # If the neighbor is itself a claim/concept with further hops
            neighbor_type = dag.nodes[neighbor].get("node_type", "")
            if neighbor_type in ("claim", "concept"):
                return HopDepth.MULTI

        return HopDepth.SINGLE

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
            return self._npm_verify(claim_text, evidence_chunks)

        # Single-hop inferential → NLI (escalates to LLM if Neutral)
        return self._nli_verify(claim_id, claim_text, evidence_chunks, dag)


    def _npm_verify(self, claim_text: str, evidence_chunks: list[str]) -> dict:
        if not evidence_chunks:
            return _verdict_dict(VerdictType.NOT_SUPPORTED, "npm", [], "no_evidence")

        key_tokens = self._extract_key_tokens(claim_text)
        if not key_tokens:
            return _verdict_dict(VerdictType.INCONCLUSIVE, "npm", [])

        evidence_combined = " ".join(evidence_chunks).lower()
        matched = sum(1 for t in key_tokens if t.lower() in evidence_combined)
        coverage = matched / len(key_tokens)

        trail = [{"text": chunk[:200]} for chunk in evidence_chunks]

        if coverage >= _NPM_THRESHOLD:
            return _verdict_dict(VerdictType.SUPPORTED, "npm", trail)
        return _verdict_dict(VerdictType.NOT_SUPPORTED, "npm", trail)

    @staticmethod
    def _extract_key_tokens(text: str) -> list[str]:

        tokens: list[str] = []
        # Numeric values
        tokens += re.findall(r"\b\d+(?:\.\d+)?(?:%|x)?\b", text)
        # Acronyms / named entities (all-caps sequences ≥2 chars)
        tokens += re.findall(r"\b[A-Z][A-Z0-9\-]{1,}\b", text)
        # Quoted terms
        tokens += re.findall(r'"([^"]+)"', text)
        return list(dict.fromkeys(tokens))  # deduplicate, preserve order

    def _nli_verify(
        self,
        claim_id: str,
        claim_text: str,
        evidence_chunks: list[str],
        dag,
    ) -> dict:
        if not evidence_chunks:
            return _verdict_dict(VerdictType.NOT_SUPPORTED, "nli", [], "no_evidence")

        try:
            nli = _NLIModel.get()
        except Exception as exc:
            logger.warning("[JUDGE][NLI] Model load failed, escalating to LLM judge: %s", exc)
            return self._llm_judge(claim_id, claim_text, dag, verifier_label="nli→llm")

        # Aggregate NLI scores across all evidence chunks
        agg: dict[str, float] = {"entails": 0.0, "contradicts": 0.0, "neutral": 0.0}
        trail: list[dict] = []

        for chunk in evidence_chunks:
            scores = nli.classify(claim_text, chunk)
            for k in agg:
                agg[k] = max(agg[k], scores.get(k, 0.0))
            trail.append({"text": chunk[:200], "scores": scores})

        if agg["entails"] >= _NLI_THRESHOLD:
            return _verdict_dict(VerdictType.SUPPORTED, "nli", trail)
        if agg["contradicts"] >= _NLI_THRESHOLD:
            return _verdict_dict(VerdictType.CONTRADICTED, "nli", trail)

        # Neutral → escalate to LLM judge
        return self._llm_judge(claim_id, claim_text, dag, verifier_label="nli→llm")

    def _llm_judge(
        self,
        claim_id: str,
        claim_text: str,
        dag,
        *,
        verifier_label: str = "llm_judge",
        error_stage: str | None = None,
    ) -> dict:

        trail = self._backwards_traverse(claim_id, dag)
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
            payload = self._parse_verdict_json(raw)
            verdict = VerdictType(payload.get("verdict", VerdictType.INCONCLUSIVE.value))
        except Exception as exc:
            logger.warning("[JUDGE][LLM] Verdict parse failed for %s: %s", claim_id, exc)
            verdict = VerdictType.INCONCLUSIVE

        return _verdict_dict(verdict, verifier_label, trail, error_stage)

    def _backwards_traverse(self, claim_id: str, dag) -> list[dict]:

        trail: list[dict] = []
        visited: set[str] = set()
        queue: list[str] = [claim_id]

        while queue and len(trail) < _MAX_TRAVERSAL_DEPTH:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)

            data = dag.nodes[node_id]
            node_type = data.get("node_type", "")

            # Only add chunk nodes to the trail (skip claim/paper nodes)
            if node_type == "chunk":
                neighbors = list(dag.successors(node_id))
                # Find the incoming edge label (from the child that led us here)
                scicite_label = ""
                for child in dag.predecessors(node_id):
                    if child in visited or child == claim_id:
                        edge_data = dag.edges.get((child, node_id), {})
                        scicite_label = edge_data.get("relation", "")
                        break

                trail.append({
                    "node_id": node_id,
                    "text": (data.get("text") or "")[:400],
                    "scicite_label": scicite_label,
                })

                # If no further chunk ancestors, stop (this is root)
                has_chunk_parent = any(
                    dag.nodes[s].get("node_type") == "chunk"
                    for s in dag.successors(node_id)
                )
                if not has_chunk_parent:
                    break
                queue.extend(dag.successors(node_id))

            else:
                queue.extend(dag.successors(node_id))

        return trail

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
