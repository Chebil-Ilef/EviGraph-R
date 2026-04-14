from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional
import re
from datetime import datetime
from config.prompts import EVIDENCE_GRAPH_SYSTEM_PROMPT, build_claim_extraction_prompt
from config.settings import AGENT_MODELS, GRAPH_CONFIG
from schemas.objects import ClaimSubtype, EvidenceGraph, NodeType, SubQuery
from utils.graph import add_hop_to_graph, build_graph_from_documents, evidence_graph_from_networkx, evidence_graph_to_networkx
from visualization.cytoscape_renderer import render_cytoscape
from utils.llm import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


class EvidenceGraphBuilderAgent:

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        output_dir: Optional[Path] = None,
        retriever=None,
        embedder=None,
    ) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.config = AGENT_MODELS["evidence_graph_builder"]
        self.output_dir = output_dir
        self._retriever = retriever
        self._embedder = embedder

    def build(
        self,
        query: str,
        sub_queries: List[SubQuery],
        documents: List,
    ) -> EvidenceGraph:

        if not documents:
            return EvidenceGraph()

        # Step 1: structural graph from documents (no LLM)
        G = build_graph_from_documents(documents)
        logger.info("[EVIDENCE GRAPH AGENT] Base graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

        hop_budget: int = GRAPH_CONFIG.hop_max_per_build

        # Step 2: LLM claim extraction : one call per chunk
        for doc in documents:
            try:
                claims = self._extract_claims(doc)
            except Exception as exc:
                logger.warning("[EVIDENCE GRAPH AGENT] Claim extraction failed for chunk %s: %s", doc.chunk_id, exc)
                claims = []

            for item in claims:
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                raw_type = item.get("type", NodeType.CLAIM.value)
                node_type = raw_type if raw_type in NodeType._value2member_map_ else NodeType.CLAIM.value
                # Deduplicate by content: same type + same text → same node ID across all chunks
                text_hash = hashlib.sha1(f"{node_type}:{text}".encode()).hexdigest()[:12]
                node_id = f"{node_type}:{text_hash}"
                if not G.has_node(node_id):
                    node_attrs: dict = {
                        "node_type": node_type,
                        "text": text,
                        "doc_id": doc.doc_id,
                        "chunk_id": doc.chunk_id,
                        "source_chunk_id": doc.chunk_id,
                    }
                    if node_type == NodeType.CLAIM.value:
                        subtype = item.get("subtype", "")
                        if subtype in ClaimSubtype._value2member_map_:
                            node_attrs["claim_subtype"] = subtype
                        if doc.sub_query_indices:
                            node_attrs["sub_query_indices"] = doc.sub_query_indices
                            node_attrs["sub_query_texts"] = [
                                sub_queries[i].text
                                for i in doc.sub_query_indices
                                if sub_queries and i < len(sub_queries)
                            ]
                    G.add_node(node_id, **node_attrs)
                G.add_edge(node_id, doc.chunk_id, relation="extracted_from", score=1.0)

                # hop retrieval for eligible claim nodes
                if (
                    node_type == NodeType.CLAIM.value
                    and hop_budget > 0
                    and self._retriever is not None
                    and self._embedder is not None
                ):
                    hop_budget = add_hop_to_graph(
                        G,
                        claim_node_id=node_id,
                        item=item,
                        doc=doc,
                        retriever=self._retriever,
                        embedder=self._embedder,
                        hop_budget=hop_budget,
                        max_chunks_per_claim=GRAPH_CONFIG.hop_max_chunks_per_claim,
                    )

        logger.info(
            "[EVIDENCE GRAPH AGENT] Enriched graph: %d nodes, %d edges",
            G.number_of_nodes(),
            G.number_of_edges(),
        )

        if self.output_dir is not None:
            self._dump_outputs(G, query, self.output_dir)

        return evidence_graph_from_networkx(G)

    def _extract_claims(self, doc) -> list[dict]:

        raw = self.llm_client.chat_text(
            model=self.config.model,
            system_prompt=EVIDENCE_GRAPH_SYSTEM_PROMPT,
            user_prompt=build_claim_extraction_prompt(doc),
            temperature=self.config.temperature,
            timeout=self.config.timeout_seconds,
        )
        return self._parse_claims_json(raw)

    def _dump_outputs(self, G, query: str, output_dir: Path) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^\w]+", "_", query.lower().strip())[:40].strip("_")
        run_dir = output_dir / f"{ts}_{slug}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._graph_output_path = run_dir  # Store for later graph-after rendering

        try:
            render_cytoscape(G, run_dir / "graph-before.html")
            logger.info("[EVIDENCE GRAPH AGENT] Cytoscape HTML (before judging) → %s", run_dir / "graph-before.html")
        except Exception as exc:
            logger.warning("[EVIDENCE GRAPH AGENT] Cytoscape render failed: %s", exc)

    def render_after(self, evidence_graph: EvidenceGraph) -> None:
        run_dir = getattr(self, "_graph_output_path", None)
        if run_dir is None:
            return
        try:
            G = evidence_graph_to_networkx(evidence_graph)
            render_cytoscape(G, run_dir / "graph-after.html")
            logger.info("[EVIDENCE GRAPH AGENT] Cytoscape HTML (after judging) → %s", run_dir / "graph-after.html")
        except Exception as exc:
            logger.warning("[EVIDENCE GRAPH AGENT] Cytoscape render (after) failed: %s", exc)

    @staticmethod
    def _parse_claims_json(raw: str) -> list[dict]:
        text = raw.strip()

        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end == -1 or start >= end:
                return []
            try:
                parsed = json.loads(text[start: end + 1])
            except json.JSONDecodeError:
                return []

        if not isinstance(parsed, list):
            return []

        return [item for item in parsed if isinstance(item, dict) and item.get("text")]
