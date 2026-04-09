from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional
import re
from datetime import datetime
from config.prompts import EVIDENCE_GRAPH_SYSTEM_PROMPT, build_claim_extraction_prompt
from config.settings import AGENT_MODELS
from schemas.objects import EvidenceGraph, SubQuery
from utils.graph import build_graph_from_documents, evidence_graph_from_networkx
from visualization.cytoscape_renderer import render_cytoscape
from utils.llm import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


class EvidenceGraphBuilderAgent:

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.config = AGENT_MODELS["evidence_graph_builder"]
        self.output_dir = output_dir

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

        # Step 2: LLM claim extraction : one call per chunk
        for doc in documents:
            try:
                claims = self._extract_claims(doc)
            except Exception as exc:
                logger.warning("[EVIDENCE GRAPH AGENT] Claim extraction failed for chunk %s: %s", doc.chunk_id, exc)
                claims = []

            for i, item in enumerate(claims):
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                node_type = item.get("type", "claim")
                if node_type not in ("claim", "concept"):
                    node_type = "claim"
                claim_node_id = f"{node_type}:{doc.chunk_id}:{i}"
                G.add_node(
                    claim_node_id,
                    node_type=node_type,
                    text=text,
                    doc_id=doc.doc_id,
                    chunk_id=doc.chunk_id,
                    source_chunk_id=doc.chunk_id,
                )
                G.add_edge(claim_node_id, doc.chunk_id, relation="extracted_from", score=1.0)

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

        try:
            render_cytoscape(G, run_dir / "graph.html")
            logger.info("[EVIDENCE GRAPH AGENT] Cytoscape HTML → %s", run_dir / "graph.html")
        except Exception as exc:
            logger.warning("[EVIDENCE GRAPH AGENT] Cytoscape render failed: %s", exc)


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
