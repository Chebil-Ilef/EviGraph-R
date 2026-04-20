from __future__ import annotations
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import re
from datetime import datetime
from config.prompts import EVIDENCE_GRAPH_SYSTEM_PROMPT, build_claim_extraction_prompt, build_claim_extraction_prompt_batch
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
        
        # Warn if hop capability is not available
        if (self._retriever is None or self._embedder is None) and GRAPH_CONFIG.hop_max_per_build > 0:
            logger.warning(
                "[EVIDENCE GRAPH AGENT] Hop capability disabled: retriever=%s, embedder=%s. "
                "To enable hop reasoning, both retriever and embedder must be provided.",
                "✓" if self._retriever else "✗",
                "✓" if self._embedder else "✗",
            )

    def build(
        self,
        query: str,
        sub_queries: List[SubQuery],
        documents: List,
    ) -> tuple["EvidenceGraph", dict]:

        if not documents:
            return EvidenceGraph(), {}

        # Step 1: structural graph from documents (no LLM)
        G = build_graph_from_documents(documents)
        logger.info("[EVIDENCE GRAPH AGENT] Base graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

        hop_budget: int = GRAPH_CONFIG.hop_max_per_build
        hop_budget_start: int = hop_budget
        _total_claims = 0
        _total_hop_claims = 0
        # Detailed hop tracking per failure mode
        _hop_no_linked_citations = 0
        _hop_citation_raw_mismatch = 0
        _hop_no_resolved_id = 0
        _hop_retrieval_zero = 0
        _hop_succeeded = 0

        # Step 2: LLM claim extraction — parallel batched calls
        # Grouping chunks per call reduces round-trips while keeping prompts manageable.
        batch_size = GRAPH_CONFIG.claim_extraction_batch_size
        doc_claims: dict[str, list[dict]] = {}
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        max_workers = min(len(batches), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_batch = {pool.submit(self._extract_claims_batch, batch): batch for batch in batches}
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    for doc, claims in zip(batch, batch_results):
                        doc_claims[doc.chunk_id] = claims
                except Exception as exc:
                    logger.warning("[EVIDENCE GRAPH AGENT] Batch claim extraction failed: %s", exc)
                    for doc in batch:
                        doc_claims[doc.chunk_id] = []

        for doc in documents:
            claims = doc_claims.get(doc.chunk_id, [])

            claim_items = [c for c in claims if c.get("type") == "claim"]
            hop_items = [c for c in claim_items if (c.get("hop_reason") or "none") != "none"]
            cite_count = len((doc.cite_spans or {}).get("cite_spans", []))
            _total_claims += len(claim_items)
            _total_hop_claims += len(hop_items)

            if hop_items and (self._retriever is None or self._embedder is None):
                logger.error(
                    "[EVIDENCE GRAPH AGENT] ✗ CRITICAL: %d hop claims extracted but retriever/embedder missing! "
                    "Retriever: %s, Embedder: %s. Hop reasoning will be SKIPPED for this chunk. "
                    "To fix: pass both retriever and embedder to EvidenceGraphBuilderAgent()",
                    len(hop_items),
                    "✓" if self._retriever else "✗",
                    "✓" if self._embedder else "✗",
                )

            logger.info(
                "[EVIDENCE GRAPH AGENT] chunk=%s | claims=%d hop_claims=%d | cite_spans=%d",
                doc.chunk_id[:20],
                len(claim_items),
                len(hop_items),
                cite_count,
            )
            for h in hop_items:
                linked = h.get("linked_citations") or []
                logger.info(
                    "[EVIDENCE GRAPH AGENT]   hop claim: reason=%s look_for=%r linked_citations=%d",
                    h.get("hop_reason"),
                    (h.get("look_for") or "")[:80],
                    len(linked),
                )
                for lc in linked:
                    logger.info(
                        "[EVIDENCE GRAPH AGENT]     citation_raw=%r score=%.2f",
                        lc.get("citation_raw", ""),
                        lc.get("alignment_score", 0.0),
                    )

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
                    budget_before = hop_budget
                    hop_budget, hop_outcome = add_hop_to_graph(
                        G,
                        claim_node_id=node_id,
                        item=item,
                        doc=doc,
                        retriever=self._retriever,
                        embedder=self._embedder,
                        hop_budget=hop_budget,
                        max_chunks_per_claim=GRAPH_CONFIG.hop_max_chunks_per_claim,
                    )
                    if hop_outcome == "no_hop_reason":
                        pass  # not a hop claim, don't count
                    elif hop_outcome == "no_linked_citations":
                        _hop_no_linked_citations += 1
                    elif hop_outcome == "citation_raw_mismatch":
                        _hop_citation_raw_mismatch += 1
                    elif hop_outcome == "no_resolved_id":
                        _hop_no_resolved_id += 1
                    elif hop_outcome == "retrieval_zero":
                        _hop_retrieval_zero += 1
                    elif hop_outcome == "success":
                        _hop_succeeded += 1

        # Compute cite_span coverage across all chunks for diagnostics
        _chunks_with_spans = sum(
            1 for d in documents if (d.cite_spans or {}).get("cite_spans")
        )
        _total_cite_spans = sum(
            len((d.cite_spans or {}).get("cite_spans", [])) for d in documents
        )

        logger.info(
            "[EVIDENCE GRAPH AGENT] Claim extraction summary: %d chunk(s) → %d claims extracted (pre-dedup), "
            "%d with hop_reason≠none (budget_left=%d/%d) | cite_spans: %d chunk(s) have spans, %d total",
            len(documents),
            _total_claims,
            _total_hop_claims,
            hop_budget,
            hop_budget_start,
            _chunks_with_spans,
            _total_cite_spans,
        )
        logger.info(
            "[EVIDENCE GRAPH AGENT] Hop outcome breakdown: "
            "succeeded=%d | no_linked_citations=%d | citation_raw_mismatch=%d | "
            "no_resolved_id=%d | retrieval_zero=%d",
            _hop_succeeded,
            _hop_no_linked_citations,
            _hop_citation_raw_mismatch,
            _hop_no_resolved_id,
            _hop_retrieval_zero,
        )

        if hop_budget_start > 0 and _total_hop_claims == 0 and _total_cite_spans == 0:
            logger.warning(
                "[EVIDENCE GRAPH AGENT] 0 hop claims generated — ALL %d chunk(s) have 0 cite_spans. "
                "LLM cannot produce hop claims without available citations. "
                "Root cause: cite_spans were not resolved for these chunks (check ID resolution pipeline).",
                len(documents),
            )
        elif hop_budget_start > 0 and _total_hop_claims == 0 and _total_cite_spans > 0:
            logger.warning(
                "[EVIDENCE GRAPH AGENT] 0 hop claims despite %d cite_spans available — "
                "LLM judged ALL claims self-contained (hop_reason=none). "
                "Check prompt / model if this is unexpected.",
                _total_cite_spans,
            )

        # CRITICAL VALIDATION: If hops were extracted but can't be processed
        if _total_hop_claims > 0 and (self._retriever is None or self._embedder is None):
            logger.error(
                "[EVIDENCE GRAPH AGENT] X CRITICAL: %d hop claims extracted but retriever/embedder NOT PROVIDED! "
                "Hop reasoning WILL NOT execute. This is likely a silent regression. "
                "FIX: Pass both retriever and embedder to EvidenceGraphBuilderAgent constructor.",
                _total_hop_claims,
            )
        elif _total_hop_claims > 0 and self._retriever is not None and self._embedder is not None:
            logger.info(
                "[EVIDENCE GRAPH AGENT] ✓ Hop reasoning active: %d hop claims attempted → %d succeeded",
                _total_hop_claims,
                _hop_succeeded,
            )

        logger.info(
            "[EVIDENCE GRAPH AGENT] Enriched graph: %d nodes, %d edges",
            G.number_of_nodes(),
            G.number_of_edges(),
        )

        if self.output_dir is not None:
            self._dump_outputs(G, query, self.output_dir)

        hop_stats = {
            "n_chunks": len(documents),
            "n_claims_extracted": _total_claims,
            "n_hop_claims": _total_hop_claims,
            "n_chunks_with_cite_spans": _chunks_with_spans,
            "n_total_cite_spans": _total_cite_spans,
            "hop_budget_start": hop_budget_start,
            "hop_budget_left": hop_budget,
            "hop_succeeded": _hop_succeeded,
            "hop_no_linked_citations": _hop_no_linked_citations,
            "hop_citation_raw_mismatch": _hop_citation_raw_mismatch,
            "hop_no_resolved_id": _hop_no_resolved_id,
            "hop_retrieval_zero": _hop_retrieval_zero,
        }
        return evidence_graph_from_networkx(G), hop_stats

    def _extract_claims(self, doc) -> list[dict]:

        raw = self.llm_client.chat_text(
            model=self.config.model,
            system_prompt=EVIDENCE_GRAPH_SYSTEM_PROMPT,
            user_prompt=build_claim_extraction_prompt(doc),
            temperature=self.config.temperature,
            timeout=self.config.timeout_seconds,
            max_tokens=self.config.max_tokens,
        )
        return self._parse_claims_json(raw)

    def _extract_claims_batch(self, docs: list) -> list[list[dict]]:

        if len(docs) == 1:
            return [self._extract_claims(docs[0])]

        raw = self.llm_client.chat_text(
            model=self.config.model,
            system_prompt=EVIDENCE_GRAPH_SYSTEM_PROMPT,
            user_prompt=build_claim_extraction_prompt_batch(docs),
            temperature=self.config.temperature,
            timeout=self.config.timeout_seconds,
            max_tokens=self.config.max_tokens * len(docs) if self.config.max_tokens else None,
        )
        return self._parse_claims_json_batch(raw, len(docs))

    @staticmethod
    def _parse_claims_json_batch(raw: str, n_chunks: int) -> list[list[dict]]:

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
                return [[] for _ in range(n_chunks)]
            try:
                parsed = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return [[] for _ in range(n_chunks)]

        if not isinstance(parsed, list):
            return [[] for _ in range(n_chunks)]

        # If LLM returned flat array (forgot nesting), treat as single-chunk response
        if parsed and not isinstance(parsed[0], list):
            result = [[item for item in parsed if isinstance(item, dict) and item.get("text")]]
            result += [[] for _ in range(n_chunks - 1)]
            return result

        results: list[list[dict]] = []
        for i in range(n_chunks):
            if i < len(parsed) and isinstance(parsed[i], list):
                results.append([item for item in parsed[i] if isinstance(item, dict) and item.get("text")])
            else:
                results.append([])
        return results

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
