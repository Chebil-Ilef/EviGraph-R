from __future__ import annotations
import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import re
from datetime import datetime
from config.prompts import EVIDENCE_GRAPH_SYSTEM_PROMPT, build_claim_extraction_prompt, build_claim_extraction_prompt_batch
from config.settings import AGENT_MODELS, GRAPH_CONFIG
from schemas.objects import ClaimSubtype, EdgeRelation, EvidenceGraph, NodeType, SubQuery
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

        # Pre-warm local models so first build() call doesn't pay cold-start inside the hot path
        try:
            from utils.scicite import SciCiteModel
            SciCiteModel.get()
            logger.debug("[EVIDENCE GRAPH AGENT] SciCite model pre-warmed")
        except Exception as exc:
            logger.warning("[EVIDENCE GRAPH AGENT] SciCite pre-warm failed (non-fatal): %s", exc)

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
        enable_hop: bool = True,
    ) -> tuple["EvidenceGraph", dict]:

        if not documents:
            return EvidenceGraph(), {}

        t_build_start = time.perf_counter()

        hop_budget: int = GRAPH_CONFIG.hop_max_per_build if enable_hop else 0
        hop_budget_start: int = hop_budget
        _total_claims = 0
        _total_hop_claims = 0
        _hop_no_linked_citations = 0
        _hop_citation_raw_mismatch = 0
        _hop_no_resolved_id = 0
        _hop_retrieval_zero = 0
        _hop_succeeded = 0

        # Phase 1 + Phase 2 overlap: kick off LLM batches immediately while
        # build_graph_from_documents (SciCite CPU inference) runs concurrently.
        batch_size = GRAPH_CONFIG.claim_extraction_batch_size
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        # +1 worker slot for the graph-build future
        max_workers = min(len(batches) + 1, 9)

        doc_claims: dict[str, list[dict]] = {}

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # Submit graph construction alongside LLM calls so SciCite runs in parallel
            graph_future = pool.submit(build_graph_from_documents, documents)
            future_to_batch = {
                pool.submit(self._extract_claims_batch, batch, query): batch
                for batch in batches
            }

            # Collect LLM results as they arrive
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

        # Await graph build result outside the pool — raises here are visible in logs
        try:
            G = graph_future.result()
        except Exception as exc:
            logger.error(
                "[EVIDENCE GRAPH AGENT] build_graph_from_documents failed: %s — falling back to empty graph",
                exc, exc_info=True,
            )
            import networkx as nx
            G = nx.DiGraph()
            doc_claims = {doc.chunk_id: [] for doc in documents}

        t_extraction = time.perf_counter() - t0
        logger.info(
            "[EVIDENCE GRAPH AGENT] [TIMING] Phase 1+2 (graph build + LLM extraction, overlapped): %.3fs"
            " | %d chunks → %d batches (batch_size=%d, workers=%d)",
            t_extraction, len(documents), len(batches), batch_size, max_workers,
        )
        logger.info(
            "[EVIDENCE GRAPH AGENT] Base graph: %d nodes, %d edges",
            G.number_of_nodes(), G.number_of_edges(),
        )

        # Phase 3a: add claim nodes to graph (fast, pure Python — keep sequential)
        t0 = time.perf_counter()
        # Collect hop candidates while building the graph so we don't loop twice
        hop_candidates: list[tuple] = []  # (node_id, item, doc)

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
                doc.chunk_id[:20], len(claim_items), len(hop_items), cite_count,
            )
            for h in hop_items:
                linked = h.get("linked_citations") or []
                logger.info(
                    "[EVIDENCE GRAPH AGENT]   hop claim: reason=%s look_for=%r linked_citations=%d",
                    h.get("hop_reason"), (h.get("look_for") or "")[:80], len(linked),
                )
                for lc in linked:
                    logger.info(
                        "[EVIDENCE GRAPH AGENT]     citation_raw=%r score=%.2f",
                        lc.get("citation_raw", ""), lc.get("alignment_score", 0.0),
                    )

            for item in claims:
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                if item.get("type", NodeType.CLAIM.value) != NodeType.CLAIM.value:
                    continue
                node_type = NodeType.CLAIM.value
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
                raw_label = (item.get("scicite_label") or "").upper()
                if raw_label not in EdgeRelation.scicite_labels():
                    raw_label = EdgeRelation.BACKGROUND.value
                G.add_edge(node_id, doc.chunk_id, relation=raw_label, score=1.0)

                if (
                    hop_budget > 0
                    and self._retriever is not None
                    and self._embedder is not None
                    and (item.get("hop_reason") or "none") != "none"
                ):
                    hop_candidates.append((node_id, item, doc))

        t_graph_enrich = time.perf_counter() - t0
        logger.info(
            "[EVIDENCE GRAPH AGENT] [TIMING] Phase 3a (claim node insertion): %.3fs"
            " | %d claims inserted, %d hop candidates queued",
            t_graph_enrich, _total_claims, len(hop_candidates),
        )

        # Phase 3b: parallel hop retrieval — Qdrant calls run concurrently.
        # NetworkX is not thread-safe so all graph mutations go through a lock.
        if hop_candidates:
            t0 = time.perf_counter()
            graph_lock = threading.Lock()
            hop_workers = min(len(hop_candidates), 8)

            def _do_hop(args):
                node_id, item, doc = args
                # add_hop_to_graph does Qdrant I/O then graph mutation
                # We wrap graph mutations under the lock inside a local helper
                # so the Qdrant call itself is fully parallel.
                return add_hop_to_graph(
                    G,
                    claim_node_id=node_id,
                    item=item,
                    doc=doc,
                    retriever=self._retriever,
                    embedder=self._embedder,
                    hop_budget=1,  # each worker manages its own 1-unit budget
                    max_chunks_per_claim=GRAPH_CONFIG.hop_max_chunks_per_claim,
                    graph_lock=graph_lock,
                )

            # Cap total hops dispatched by original budget
            candidates_to_run = hop_candidates[:hop_budget]
            with ThreadPoolExecutor(max_workers=hop_workers) as pool:
                hop_futures = {pool.submit(_do_hop, args): args for args in candidates_to_run}
                for future in as_completed(hop_futures):
                    try:
                        _used, outcome = future.result()
                        if outcome == "no_hop_reason":
                            pass
                        elif outcome == "no_linked_citations":
                            _hop_no_linked_citations += 1
                        elif outcome == "citation_raw_mismatch":
                            _hop_citation_raw_mismatch += 1
                        elif outcome == "no_resolved_id":
                            _hop_no_resolved_id += 1
                        elif outcome == "retrieval_zero":
                            _hop_retrieval_zero += 1
                        elif outcome == "success":
                            _hop_succeeded += 1
                            hop_budget -= 1
                    except Exception as exc:
                        logger.warning("[EVIDENCE GRAPH AGENT] Hop future failed: %s", exc)

            t_hop = time.perf_counter() - t0
            logger.info(
                "[EVIDENCE GRAPH AGENT] [TIMING] Phase 3b (parallel hop retrieval): %.3fs"
                " | %d candidates dispatched, %d workers",
                t_hop, len(candidates_to_run), hop_workers,
            )

        # Diagnostics
        _chunks_with_spans = sum(1 for d in documents if (d.cite_spans or {}).get("cite_spans"))
        _total_cite_spans = sum(len((d.cite_spans or {}).get("cite_spans", [])) for d in documents)

        logger.info(
            "[EVIDENCE GRAPH AGENT] Claim extraction summary: %d chunk(s) → %d claims extracted (pre-dedup), "
            "%d with hop_reason≠none (budget_left=%d/%d) | cite_spans: %d chunk(s) have spans, %d total",
            len(documents), _total_claims, _total_hop_claims,
            hop_budget, hop_budget_start, _chunks_with_spans, _total_cite_spans,
        )
        logger.info(
            "[EVIDENCE GRAPH AGENT] Hop outcome breakdown: "
            "succeeded=%d | no_linked_citations=%d | citation_raw_mismatch=%d | "
            "no_resolved_id=%d | retrieval_zero=%d",
            _hop_succeeded, _hop_no_linked_citations, _hop_citation_raw_mismatch,
            _hop_no_resolved_id, _hop_retrieval_zero,
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
                _total_hop_claims, _hop_succeeded,
            )

        logger.info(
            "[EVIDENCE GRAPH AGENT] Enriched graph: %d nodes, %d edges",
            G.number_of_nodes(), G.number_of_edges(),
        )

        t0 = time.perf_counter()
        result_graph = evidence_graph_from_networkx(G)
        logger.info(
            "[EVIDENCE GRAPH AGENT] [TIMING] Phase 4 (NetworkX → EvidenceGraph serialization): %.3fs",
            time.perf_counter() - t0,
        )

        logger.info(
            "[EVIDENCE GRAPH AGENT] [TIMING] build() TOTAL: %.3fs",
            time.perf_counter() - t_build_start,
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
        return result_graph, hop_stats

    def _extract_claims(self, doc, query: str) -> list[dict]:

        raw = self.llm_client.chat_text(
            model=self.config.model,
            system_prompt=EVIDENCE_GRAPH_SYSTEM_PROMPT,
            user_prompt=build_claim_extraction_prompt(doc, query),
            temperature=self.config.temperature,
            timeout=self.config.timeout_seconds,
            max_tokens=self.config.max_tokens,
        )
        return self._parse_claims_json(raw)

    def _extract_claims_batch(self, docs: list, query: str) -> list[list[dict]]:

        if len(docs) == 1:
            return [self._extract_claims(docs[0], query)]

        raw = self.llm_client.chat_text(
            model=self.config.model,
            system_prompt=EVIDENCE_GRAPH_SYSTEM_PROMPT,
            user_prompt=build_claim_extraction_prompt_batch(docs, query),
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
