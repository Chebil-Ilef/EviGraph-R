from __future__ import annotations
import difflib
import json
import logging
import re
import threading
from typing import List
import networkx as nx
from schemas.objects import EvidenceGraph, EvidenceNode, EvidenceEdge, NodeType, HopDepth, EdgeRelation, HopReason
from utils.scicite import classify_citation
from visualization.cytoscape_renderer import render_cytoscape  
import time
from config.settings import GRAPH_CONFIG

logger = logging.getLogger(__name__)

def build_graph_from_documents(documents: List) -> "nx.DiGraph":

    G: nx.DiGraph = nx.DiGraph()

    for doc in documents:
        paper_id: str = doc.doc_id
        chunk_id: str = doc.chunk_id

        if not G.has_node(paper_id):
            G.add_node(paper_id, node_type="paper", text="", paper_id=paper_id)

        G.add_node(
            chunk_id,
            node_type="chunk",
            text=doc.content,
            paper_id=paper_id,
            doc_id=paper_id,
            chunk_id=chunk_id,
            section=doc.section_title or "",
            score=doc.score,
        )
        G.add_edge(chunk_id, paper_id, relation="CHUNK_OF", score=1.0)

        spans_data = doc.cite_spans or {}
        for span in spans_data.get("cite_spans", []):
            cited_id = span.get("arxiv_id") or span.get("doi") or ""
            if not cited_id:
                continue
            if not G.has_node(cited_id):
                G.add_node(cited_id, node_type="paper", text="", paper_id=cited_id)
            if not G.has_edge(chunk_id, cited_id):
                start, end = span.get("start", 0), span.get("end", 0)
                citation_sentence = doc.content[start:end] if end > start else doc.content[:300]
                label, confidence = classify_citation(citation_sentence)
                G.add_edge(chunk_id, cited_id, relation=label, score=confidence)

    return G

def evidence_graph_to_networkx(graph) -> nx.DiGraph:

    G: nx.DiGraph = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(
            node.node_id,
            node_type=node.node_type.value,
            text=node.text,
            doc_id=node.doc_id or "",
            chunk_id=node.chunk_id or "",
            **node.metadata,
        )
    for edge in graph.edges:
        G.add_edge(
            edge.source,
            edge.target,
            relation=edge.relation,
            score=edge.score,
            **edge.metadata,
        )
    return G


def evidence_graph_from_networkx(G: nx.DiGraph):

    _reserved_node = {"node_type", "text", "doc_id", "chunk_id"}
    _reserved_edge = {"relation", "score"}

    nodes = [
        EvidenceNode(
            node_id=str(node_id),
            node_type=NodeType(data.get("node_type", NodeType.CHUNK.value)),
            text=data.get("text", ""),
            doc_id=data.get("doc_id") or None,
            chunk_id=data.get("chunk_id") or None,
            metadata={k: v for k, v in data.items() if k not in _reserved_node},
        )
        for node_id, data in G.nodes(data=True)
    ]
    edges = [
        EvidenceEdge(
            source=str(src),
            target=str(tgt),
            relation=data.get("relation", ""),
            score=float(data.get("score", 0.0)),
            metadata={k: v for k, v in data.items() if k not in _reserved_edge},
        )
        for src, tgt, data in G.edges(data=True)
    ]
    return EvidenceGraph(nodes=nodes, edges=edges)


def _to_primitive(value) -> str | int | float | bool:
    
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value)


# Judge-specific utilities


def project_dag(G: nx.DiGraph) -> nx.DiGraph:

    dag: nx.DiGraph = nx.DiGraph()

    # Copy all nodes
    for node_id, data in G.nodes(data=True):
        dag.add_node(node_id, **data)

    # Filter edges: keep only evidence relations
    for src, tgt, data in G.edges(data=True):
        relation = data.get("relation") or ""
        if relation in EdgeRelation.evidence_relations():
            dag.add_edge(src, tgt, **data)

    # Detect cycles in the evidence-only subgraph and mark affected nodes.
    try:
        t0 = time.perf_counter()
        cycles = list(nx.simple_cycles(dag))
        t1 = time.perf_counter()
        if cycles:
            logger.warning(
                "[JUDGE][DAG] Cycle(s) detected in DAG: %d cycle(s); affected claims marked inconclusive (%.3fs)",
                len(cycles),
                t1 - t0,
            )
            cyclic_nodes: set[str] = {n for cycle in cycles for n in cycle}
            for n in cyclic_nodes:
                dag.nodes[n]["cycle_detected"] = True
        else:
            logger.debug("[JUDGE][DAG] Cycle detection completed: 0 cycles (%.3fs)", t1 - t0)
    except Exception:
        logger.exception("[JUDGE][DAG] Cycle detection failed")

    return dag


def backwards_traverse(claim_id: str, dag: nx.DiGraph, max_depth: int | None = None) -> list[dict]:
    if max_depth is None:
        max_depth = GRAPH_CONFIG.max_evidence_trail_depth

    trail: list[dict] = []
    visited: set[str] = set()
    queue: list[str] = [claim_id]

    while queue and len(trail) < max_depth:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        data = dag.nodes[node_id]
        node_type = data.get("node_type", "")

        # Only add chunk nodes to trail (skip claim/paper nodes)
        if node_type == "chunk":
            # Find the incoming edge label (from the predecessor that led us here)
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

            # Continue traversal only if this chunk has further chunk parents
            # (i.e. it is not a root/leaf chunk). Hop chunks are always leaves
            # and will not extend the queue further.
            chunk_parents = [
                s for s in dag.successors(node_id)
                if dag.nodes[s].get("node_type") == "chunk"
            ]
            queue.extend(chunk_parents)
        else:
            queue.extend(dag.successors(node_id))

    return trail


def compute_hop_depth(claim_id: str, dag: nx.DiGraph) -> HopDepth:

    successors = list(dag.successors(claim_id))
    if not successors:
        return HopDepth.SINGLE

    for neighbor in successors:
        edge_data = dag.edges[claim_id, neighbor]
        relation = edge_data.get("relation") or ""
        # SciCite boundary labels indicate multi-hop
        if relation not in EdgeRelation.single_hop():
            return HopDepth.MULTI

        # If the neighbor is itself a claim with further hops
        neighbor_type = dag.nodes[neighbor].get("node_type", "")
        if neighbor_type == "claim":
            return HopDepth.MULTI

    return HopDepth.SINGLE


# Matches: arXiv:1502.03167, arxiv:1502.03167v2, arXiv:cs/0612054,
#          https://arxiv.org/abs/2005.14165
_ARXIV_ID_RE = re.compile(
    r"(?:\barxiv[:\s/]+|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)
# Matches DOI embedded in citation_raw, e.g. "doi:10.1145/..." or "https://doi.org/10.1145/..."
_DOI_RE = re.compile(
    r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)(10\.\d{4,}/\S+)",
    re.IGNORECASE,
)


def _extract_ids_from_citation_raw(citation_raw: str) -> tuple[str, str]:
    """
    Try to pull arxiv_id and/or doi directly out of the citation_raw string.
    Returns (arxiv_id, doi) — either or both may be empty.
    """
    arxiv_id = ""
    doi = ""

    m = _ARXIV_ID_RE.search(citation_raw)
    if m:
        raw_id = m.group(1).rstrip(".,;)")
        # Normalise: strip version suffix for corpus matching
        arxiv_id = re.sub(r"v\d+$", "", raw_id)

    m = _DOI_RE.search(citation_raw)
    if m:
        doi = m.group(1).rstrip(".,;)")

    # If we got an arxiv_id but no doi, synthesise the canonical arxiv DOI
    # (10.48550/arXiv.XXXX.XXXXX) which is how unarXive stores it
    if arxiv_id and not doi:
        doi = f"10.48550/arxiv.{arxiv_id}"

    return arxiv_id, doi


def resolve_cited_paper_id(
    linked_citations: list[dict],
    cite_spans_data: dict | None,
) -> list[dict]:

    if not linked_citations or not cite_spans_data:
        return []

    spans = cite_spans_data.get("cite_spans", [])
    if not spans:
        return []

    # Build raw → span index (first occurrence wins for duplicates)
    raw_index: dict[str, dict] = {}
    for span in spans:
        raw = (span.get("raw") or "").strip()
        if raw and raw not in raw_index:
            raw_index[raw] = span

    resolved: list[dict] = []
    for cit in linked_citations:
        citation_raw = (cit.get("citation_raw") or "").strip()
        if not citation_raw:
            continue

        span = raw_index.get(citation_raw)
        if span is None:
            # Exact match failed — try fuzzy match (handles minor LLM formatting differences)
            candidates = difflib.get_close_matches(
                citation_raw, raw_index.keys(), n=1, cutoff=0.85
            )
            if candidates:
                logger.info(
                    "[GRAPH][HOP] citation_raw fuzzy match: %r → %r",
                    citation_raw, candidates[0],
                )
                span = raw_index[candidates[0]]
            else:
                logger.warning(
                    "[GRAPH][HOP] citation_raw mismatch — LLM said %r but not in raw_index. "
                    "Available keys (%d total): %s",
                    citation_raw,
                    len(raw_index),
                    list(raw_index.keys())[:10],
                )
                continue

        arxiv_id = (span.get("arxiv_id") or "").strip()
        doi = (span.get("doi") or "").strip()
        openalex_id = (span.get("openalex_id") or "").strip()

        if not arxiv_id and not doi and not openalex_id:
            logger.warning(
                "[GRAPH][HOP] Span matched citation_raw=%r but has no arxiv_id/doi/openalex_id — "
                "span keys: %s. Attempting ID extraction from citation_raw …",
                citation_raw, list(span.keys()),
            )

        # openalex_id alone is not filterable in Qdrant (corpus is indexed by arxiv_id/doi).
        # If we already have arxiv/doi from the span we can proceed; otherwise try regex rescue.
        if not arxiv_id and not doi:
            extracted_arxiv, extracted_doi = _extract_ids_from_citation_raw(citation_raw)
            if extracted_arxiv or extracted_doi:
                logger.info(
                    "[GRAPH][HOP]   → extracted from citation_raw: arxiv=%r doi=%r (source=regex)",
                    extracted_arxiv or "None", extracted_doi or "None",
                )
                arxiv_id, doi = extracted_arxiv, extracted_doi
            else:
                logger.warning(
                    "[GRAPH][HOP]   → no arxiv/doi found in citation_raw=%r "
                    "(span ids: arxiv=%r doi=%r openalex=%r, regex found nothing). Skipping.",
                    citation_raw, arxiv_id or "None", doi or "None", openalex_id or "None",
                )
                continue

        resolved.append({
            "arxiv_id": arxiv_id,
            "doi": doi,
            "openalex_id": openalex_id,
            "alignment_score": cit.get("alignment_score", 0.0),
            "alignment_reason": cit.get("alignment_reason", ""),
            "citation_raw": citation_raw,
        })

    return resolved


def add_hop_to_graph(
    G: "nx.DiGraph",
    *,
    claim_node_id: str,
    item: dict,
    doc,
    retriever,
    embedder,
    hop_budget: int,
    max_chunks_per_claim: int,
    graph_lock: "threading.Lock | None" = None,
) -> tuple[int, str]:
   
    _lock = graph_lock or threading.Lock()

    hop_reason_raw = (item.get("hop_reason") or HopReason.NONE.value).strip()
    if hop_reason_raw == HopReason.NONE.value or hop_reason_raw not in HopReason._value2member_map_:
        return hop_budget, "no_hop_reason"

    if retriever is None or embedder is None:
        logger.error(
            "[GRAPH][HOP] ✗ CRITICAL BUG: Hop claim detected but retriever/embedder are None! "
            "This should never happen. claim=%s hop_reason=%s retriever=%s embedder=%s",
            claim_node_id[:40], hop_reason_raw,
            "✓" if retriever else "✗", "✓" if embedder else "✗",
        )
        return hop_budget, "no_hop_reason"

    look_for = (item.get("look_for") or "").strip()
    if not look_for:
        logger.warning(
            "[GRAPH][HOP] Skipping hop for claim=%s hop_reason=%s — look_for is empty",
            claim_node_id[:40], hop_reason_raw,
        )
        with _lock:
            G.nodes[claim_node_id]["hop_attempted"] = True
            G.nodes[claim_node_id]["hop_fail_reason"] = "no_look_for"
        return hop_budget, "no_linked_citations"

    linked_citations = item.get("linked_citations") or []
    if not linked_citations:
        logger.warning(
            "[GRAPH][HOP] Skipping hop: claim=%s hop_reason=%s look_for=%r — "
            "LLM set hop_reason but provided no linked_citations. "
            "Available cite_spans in chunk: %d",
            claim_node_id[:40], hop_reason_raw, look_for,
            len((doc.cite_spans or {}).get("cite_spans", [])),
        )
        with _lock:
            G.nodes[claim_node_id]["hop_attempted"] = True
            G.nodes[claim_node_id]["hop_fail_reason"] = "no_linked_citations"
        return hop_budget, "no_linked_citations"

    logger.info(
        "[GRAPH][HOP] Attempting hop: claim=%s hop_reason=%s look_for=%r linked=%d cite_spans=%d",
        claim_node_id[:40], hop_reason_raw, look_for,
        len(linked_citations), len((doc.cite_spans or {}).get("cite_spans", [])),
    )
    resolved = resolve_cited_paper_id(linked_citations, doc.cite_spans)
    if not resolved:
        cite_spans_available = len((doc.cite_spans or {}).get("cite_spans", []))
        lc_raws = [c.get("citation_raw", "") for c in linked_citations]
        available_raws = [(s.get("raw") or "") for s in (doc.cite_spans or {}).get("cite_spans", [])]
        logger.warning(
            "[GRAPH][HOP] citation_raw mismatch OR missing IDs: claim=%s look_for=%r — "
            "LLM cited %d ref(s) %s | chunk cite_spans=%d | "
            "available raw keys (first 10): %s",
            claim_node_id[:40], look_for, len(linked_citations), lc_raws,
            cite_spans_available, available_raws[:10],
        )
        spans_data = doc.cite_spans or {}
        raw_index = {(s.get("raw") or "").strip(): s for s in spans_data.get("cite_spans", [])}
        any_matched = any((c.get("citation_raw") or "").strip() in raw_index for c in linked_citations)
        if any_matched:
            logger.warning(
                "[GRAPH][HOP]   → raw key matched but span has no arxiv_id/doi/openalex_id "
                "AND regex extraction also failed — ID resolution is fully incomplete for this citation.",
            )
            with _lock:
                G.nodes[claim_node_id]["hop_attempted"] = True
                G.nodes[claim_node_id]["hop_fail_reason"] = "no_resolved_id"
            return hop_budget, "no_resolved_id"
        else:
            logger.warning(
                "[GRAPH][HOP]   → raw key NOT matched: LLM citation_raw strings don't match "
                "any cite_span raw field (hallucination or formatting mismatch)",
            )
            with _lock:
                G.nodes[claim_node_id]["hop_attempted"] = True
                G.nodes[claim_node_id]["hop_fail_reason"] = "citation_raw_mismatch"
            return hop_budget, "citation_raw_mismatch"

    top_citation = max(resolved, key=lambda c: c.get("alignment_score", 0.0))
    arxiv_id = top_citation.get("arxiv_id") or ""
    doi = top_citation.get("doi") or ""
    openalex_id = top_citation.get("openalex_id") or ""

    if not arxiv_id and not doi and not openalex_id:
        logger.warning(
            "[GRAPH][HOP] Top citation for claim=%s has no arxiv_id/doi/openalex_id after resolution",
            claim_node_id[:40],
        )
        with _lock:
            G.nodes[claim_node_id]["hop_attempted"] = True
            G.nodes[claim_node_id]["hop_fail_reason"] = "no_resolved_id"
        return hop_budget, "no_resolved_id"

    top_citation_raw = top_citation.get("citation_raw") or ""

    # Qdrant I/O — runs outside the lock so parallel workers don't block each other
    try:
        hop_chunks = retriever.retrieve_hop_chunks(
            arxiv_id=arxiv_id,
            doi=doi,
            openalex_id=openalex_id,
            look_for=look_for,
            embedder=embedder,
            top_k=max_chunks_per_claim,
            citation_raw=top_citation_raw,
        )
    except Exception as exc:
        logger.warning(
            "[GRAPH][HOP] Hop retrieval failed for paper (arxiv=%s doi=%s openalex=%s): %s",
            arxiv_id or "None", doi or "None", openalex_id or "None", exc,
        )
        with _lock:
            G.nodes[claim_node_id]["hop_attempted"] = True
            G.nodes[claim_node_id]["hop_fail_reason"] = "retrieval_error"
        return hop_budget, "retrieval_zero"

    if not hop_chunks:
        logger.warning(
            "[GRAPH][HOP] Retriever returned 0 chunks for claim=%s paper=%s (arxiv=%s doi=%s openalex=%s) look_for=%r — "
            "see [RETRIEVER][HOP] log above for paper-found/not-found diagnosis",
            claim_node_id[:40], doi or arxiv_id or openalex_id,
            arxiv_id or "None", doi or "None", openalex_id or "None", look_for,
        )
        with _lock:
            G.nodes[claim_node_id]["hop_attempted"] = True
            G.nodes[claim_node_id]["hop_fail_reason"] = "retrieval_zero"
        return hop_budget, "retrieval_zero"

    hop_budget -= 1
    paper_id = doi or arxiv_id or openalex_id

    # Graph mutations — serialized under lock
    with _lock:
        if not G.has_node(paper_id):
            G.add_node(paper_id, node_type=NodeType.PAPER.value, text="", paper_id=paper_id)

        for hc in hop_chunks:
            hop_chunk_id = hc.chunk_uid or f"hop:{paper_id}:{hc.chunk_index}"
            if not hop_chunk_id:
                continue
            if not G.has_node(hop_chunk_id):
                G.add_node(
                    hop_chunk_id,
                    node_type=NodeType.CHUNK.value,
                    text=hc.embed_text,
                    paper_id=paper_id,
                    doc_id=paper_id,
                    chunk_id=hop_chunk_id,
                    section=hc.section_title or "",
                    score=hc.score,
                    is_hop=True,
                    hop_reason=hop_reason_raw,
                )
                G.add_edge(hop_chunk_id, paper_id, relation=EdgeRelation.CHUNK_OF.value, score=1.0)
            if not G.has_edge(claim_node_id, hop_chunk_id):
                G.add_edge(
                    claim_node_id,
                    hop_chunk_id,
                    relation=EdgeRelation.HOP_EVIDENCE.value,
                    score=top_citation.get("alignment_score", 1.0),
                )

    logger.info(
        "[GRAPH][HOP] SUCCESS: claim=%s paper=%s (arxiv=%s doi=%s openalex=%s) look_for=%r → %d hop chunk(s) added (budget left=%d)",
        claim_node_id[:40], paper_id, arxiv_id or "None", doi or "None", openalex_id or "None",
        look_for, len(hop_chunks), hop_budget,
    )
    return hop_budget, "success"

