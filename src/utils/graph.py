from __future__ import annotations
import json
import logging
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
        relation = (data.get("relation") or "").lower()
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
        relation = (edge_data.get("relation") or "").lower()
        # SciCite boundary labels indicate multi-hop
        if relation not in EdgeRelation.single_hop():
            return HopDepth.MULTI

        # If the neighbor is itself a claim/concept with further hops
        neighbor_type = dag.nodes[neighbor].get("node_type", "")
        if neighbor_type in ("claim", "concept"):
            return HopDepth.MULTI

    return HopDepth.SINGLE


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
            logger.debug("[GRAPH][HOP] No span found for citation_raw=%r", citation_raw)
            continue

        arxiv_id = (span.get("arxiv_id") or "").strip()
        doi = (span.get("doi") or "").strip()

        if not arxiv_id and not doi:
            logger.debug("[GRAPH][HOP] Span for citation_raw=%r has no usable ID", citation_raw)
            continue

        resolved.append({
            "arxiv_id": arxiv_id,
            "doi": doi,
            "alignment_score": cit.get("alignment_score", 0.0),
            "alignment_reason": cit.get("alignment_reason", ""),
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
) -> int:

    hop_reason_raw = (item.get("hop_reason") or HopReason.NONE.value).strip()
    if hop_reason_raw == HopReason.NONE.value or hop_reason_raw not in HopReason._value2member_map_:
        return hop_budget

    look_for = (item.get("look_for") or "").strip()
    if not look_for:
        return hop_budget

    linked_citations = item.get("linked_citations") or []
    if not linked_citations:
        return hop_budget

    resolved = resolve_cited_paper_id(linked_citations, doc.cite_spans)
    if not resolved:
        return hop_budget

    # only the top-aligned citation for this claim
    top_citation = max(resolved, key=lambda c: c.get("alignment_score", 0.0))
    paper_id = top_citation.get("arxiv_id") or top_citation.get("doi") or ""
    if not paper_id:
        return hop_budget

    try:
        hop_chunks = retriever.retrieve_hop_chunks(
            paper_id_arxiv=paper_id,
            look_for=look_for,
            embedder=embedder,
            top_k=max_chunks_per_claim,
        )
    except Exception as exc:
        logger.warning("[GRAPH][HOP] Hop retrieval failed for paper=%s: %s", paper_id, exc)
        return hop_budget

    if not hop_chunks:
        return hop_budget

    hop_budget -= 1

    # Ensure the cited paper node exists
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
            # Structural link: hop_chunk → cited paper
            G.add_edge(hop_chunk_id, paper_id, relation=EdgeRelation.CHUNK_OF.value, score=1.0)

        # Evidence link: claim → hop_chunk (triggers HopDepth.MULTI in Judge)
        if not G.has_edge(claim_node_id, hop_chunk_id):
            G.add_edge(
                claim_node_id,
                hop_chunk_id,
                relation=EdgeRelation.HOP_EVIDENCE.value,
                score=top_citation.get("alignment_score", 1.0),
            )

    logger.info(
        "[GRAPH][HOP] claim=%s paper=%s look_for=%r → %d hop chunk(s) added (budget left=%d)",
        claim_node_id[:40], paper_id, look_for, len(hop_chunks), hop_budget,
    )
    return hop_budget

