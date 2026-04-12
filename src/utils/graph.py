from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List
import networkx as nx
from schemas.objects import EvidenceGraph, EvidenceNode, EvidenceEdge, NodeType, HopDepth, EdgeRelation
from utils.scicite import classify_citation
from visualization.cytoscape_renderer import render_cytoscape  # noqa: F401 — re-exported
import time

logger = logging.getLogger(__name__)

_MAX_TRAVERSAL_DEPTH = 5


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


def backwards_traverse(claim_id: str, dag: nx.DiGraph, max_depth: int = _MAX_TRAVERSAL_DEPTH) -> list[dict]:

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
