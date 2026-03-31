from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List
import networkx as nx
from schemas.objects import EvidenceGraph, EvidenceNode, EvidenceEdge, NodeType

logger = logging.getLogger(__name__)

_NODE_COLORS = {
    "paper": "#2ecc71",
    "chunk": "#3498db",
    "claim": "#e67e22",
    "concept": "#9b59b6",
}
_DEFAULT_COLOR = "#95a5a6"


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
        G.add_edge(chunk_id, paper_id, relation="belongs_to", score=1.0)

        spans_data = doc.cite_spans or {}
        for span in spans_data.get("cite_spans", []):
            cited_id = span.get("arxiv_id") or span.get("doi") or ""
            if not cited_id:
                continue
            if not G.has_node(cited_id):
                G.add_node(cited_id, node_type="paper", text="", paper_id=cited_id)
            if not G.has_edge(paper_id, cited_id):
                G.add_edge(paper_id, cited_id, relation="cites", score=1.0)

    return G


def render_pyvis(G: "nx.DiGraph", output_path: Path) -> None:

    try:
        from pyvis.network import Network
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyvis is not installed. Run `pip install pyvis` to enable graph visualization."
        ) from exc

    net = Network(directed=True, height="750px", width="100%", bgcolor="#1a1a2e", font_color="white")
    net.barnes_hut()

    for node_id, data in G.nodes(data=True):
        node_type = data.get("node_type", "chunk")
        color = _NODE_COLORS.get(node_type, _DEFAULT_COLOR)
        label = str(node_id)[:50]
        tooltip = (data.get("text") or "")[:300] or str(node_id)
        net.add_node(str(node_id), label=label, color=color, title=tooltip)

    for src, tgt, data in G.edges(data=True):
        net.add_edge(str(src), str(tgt), title=data.get("relation", ""))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path))
    logger.info("[GRAPH] pyvis HTML written to %s", output_path)


def write_graphml(G: "nx.DiGraph", output_path: Path) -> None:
    
    safe_G: nx.DiGraph = nx.DiGraph()

    for node_id, data in G.nodes(data=True):
        safe_attrs = {k: _to_primitive(v) for k, v in data.items()}
        safe_G.add_node(str(node_id), **safe_attrs)

    for src, tgt, data in G.edges(data=True):
        safe_attrs = {k: _to_primitive(v) for k, v in data.items()}
        safe_G.add_edge(str(src), str(tgt), **safe_attrs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(safe_G, str(output_path))
    logger.info("[GRAPH] GraphML written to %s", output_path)


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
