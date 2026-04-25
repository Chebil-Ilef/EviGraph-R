from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from schemas.objects import EvidenceGraph



def _display_label(node_id: str, data: dict) -> str:
    ntype = data.get("node_type", "")
    text  = (data.get("text") or "").strip()

    if ntype == "paper":
        return data.get("paper_id") or node_id[:16]

    if ntype == "chunk":
        section = data.get("section") or data.get("section_title") or ""
        idx     = data.get("chunk_index")
        label   = section[:22] if section else "chunk"
        return f"{label} · {idx}" if idx is not None else label

    if ntype in ("claim", "concept"):
        # Very short hash — just 6 chars so it fits comfortably in the circle.
        # Full content is shown in the inspector panel when clicked.
        return node_id[-6:]

    return node_id[:20]


def _display_id(node_id: str, data: dict) -> str:
    ntype = data.get("node_type", "")
    if ntype == "paper":
        return data.get("paper_id") or node_id
    return node_id[:32] + ("…" if len(node_id) > 32 else "")



def _load_template_file(filename: str) -> str:

    module_dir = Path(__file__).parent
    filepath = module_dir / "templates" / filename if "html" in filename else module_dir / "static" / filename
    return filepath.read_text(encoding="utf-8")


def _serialise_graph(G: "nx.DiGraph") -> tuple[list[dict], list[dict]]:
    nodes: list[dict] = []
    edges: list[dict] = []

    for node_id, data in G.nodes(data=True):
        nid   = str(node_id)
        ntype = data.get("node_type", "chunk")
        label = _display_label(nid, data)
        did   = _display_id(nid, data)

        # strip raw latex / very long text for the tooltip field
        full_text = (data.get("text") or "").strip()
        safe_text = re.sub(r"\s+", " ", full_text)[:800]

        node_data: dict = {
            "id":        nid,
            "type":      ntype,
            "label":     label,
            "display_id": did,
            "full_text": safe_text,
            # extra metadata surfaced in inspector
            "paper_id":     data.get("paper_id") or data.get("doc_id") or "",
            "section":      data.get("section") or data.get("section_title") or "",
            "chunk_index":  data.get("chunk_index"),
            "total_chunks": data.get("total_chunks"),
            "score":        round(float(data.get("score") or 0.0), 4),
            # sub-query provenance (populated during graph building)
            "sub_query_indices": data.get("sub_query_indices"),
            "sub_query_texts":   data.get("sub_query_texts"),
            # verdict metadata (populated after judging)
            "verdict":      data.get("verdict"),
            "verifier_used": data.get("verifier_used"),
            "claim_type":   data.get("claim_type"),
            "hop_depth":    data.get("hop_depth"),
            "reason":       data.get("reason"),
        }
        nodes.append({"data": node_data})

    for i, (src, tgt, data) in enumerate(G.edges(data=True)):
        relation = data.get("relation") or "related"
        score    = round(float(data.get("score") or 0.0), 4)
        edges.append({
            "data": {
                "id":       f"e{i}",
                "source":   str(src),
                "target":   str(tgt),
                "relation": relation,
                "score":    score,
            }
        })

    return nodes, edges


def render_cytoscape_from_data(graph: "EvidenceGraph") -> str:
    nodes: list[dict] = []
    edges: list[dict] = []

    for node in graph.nodes:
        nid   = node.node_id
        ntype = node.node_type.value
        meta  = node.metadata or {}

        label = _display_label(nid, {"node_type": ntype, "text": node.text, **meta})
        did   = _display_id(nid, {"node_type": ntype, **meta})

        import re
        safe_text = re.sub(r"\s+", " ", (node.text or "").strip())[:800]

        nodes.append({"data": {
            "id":           nid,
            "type":         ntype,
            "label":        label,
            "display_id":   did,
            "full_text":    safe_text,
            "paper_id":     node.doc_id or meta.get("paper_id", ""),
            "section":      meta.get("section") or meta.get("section_title", ""),
            "chunk_index":  meta.get("chunk_index"),
            "total_chunks": meta.get("total_chunks"),
            "score":        round(float(meta.get("score") or 0.0), 4),
            "sub_query_indices": meta.get("sub_query_indices"),
            "sub_query_texts":   meta.get("sub_query_texts"),
            "verdict":      meta.get("verdict"),
            "verifier_used": meta.get("verifier_used"),
            "claim_type":   meta.get("claim_type"),
            "hop_depth":    meta.get("hop_depth"),
            "reason":       meta.get("reason"),
        }})

    for i, edge in enumerate(graph.edges):
        edges.append({"data": {
            "id":       f"e{i}",
            "source":   edge.source,
            "target":   edge.target,
            "relation": edge.relation,
            "score":    round(float(edge.score or 0.0), 4),
        }})

    is_judged = any(
        n["data"].get("verdict")
        for n in nodes
        if n["data"].get("type") == "claim"
    )

    template = _load_template_file("cytoscape_graph.html")
    css      = _load_template_file("cytoscape_graph.css")
    js       = _load_template_file("cytoscape_graph.js")

    html = template.replace("__CSS__", css).replace("__JS__", js)
    html = html.replace("__NODES__",     json.dumps(nodes,  ensure_ascii=False))
    html = html.replace("__EDGES__",     json.dumps(edges,  ensure_ascii=False))
    html = html.replace("__IS_JUDGED__", "true" if is_judged else "false")
    return html


def render_cytoscape(G: "nx.DiGraph", output_path: Path) -> None:
    logger = logging.getLogger(__name__)

    nodes, edges = _serialise_graph(G)

    # Detect whether this is a post-judging graph (any claim node has a verdict)
    is_judged = any(
        n["data"].get("verdict") for n in nodes if n["data"].get("type") == "claim"
    )

    # Load template and assets
    template = _load_template_file("cytoscape_graph.html")
    css = _load_template_file("cytoscape_graph.css")
    js = _load_template_file("cytoscape_graph.js")

    # Inject CSS and JS into template
    html = template.replace("__CSS__", css)
    html = html.replace("__JS__", js)

    # Inject graph data
    html = html.replace("__NODES__", json.dumps(nodes, ensure_ascii=False))
    html = html.replace("__EDGES__", json.dumps(edges, ensure_ascii=False))
    html = html.replace("__IS_JUDGED__", "true" if is_judged else "false")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("[GRAPH] Cytoscape HTML written to %s", output_path)
