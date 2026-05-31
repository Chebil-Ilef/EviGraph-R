from __future__ import annotations
import json
import networkx as nx
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from evigraph.schemas.objects import EvidenceGraph
from evigraph.visualization.cytoscape_renderer import render_cytoscape_from_data

router = APIRouter()


@router.post("/graph/render", response_class=HTMLResponse)
async def render_graph(graph: EvidenceGraph) -> HTMLResponse:
    html = render_cytoscape_from_data(graph)
    return HTMLResponse(content=html)
