from __future__ import annotations
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock
import networkx as nx
from visualization.cytoscape_renderer import render_cytoscape
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schemas.objects import (
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    NodeType,
    RetrievedDocument,
)
from utils.graph import (
    build_graph_from_documents,
    evidence_graph_from_networkx,
    evidence_graph_to_networkx,
)
from agents.evidence_graph_builder import EvidenceGraphBuilderAgent


def _doc(doc_id, chunk_id, content="Some content.", score=0.8, section=None, cite_spans=None):
    return RetrievedDocument(
        doc_id=doc_id,
        chunk_id=chunk_id,
        content=content,
        score=score,
        section_title=section,
        cite_spans=cite_spans,
    )


class TestBuildGraphFromDocuments:

    def test_chunk_and_paper_nodes_created(self):
        docs = [
            _doc("paper_A", "chunk_1"),
            _doc("paper_A", "chunk_2"),
            _doc("paper_B", "chunk_3"),
        ]
        G = build_graph_from_documents(docs)

        node_ids = set(G.nodes)
        assert "paper_A" in node_ids
        assert "paper_B" in node_ids
        assert "chunk_1" in node_ids
        assert "chunk_2" in node_ids
        assert "chunk_3" in node_ids

    def test_node_count(self):
        docs = [
            _doc("paper_A", "chunk_1"),
            _doc("paper_A", "chunk_2"),
            _doc("paper_B", "chunk_3"),
        ]
        G = build_graph_from_documents(docs)
        # 2 papers + 3 chunks
        assert G.number_of_nodes() == 5

    def test_chunk_of_edges(self):
        docs = [_doc("paper_A", "chunk_1"), _doc("paper_B", "chunk_2")]
        G = build_graph_from_documents(docs)

        assert G.has_edge("chunk_1", "paper_A")
        assert G.edges["chunk_1", "paper_A"]["relation"] == "CHUNK_OF"
        assert G.has_edge("chunk_2", "paper_B")

    def test_node_attributes(self):
        docs = [_doc("paper_A", "chunk_1", content="Hello world.", score=0.95, section="Results")]
        G = build_graph_from_documents(docs)

        chunk_data = G.nodes["chunk_1"]
        assert chunk_data["node_type"] == "chunk"
        assert chunk_data["text"] == "Hello world."
        assert chunk_data["score"] == 0.95
        assert chunk_data["section"] == "Results"
        assert chunk_data["paper_id"] == "paper_A"

        paper_data = G.nodes["paper_A"]
        assert paper_data["node_type"] == "paper"

    def test_cite_spans_create_scicite_edge_from_chunk(self):
        # Edge must originate from the chunk node, not the paper node
        with mock.patch("utils.graph.classify_citation", return_value=("METHOD", 0.91)):
            docs = [
                _doc(
                    "paper_A", "chunk_1",
                    cite_spans={"cite_spans": [{"arxiv_id": "paper_C", "doi": "", "source_ref_id": "ref1", "start": 0, "end": 10}]},
                )
            ]
            G = build_graph_from_documents(docs)

        assert "paper_C" in G.nodes
        assert G.nodes["paper_C"]["node_type"] == "paper"
        assert G.has_edge("chunk_1", "paper_C")
        assert G.edges["chunk_1", "paper_C"]["relation"] == "METHOD"
        assert not G.has_edge("paper_A", "paper_C")

    def test_cite_spans_scicite_label_stored_on_edge(self):
        for label in ("METHOD", "BACKGROUND", "RESULT_COMPARISON"):
            with mock.patch("utils.graph.classify_citation", return_value=(label, 0.80)):
                docs = [
                    _doc(
                        "paper_A", "chunk_1",
                        cite_spans={"cite_spans": [{"arxiv_id": "paper_C", "doi": "", "source_ref_id": "ref1", "start": 0, "end": 5}]},
                    )
                ]
                G = build_graph_from_documents(docs)
            assert G.edges["chunk_1", "paper_C"]["relation"] == label

    def test_cite_spans_confidence_stored_on_edge(self):
        with mock.patch("utils.graph.classify_citation", return_value=("BACKGROUND", 0.77)):
            docs = [
                _doc(
                    "paper_A", "chunk_1",
                    cite_spans={"cite_spans": [{"arxiv_id": "paper_C", "doi": "", "source_ref_id": "ref1", "start": 0, "end": 5}]},
                )
            ]
            G = build_graph_from_documents(docs)
        assert G.edges["chunk_1", "paper_C"]["score"] == pytest.approx(0.77)

    def test_cite_spans_doi_fallback(self):
        with mock.patch("utils.graph.classify_citation", return_value=("BACKGROUND", 0.0)):
            docs = [
                _doc(
                    "paper_A", "chunk_1",
                    cite_spans={"cite_spans": [{"arxiv_id": "", "doi": "10.1000/xyz", "source_ref_id": "ref1", "start": 0, "end": 5}]},
                )
            ]
            G = build_graph_from_documents(docs)

        assert "10.1000/xyz" in G.nodes
        assert G.has_edge("chunk_1", "10.1000/xyz")

    def test_no_flat_cites_paper_to_paper_edges(self):
        # The old paper→paper "cites" edge must never be produced
        with mock.patch("utils.graph.classify_citation", return_value=("BACKGROUND", 0.5)):
            docs = [
                _doc("paper_A", "chunk_1",
                     cite_spans={"cite_spans": [{"arxiv_id": "paper_C", "doi": "", "source_ref_id": "ref1", "start": 0, "end": 5}]}),
            ]
            G = build_graph_from_documents(docs)

        assert not any(
            G.edges[u, v].get("relation") == "cites"
            for u, v in G.edges
        )

    def test_empty_cite_spans_no_scicite_edge(self):
        docs = [
            _doc("paper_A", "chunk_1", cite_spans={"cite_spans": []}),
        ]
        G = build_graph_from_documents(docs)

        scicite_labels = {"METHOD", "BACKGROUND", "RESULT_COMPARISON"}
        assert not any(
            G.edges[u, v].get("relation") in scicite_labels
            for u, v in G.edges
        )

    def test_none_cite_spans_no_crash(self):
        docs = [_doc("paper_A", "chunk_1", cite_spans=None)]
        G = build_graph_from_documents(docs)
        assert G.number_of_nodes() == 2  # paper + chunk

    def test_duplicate_paper_id_single_paper_node(self):
        docs = [_doc("paper_A", "chunk_1"), _doc("paper_A", "chunk_2")]
        G = build_graph_from_documents(docs)

        paper_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "paper"]
        assert len(paper_nodes) == 1

    def test_empty_documents_returns_empty_graph(self):
        G = build_graph_from_documents([])
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0



class TestEvidenceGraphRoundTrip:

    def _make_graph(self):
        return EvidenceGraph(
            nodes=[
                EvidenceNode(node_id="p1", node_type=NodeType.PAPER, text="", doc_id="paper_A"),
                EvidenceNode(node_id="c1", node_type=NodeType.CHUNK, text="Some text.", chunk_id="chunk_1"),
                EvidenceNode(node_id="cl1", node_type=NodeType.CLAIM, text="X achieves 90% F1.", chunk_id="chunk_1"),
            ],
            edges=[
                EvidenceEdge(source="c1", target="p1", relation="CHUNK_OF", score=1.0),
                EvidenceEdge(source="cl1", target="c1", relation="extracted_from", score=1.0),
            ],
        )

    def test_node_count_preserved(self):
        original = self._make_graph()
        G = evidence_graph_to_networkx(original)
        recovered = evidence_graph_from_networkx(G)
        assert len(recovered.nodes) == 3

    def test_edge_count_preserved(self):
        original = self._make_graph()
        G = evidence_graph_to_networkx(original)
        recovered = evidence_graph_from_networkx(G)
        assert len(recovered.edges) == 2

    def test_node_ids_preserved(self):
        original = self._make_graph()
        G = evidence_graph_to_networkx(original)
        recovered = evidence_graph_from_networkx(G)
        assert {n.node_id for n in recovered.nodes} == {"p1", "c1", "cl1"}

    def test_node_types_preserved(self):
        original = self._make_graph()
        G = evidence_graph_to_networkx(original)
        recovered = evidence_graph_from_networkx(G)
        type_map = {n.node_id: n.node_type for n in recovered.nodes}
        assert type_map["p1"] == NodeType.PAPER
        assert type_map["c1"] == NodeType.CHUNK
        assert type_map["cl1"] == NodeType.CLAIM

    def test_edge_relations_preserved(self):
        original = self._make_graph()
        G = evidence_graph_to_networkx(original)
        recovered = evidence_graph_from_networkx(G)
        relations = {(e.source, e.target, e.relation) for e in recovered.edges}
        assert ("c1", "p1", "CHUNK_OF") in relations
        assert ("cl1", "c1", "extracted_from") in relations

    def test_text_preserved(self):
        original = self._make_graph()
        G = evidence_graph_to_networkx(original)
        recovered = evidence_graph_from_networkx(G)
        claim = next(n for n in recovered.nodes if n.node_id == "cl1")
        assert claim.text == "X achieves 90% F1."

    def test_empty_graph_round_trips(self):
        original = EvidenceGraph()
        G = evidence_graph_to_networkx(original)
        recovered = evidence_graph_from_networkx(G)
        assert recovered.nodes == []
        assert recovered.edges == []


class TestRenderCytoscape:

    def test_html_file_is_created(self):
        import networkx as nx
        from visualization.cytoscape_renderer import render_cytoscape

        G = nx.DiGraph()
        G.add_node("paper_A", node_type="paper", text="", paper_id="paper_A")
        G.add_node("chunk_1", node_type="chunk", text="Some text.", section="Introduction", chunk_index=0)
        G.add_edge("chunk_1", "paper_A", relation="CHUNK_OF", score=0.9)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.html"
            render_cytoscape(G, path)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_html_contains_node_data(self):

        G = nx.DiGraph()
        G.add_node("paper_A", node_type="paper", text="", paper_id="paper_A")
        G.add_node("chunk_1", node_type="chunk", text="content", section="Methods")
        G.add_edge("chunk_1", "paper_A", relation="cites")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.html"
            render_cytoscape(G, path)
            html = path.read_text()
            
            assert "paper_A" in html
            assert "chunk_1" in html
            
            assert "paper" in html
            assert "chunk" in html
            
            assert "__CSS__" not in html
            assert "__JS__" not in html
            assert "__NODES__" not in html
            assert "__EDGES__" not in html
            
            assert "cytoscape" in html.lower()

    def test_html_contains_edge_data(self):
        import networkx as nx
        from visualization.cytoscape_renderer import render_cytoscape

        G = nx.DiGraph()
        G.add_node("chunk_1", node_type="chunk", text="text", section="Intro")
        G.add_node("claim_1", node_type="claim", text="Test claim")
        G.add_edge("chunk_1", "claim_1", relation="supports", score=0.8)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.html"
            render_cytoscape(G, path)
            html = path.read_text()
            
            # Check that edge relation is in the HTML
            assert "supports" in html
            assert "0.8" in html or "0.79" in html  # May be rounded


class TestEvidenceGraphBuilderAgent:

    @pytest.fixture
    def mock_llm(self):
        return mock.MagicMock()

    @pytest.fixture
    def agent(self, mock_llm):
        return EvidenceGraphBuilderAgent(llm_client=mock_llm)

    def test_empty_documents_returns_empty_graph(self, agent, mock_llm):
        result = agent.build(query="test", sub_queries=[], documents=[])

        assert isinstance(result, EvidenceGraph)
        assert result.nodes == []
        assert result.edges == []
        mock_llm.chat_text.assert_not_called()

    def test_chunk_and_paper_nodes_present_when_llm_returns_empty(self, agent, mock_llm):
        mock_llm.chat_text.return_value = "[]"
        docs = [_doc("paper_A", "chunk_1"), _doc("paper_B", "chunk_2")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        node_ids = {n.node_id for n in result.nodes}
        assert "paper_A" in node_ids
        assert "paper_B" in node_ids
        assert "chunk_1" in node_ids
        assert "chunk_2" in node_ids

    def test_claim_node_added_from_llm_output(self, agent, mock_llm):
        mock_llm.chat_text.return_value = json.dumps([
            {"text": "BERT achieves 93.5% F1 on SQuAD.", "type": "claim"}
        ])
        docs = [_doc("paper_A", "chunk_1")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        assert len(claim_nodes) == 1
        assert claim_nodes[0].text == "BERT achieves 93.5% F1 on SQuAD."

    def test_concept_node_added_from_llm_output(self, agent, mock_llm):
        mock_llm.chat_text.return_value = json.dumps([
            {"text": "contrastive learning", "type": "concept"}
        ])
        docs = [_doc("paper_A", "chunk_1")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        concept_nodes = [n for n in result.nodes if n.node_type == NodeType.CONCEPT]
        assert len(concept_nodes) == 1

    def test_extracted_from_edge_added(self, agent, mock_llm):
        mock_llm.chat_text.return_value = json.dumps([
            {"text": "X achieves 90%.", "type": "claim"}
        ])
        docs = [_doc("paper_A", "chunk_1")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        extracted_edges = [e for e in result.edges if e.relation == "extracted_from"]
        assert len(extracted_edges) == 1
        assert extracted_edges[0].target == "chunk_1"

    def test_multiple_docs_llm_called_once_per_doc(self, agent, mock_llm):
        mock_llm.chat_text.return_value = "[]"
        docs = [_doc("paper_A", "chunk_1"), _doc("paper_A", "chunk_2"), _doc("paper_B", "chunk_3")]

        agent.build(query="test", sub_queries=[], documents=docs)

        assert mock_llm.chat_text.call_count == 3

    def test_llm_failure_per_chunk_does_not_abort(self, agent, mock_llm):
        mock_llm.chat_text.side_effect = Exception("LLM unavailable")
        docs = [_doc("paper_A", "chunk_1"), _doc("paper_B", "chunk_2")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        # Structural nodes still present despite LLM failures
        node_ids = {n.node_id for n in result.nodes}
        assert "chunk_1" in node_ids
        assert "chunk_2" in node_ids
        assert all(n.node_type != NodeType.CLAIM for n in result.nodes)

    def test_invalid_llm_json_yields_no_claims(self, agent, mock_llm):
        mock_llm.chat_text.return_value = "not json at all"
        docs = [_doc("paper_A", "chunk_1")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        assert claim_nodes == []

    def test_unknown_type_defaults_to_claim(self, agent, mock_llm):
        mock_llm.chat_text.return_value = json.dumps([
            {"text": "Some fact.", "type": "weird_type"}
        ])
        docs = [_doc("paper_A", "chunk_1")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        assert len(claim_nodes) == 1

    def test_empty_text_items_are_skipped(self, agent, mock_llm):
        mock_llm.chat_text.return_value = json.dumps([
            {"text": "", "type": "claim"},
            {"text": "  ", "type": "claim"},
            {"text": "Valid claim.", "type": "claim"},
        ])
        docs = [_doc("paper_A", "chunk_1")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        assert len(claim_nodes) == 1

    def test_returns_evidence_graph_instance(self, agent, mock_llm):
        mock_llm.chat_text.return_value = "[]"
        docs = [_doc("paper_A", "chunk_1")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        assert isinstance(result, EvidenceGraph)


class TestParseClaimsJson:

    def test_valid_array(self):
        raw = json.dumps([{"text": "X is true.", "type": "claim"}])
        result = EvidenceGraphBuilderAgent._parse_claims_json(raw)
        assert len(result) == 1
        assert result[0]["text"] == "X is true."

    def test_markdown_fenced_json(self):
        raw = '```json\n[{"text": "X is true.", "type": "claim"}]\n```'
        result = EvidenceGraphBuilderAgent._parse_claims_json(raw)
        assert len(result) == 1

    def test_array_embedded_in_prose(self):
        raw = 'Here are the claims:\n[{"text": "X.", "type": "claim"}]\nEnd.'
        result = EvidenceGraphBuilderAgent._parse_claims_json(raw)
        assert len(result) == 1

    def test_invalid_json_returns_empty(self):
        result = EvidenceGraphBuilderAgent._parse_claims_json("not json")
        assert result == []

    def test_empty_array_returns_empty(self):
        result = EvidenceGraphBuilderAgent._parse_claims_json("[]")
        assert result == []

    def test_non_dict_items_filtered(self):
        raw = json.dumps(["string item", {"text": "Valid.", "type": "claim"}])
        result = EvidenceGraphBuilderAgent._parse_claims_json(raw)
        assert len(result) == 1

    def test_items_missing_text_filtered(self):
        raw = json.dumps([{"type": "claim"}, {"text": "Valid.", "type": "claim"}])
        result = EvidenceGraphBuilderAgent._parse_claims_json(raw)
        assert len(result) == 1

    def test_subtype_field_preserved(self):
        raw = json.dumps([{"text": "X achieves 93% F1.", "type": "claim", "subtype": "result"}])
        result = EvidenceGraphBuilderAgent._parse_claims_json(raw)
        assert result[0]["subtype"] == "result"


class TestClaimSubtypeOnNode:

    @pytest.fixture
    def mock_llm(self):
        return mock.MagicMock()

    @pytest.fixture
    def agent(self, mock_llm):
        return EvidenceGraphBuilderAgent(llm_client=mock_llm)

    @pytest.mark.parametrize("subtype", ["definition", "method", "result", "assumption"])
    def test_valid_subtype_stored_on_claim_node(self, agent, mock_llm, subtype):
        mock_llm.chat_text.return_value = json.dumps([
            {"text": "Some claim text.", "type": "claim", "subtype": subtype}
        ])
        docs = [_doc("paper_A", "chunk_1")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        assert len(claim_nodes) == 1
        assert claim_nodes[0].metadata.get("claim_subtype") == subtype

    def test_invalid_subtype_not_stored(self, agent, mock_llm):
        mock_llm.chat_text.return_value = json.dumps([
            {"text": "Some claim text.", "type": "claim", "subtype": "nonsense"}
        ])
        docs = [_doc("paper_A", "chunk_1")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        assert len(claim_nodes) == 1
        assert "claim_subtype" not in claim_nodes[0].metadata

    def test_missing_subtype_not_stored(self, agent, mock_llm):
        mock_llm.chat_text.return_value = json.dumps([
            {"text": "Some claim text.", "type": "claim"}
        ])
        docs = [_doc("paper_A", "chunk_1")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        assert len(claim_nodes) == 1
        assert "claim_subtype" not in claim_nodes[0].metadata

    def test_concept_node_has_no_subtype(self, agent, mock_llm):
        mock_llm.chat_text.return_value = json.dumps([
            {"text": "BERT", "type": "concept"}
        ])
        docs = [_doc("paper_A", "chunk_1")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        concept_nodes = [n for n in result.nodes if n.node_type == NodeType.CONCEPT]
        assert len(concept_nodes) == 1
        assert "claim_subtype" not in concept_nodes[0].metadata

    def test_multiple_subtypes_in_one_extraction(self, agent, mock_llm):
        mock_llm.chat_text.return_value = json.dumps([
            {"text": "Dense retrieval encodes queries into embeddings.", "type": "claim", "subtype": "definition"},
            {"text": "Dense retrieval achieves 87% recall@10.", "type": "claim", "subtype": "result"},
            {"text": "dense retrieval", "type": "concept"},
        ])
        docs = [_doc("paper_A", "chunk_1")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        subtypes = {n.metadata.get("claim_subtype") for n in claim_nodes}
        assert subtypes == {"definition", "result"}
