from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock
import networkx as nx
from visualization.cytoscape_renderer import render_cytoscape
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schemas.objects import (
    EdgeRelation,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    HopReason,
    NodeType,
    RetrievedDocument,
)
from utils.graph import (
    build_graph_from_documents,
    evidence_graph_from_networkx,
    evidence_graph_to_networkx,
    resolve_cited_paper_id,
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
            path = Path(tmp) / "graph-before.html"
            render_cytoscape(G, path)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_html_contains_node_data(self):

        G = nx.DiGraph()
        G.add_node("paper_A", node_type="paper", text="", paper_id="paper_A")
        G.add_node("chunk_1", node_type="chunk", text="content", section="Methods")
        G.add_edge("chunk_1", "paper_A", relation="cites")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph-before.html"
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

        G = nx.DiGraph()
        G.add_node("chunk_1", node_type="chunk", text="text", section="Intro")
        G.add_node("claim_1", node_type="claim", text="Test claim")
        G.add_edge("chunk_1", "claim_1", relation="supports", score=0.8)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph-before.html"
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

    def test_duplicate_claim_text_across_chunks_produces_one_node(self, agent, mock_llm):
        same_claim = {"text": "BERT achieves 93.5% F1.", "type": "claim", "subtype": "result"}
        mock_llm.chat_text.return_value = json.dumps([same_claim])
        docs = [_doc("paper_A", "chunk_1"), _doc("paper_A", "chunk_2")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        assert len(claim_nodes) == 1

    def test_duplicate_claim_gets_extracted_from_edge_per_chunk(self, agent, mock_llm):
        same_claim = {"text": "BERT achieves 93.5% F1.", "type": "claim", "subtype": "result"}
        mock_llm.chat_text.return_value = json.dumps([same_claim])
        docs = [_doc("paper_A", "chunk_1"), _doc("paper_A", "chunk_2")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        claim_id = claim_nodes[0].node_id
        extracted_edges = [e for e in result.edges if e.source == claim_id and e.relation == "extracted_from"]
        assert len(extracted_edges) == 2
        assert {e.target for e in extracted_edges} == {"chunk_1", "chunk_2"}

    def test_duplicate_concept_across_chunks_produces_one_node(self, agent, mock_llm):
        mock_llm.chat_text.return_value = json.dumps([{"text": "BERT", "type": "concept"}])
        docs = [_doc("paper_A", "chunk_1"), _doc("paper_B", "chunk_2")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        concept_nodes = [n for n in result.nodes if n.node_type == NodeType.CONCEPT]
        assert len(concept_nodes) == 1

    def test_different_claim_texts_produce_separate_nodes(self, agent, mock_llm):
        def side_effect(*args, **kwargs):
            call_count = mock_llm.chat_text.call_count
            if call_count == 1:
                return json.dumps([{"text": "Claim A.", "type": "claim"}])
            return json.dumps([{"text": "Claim B.", "type": "claim"}])

        mock_llm.chat_text.side_effect = side_effect
        docs = [_doc("paper_A", "chunk_1"), _doc("paper_A", "chunk_2")]

        result = agent.build(query="test", sub_queries=[], documents=docs)

        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        assert len(claim_nodes) == 2

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


class TestSubQueryTagging:

    @pytest.fixture
    def agent(self):
        return EvidenceGraphBuilderAgent(llm_client=mock.MagicMock())

    def _sq(self, text: str):
        from schemas.objects import SubQuery
        return SubQuery(text=text, sections=[], budget_weight=0.5)

    def test_sub_query_indices_stamped_from_doc(self, agent):
        agent.llm_client.chat_text.return_value = json.dumps([
            {"text": "BERT achieves 93.5% F1.", "type": "claim", "subtype": "result"},
        ])
        doc = _doc("paper_A", "chunk_1")
        doc = doc.model_copy(update={"sub_query_indices": [1]})
        sqs = [self._sq("What is dense retrieval?"), self._sq("What are BERT results?")]

        result = agent.build(query="test", sub_queries=sqs, documents=[doc])

        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        assert claim_nodes[0].metadata["sub_query_indices"] == [1]
        assert claim_nodes[0].metadata["sub_query_texts"] == ["What are BERT results?"]

    def test_chunk_from_multiple_sub_queries_carries_all_indices(self, agent):
        agent.llm_client.chat_text.return_value = json.dumps([
            {"text": "Dense retrieval encodes queries.", "type": "claim", "subtype": "definition"},
        ])
        doc = _doc("paper_A", "chunk_1")
        doc = doc.model_copy(update={"sub_query_indices": [0, 2]})
        sqs = [self._sq("What is dense retrieval?"), self._sq("Performance?"), self._sq("How does encoding work?")]

        result = agent.build(query="test", sub_queries=sqs, documents=[doc])

        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        assert claim_nodes[0].metadata["sub_query_indices"] == [0, 2]
        assert claim_nodes[0].metadata["sub_query_texts"] == [
            "What is dense retrieval?", "How does encoding work?"
        ]

    def test_empty_sub_query_indices_on_doc_produces_no_tag(self, agent):
        agent.llm_client.chat_text.return_value = json.dumps([
            {"text": "BERT achieves 93.5% F1.", "type": "claim", "subtype": "result"},
        ])
        doc = _doc("paper_A", "chunk_1")  # sub_query_indices=[] by default
        sqs = [self._sq("What is BERT?")]

        result = agent.build(query="test", sub_queries=sqs, documents=[doc])

        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        assert "sub_query_indices" not in claim_nodes[0].metadata

    def test_concept_nodes_have_no_sub_query_tag(self, agent):
        agent.llm_client.chat_text.return_value = json.dumps([
            {"text": "BERT", "type": "concept"},
        ])
        doc = _doc("paper_A", "chunk_1")
        doc = doc.model_copy(update={"sub_query_indices": [0]})
        sqs = [self._sq("What is BERT?")]

        result = agent.build(query="test", sub_queries=sqs, documents=[doc])

        concept_nodes = [n for n in result.nodes if n.node_type == NodeType.CONCEPT]
        assert "sub_query_indices" not in concept_nodes[0].metadata

    def test_duplicate_claim_from_two_chunks_keeps_first_chunks_indices(self, agent):
        # Same claim text from chunk_1 ([0]) and chunk_2 ([1]) — dedup keeps first node created
        agent.llm_client.chat_text.return_value = json.dumps([
            {"text": "Dense retrieval encodes queries.", "type": "claim", "subtype": "definition"},
        ])
        doc1 = _doc("paper_A", "chunk_1")
        doc1 = doc1.model_copy(update={"sub_query_indices": [0]})
        doc2 = _doc("paper_A", "chunk_2")
        doc2 = doc2.model_copy(update={"sub_query_indices": [1]})
        sqs = [self._sq("What is dense retrieval?"), self._sq("What are evaluation results?")]

        result = agent.build(query="test", sub_queries=sqs, documents=[doc1, doc2])

        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        assert len(claim_nodes) == 1
        assert claim_nodes[0].metadata["sub_query_indices"] == [0]


class TestRendererSerializesClaimMetadata:

    def _render_to_nodes(self, G: nx.DiGraph) -> list[dict]:
        import re
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.html"
            render_cytoscape(G, path)
            html = path.read_text()
        m = re.search(
            r'nodes:\s*(\[.*?\]),\s*\n\s*edges:',
            html,
            re.DOTALL,
        )
        assert m, "GRAPH_DATA.nodes not found in rendered HTML"
        return json.loads(m.group(1))

    def _make_claim_graph(self, **claim_attrs) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_node("paper_A", node_type="paper", text="", paper_id="paper_A")
        G.add_node(
            "chunk_1",
            node_type="chunk",
            text="Some evidence.",
            paper_id="paper_A",
            doc_id="paper_A",
            chunk_id="chunk_1",
            section="Methods",
            score=0.9,
        )
        G.add_node(
            "claim:abc123",
            node_type="claim",
            text="Model achieves 90% accuracy.",
            doc_id="paper_A",
            chunk_id="chunk_1",
            **claim_attrs,
        )
        G.add_edge("chunk_1", "paper_A", relation="CHUNK_OF", score=1.0)
        G.add_edge("claim:abc123", "chunk_1", relation="extracted_from", score=1.0)
        return G

    def _claim_data(self, G: nx.DiGraph) -> dict:
        nodes = self._render_to_nodes(G)
        claim = next(n["data"] for n in nodes if n["data"]["id"] == "claim:abc123")
        return claim


    def test_sub_query_indices_serialized(self):
        G = self._make_claim_graph(sub_query_indices=[0, 2])
        d = self._claim_data(G)
        assert d["sub_query_indices"] == [0, 2]

    def test_sub_query_texts_serialized(self):
        G = self._make_claim_graph(
            sub_query_indices=[1],
            sub_query_texts=["What methods are used?"],
        )
        d = self._claim_data(G)
        assert d["sub_query_texts"] == ["What methods are used?"]

    def test_missing_sub_query_fields_serialize_as_null(self):
        # Claim with no sub-query provenance — fields must still be present in
        # the node data (as null) so the JS can safely check their existence.
        G = self._make_claim_graph()
        d = self._claim_data(G)
        assert "sub_query_indices" in d
        assert d["sub_query_indices"] is None
        assert "sub_query_texts" in d
        assert d["sub_query_texts"] is None


    def test_verdict_serialized(self):
        G = self._make_claim_graph(verdict="Supported", verifier_used="nli")
        d = self._claim_data(G)
        assert d["verdict"] == "Supported"

    def test_verifier_used_serialized(self):
        G = self._make_claim_graph(verdict="Supported", verifier_used="nli→llm")
        d = self._claim_data(G)
        assert d["verifier_used"] == "nli→llm"

    def test_claim_type_serialized(self):
        G = self._make_claim_graph(verdict="Supported", claim_type="inferential")
        d = self._claim_data(G)
        assert d["claim_type"] == "inferential"

    def test_hop_depth_serialized(self):
        G = self._make_claim_graph(verdict="Supported", hop_depth="single")
        d = self._claim_data(G)
        assert d["hop_depth"] == "single"

    def test_missing_verdict_serializes_as_null(self):
        G = self._make_claim_graph()
        d = self._claim_data(G)
        assert "verdict" in d
        assert d["verdict"] is None

    @pytest.mark.parametrize("verdict", ["Supported", "Contradicted", "Not-Supported", "Inconclusive"])
    def test_all_verdict_values_round_trip(self, verdict):
        G = self._make_claim_graph(verdict=verdict)
        d = self._claim_data(G)
        assert d["verdict"] == verdict


# ── HopReason schema ──────────────────────────────────────────────────────────

class TestHopReasonEnum:

    def test_all_values_exist(self):
        values = {r.value for r in HopReason}
        assert "none" in values
        assert "missing_scope_context" in values
        assert "missing_comparison_baseline" in values
        assert "missing_method_origin" in values
        assert "missing_definition_context" in values

    def test_hop_evidence_in_edge_relation(self):
        assert EdgeRelation.HOP_EVIDENCE.value == "hop_evidence"

    def test_hop_evidence_not_in_single_hop(self):
        assert "hop_evidence" not in EdgeRelation.single_hop()

    def test_hop_evidence_in_evidence_relations(self):
        assert "hop_evidence" in EdgeRelation.evidence_relations()

    def test_extracted_from_still_in_evidence_relations(self):
        assert "extracted_from" in EdgeRelation.evidence_relations()


# ── resolve_cited_paper_id ────────────────────────────────────────────────────

class TestResolveCitedPaperId:  # lives in utils.graph

    def _spans(self, *entries):
        """Build a cite_spans dict from (raw, arxiv_id, doi) tuples."""
        return {
            "cite_spans": [
                {"raw": r, "arxiv_id": a, "doi": d, "start": 0, "end": 5}
                for r, a, d in entries
            ]
        }

    def test_exact_raw_match_returns_arxiv_id(self):
        spans = self._spans(("Smith et al. BEIR 2021.", "2104.08663", ""))
        result = resolve_cited_paper_id(
            [{"citation_raw": "Smith et al. BEIR 2021.", "alignment_score": 0.9, "alignment_reason": "scope"}],
            spans,
        )
        assert len(result) == 1
        assert result[0]["arxiv_id"] == "2104.08663"

    def test_exact_raw_match_returns_doi(self):
        spans = self._spans(("Jones et al. DPR. ACL 2020.", "", "10.1/xyz"))
        result = resolve_cited_paper_id(
            [{"citation_raw": "Jones et al. DPR. ACL 2020.", "alignment_score": 0.85, "alignment_reason": "x"}],
            spans,
        )
        assert len(result) == 1
        assert result[0]["doi"] == "10.1/xyz"

    def test_wrong_raw_returns_empty(self):
        spans = self._spans(("Smith et al. BEIR 2021.", "1234.5678", ""))
        result = resolve_cited_paper_id(
            [{"citation_raw": "completely different raw string", "alignment_score": 0.9, "alignment_reason": "x"}],
            spans,
        )
        assert result == []

    def test_no_cite_spans_returns_empty(self):
        result = resolve_cited_paper_id(
            [{"citation_raw": "Smith et al. BEIR 2021.", "alignment_score": 0.9, "alignment_reason": "x"}],
            None,
        )
        assert result == []

    def test_empty_cite_spans_returns_empty(self):
        result = resolve_cited_paper_id(
            [{"citation_raw": "Smith et al. BEIR 2021.", "alignment_score": 0.9, "alignment_reason": "x"}],
            {"cite_spans": []},
        )
        assert result == []

    def test_span_with_no_id_skipped(self):
        spans = self._spans(("Smith et al. BEIR 2021.", "", ""))
        result = resolve_cited_paper_id(
            [{"citation_raw": "Smith et al. BEIR 2021.", "alignment_score": 0.9, "alignment_reason": "x"}],
            spans,
        )
        assert result == []

    def test_empty_linked_citations_returns_empty(self):
        spans = self._spans(("Some Paper raw.", "1234.0000", ""))
        result = resolve_cited_paper_id([], spans)
        assert result == []

    def test_alignment_score_preserved(self):
        spans = self._spans(("Smith et al. BEIR 2021.", "2104.08663", ""))
        result = resolve_cited_paper_id(
            [{"citation_raw": "Smith et al. BEIR 2021.", "alignment_score": 0.77, "alignment_reason": "x"}],
            spans,
        )
        assert result[0]["alignment_score"] == pytest.approx(0.77)

    def test_multiple_citations_matched_independently(self):
        spans = self._spans(
            ("Alpha et al. Paper. 2020.", "alpha.001", ""),
            ("Beta et al. Paper. 2021.", "beta.002", ""),
        )
        result = resolve_cited_paper_id(
            [
                {"citation_raw": "Alpha et al. Paper. 2020.", "alignment_score": 0.8, "alignment_reason": "x"},
                {"citation_raw": "Beta et al. Paper. 2021.", "alignment_score": 0.9, "alignment_reason": "y"},
            ],
            spans,
        )
        assert len(result) == 2
        arxiv_ids = {r["arxiv_id"] for r in result}
        assert arxiv_ids == {"alpha.001", "beta.002"}


# ── Multi-hop graph construction ──────────────────────────────────────────────

def _make_mock_retriever(hop_chunks):
    """Return a mock retriever whose retrieve_hop_chunks returns hop_chunks."""
    from retrieval.retriever import ChunkResult
    retriever = mock.MagicMock()
    retriever.retrieve_hop_chunks.return_value = hop_chunks
    return retriever


def _make_mock_embedder():
    embedder = mock.MagicMock()
    embedder.embed_query.return_value = [0.1] * 1024
    return embedder


def _hop_chunk_result(chunk_uid: str, paper_id: str, text: str = "hop evidence text"):
    from retrieval.retriever import ChunkResult
    return ChunkResult(
        chunk_uid=chunk_uid,
        paper_id=paper_id,
        score=0.82,
        embed_text=text,
        section_title="Results",
        chunk_index=0,
        total_chunks=5,
    )


class TestHopGraphConstruction:
    """Verify that hop evidence nodes and edges are wired correctly."""

    def _build_with_hop(self, llm_response: str, cite_spans: dict, hop_chunks):
        mock_llm = mock.MagicMock()
        mock_llm.chat_text.return_value = llm_response
        retriever = _make_mock_retriever(hop_chunks)
        embedder = _make_mock_embedder()
        agent = EvidenceGraphBuilderAgent(
            llm_client=mock_llm,
            retriever=retriever,
            embedder=embedder,
        )
        doc = _doc(
            "paper_A", "chunk_1",
            content="Contrastive models improve Recall@10 by 4.2% over BM25 on BEIR [Smith 2021].",
            cite_spans=cite_spans,
        )
        return agent.build(query="test", sub_queries=[], documents=[doc])

    def _hop_llm_response(self, hop_reason="missing_scope_context"):
        return json.dumps([{
            "text": "Contrastive models improve Recall@10 by 4.2% over BM25 on BEIR.",
            "type": "claim",
            "subtype": "result",
            "linked_citations": [{"citation_raw": "Smith et al. BEIR 2021.", "alignment_score": 0.88, "alignment_reason": "scope"}],
            "hop_reason": hop_reason,
            "look_for": "definition and scope of BEIR benchmark",
        }])

    def _cite_spans_with_beir(self):
        return {"cite_spans": [{"raw": "Smith et al. BEIR 2021.", "arxiv_id": "2104.08663", "doi": "", "start": 0, "end": 5}]}

    # ── hop nodes present ────────────────────────────────────────────────────

    def test_hop_chunk_node_added_to_graph(self):
        hop = _hop_chunk_result("hop_chunk_uid_1", "2104.08663")
        result = self._build_with_hop(self._hop_llm_response(), self._cite_spans_with_beir(), [hop])
        node_ids = {n.node_id for n in result.nodes}
        assert "hop_chunk_uid_1" in node_ids

    def test_hop_chunk_node_type_is_chunk(self):
        hop = _hop_chunk_result("hop_chunk_uid_1", "2104.08663")
        result = self._build_with_hop(self._hop_llm_response(), self._cite_spans_with_beir(), [hop])
        hop_node = next(n for n in result.nodes if n.node_id == "hop_chunk_uid_1")
        assert hop_node.node_type == NodeType.CHUNK

    def test_hop_chunk_node_carries_is_hop_flag(self):
        hop = _hop_chunk_result("hop_chunk_uid_1", "2104.08663")
        result = self._build_with_hop(self._hop_llm_response(), self._cite_spans_with_beir(), [hop])
        hop_node = next(n for n in result.nodes if n.node_id == "hop_chunk_uid_1")
        assert hop_node.metadata.get("is_hop") is True

    def test_hop_chunk_node_carries_hop_reason(self):
        hop = _hop_chunk_result("hop_chunk_uid_1", "2104.08663")
        result = self._build_with_hop(self._hop_llm_response("missing_scope_context"), self._cite_spans_with_beir(), [hop])
        hop_node = next(n for n in result.nodes if n.node_id == "hop_chunk_uid_1")
        assert hop_node.metadata.get("hop_reason") == "missing_scope_context"

    def test_cited_paper_node_present(self):
        hop = _hop_chunk_result("hop_chunk_uid_1", "2104.08663")
        result = self._build_with_hop(self._hop_llm_response(), self._cite_spans_with_beir(), [hop])
        node_ids = {n.node_id for n in result.nodes}
        assert "2104.08663" in node_ids

    # ── hop edges present ────────────────────────────────────────────────────

    def test_claim_to_hop_chunk_edge_exists(self):
        hop = _hop_chunk_result("hop_chunk_uid_1", "2104.08663")
        result = self._build_with_hop(self._hop_llm_response(), self._cite_spans_with_beir(), [hop])
        hop_evidence_edges = [
            e for e in result.edges if e.relation == EdgeRelation.HOP_EVIDENCE.value
        ]
        assert len(hop_evidence_edges) == 1
        assert hop_evidence_edges[0].target == "hop_chunk_uid_1"

    def test_hop_evidence_edge_score_matches_alignment(self):
        hop = _hop_chunk_result("hop_chunk_uid_1", "2104.08663")
        result = self._build_with_hop(self._hop_llm_response(), self._cite_spans_with_beir(), [hop])
        edge = next(e for e in result.edges if e.relation == EdgeRelation.HOP_EVIDENCE.value)
        assert edge.score == pytest.approx(0.88)

    def test_hop_chunk_to_cited_paper_chunk_of_edge(self):
        hop = _hop_chunk_result("hop_chunk_uid_1", "2104.08663")
        result = self._build_with_hop(self._hop_llm_response(), self._cite_spans_with_beir(), [hop])
        chunk_of_edges = [e for e in result.edges if e.relation == "CHUNK_OF"]
        targets = {e.target for e in chunk_of_edges}
        assert "2104.08663" in targets

    def test_original_extracted_from_edge_still_present(self):
        hop = _hop_chunk_result("hop_chunk_uid_1", "2104.08663")
        result = self._build_with_hop(self._hop_llm_response(), self._cite_spans_with_beir(), [hop])
        extracted_edges = [e for e in result.edges if e.relation == "extracted_from"]
        assert any(e.target == "chunk_1" for e in extracted_edges)

    def test_two_hop_chunks_both_wired(self):
        hops = [
            _hop_chunk_result("hop_c1", "2104.08663", "text A"),
            _hop_chunk_result("hop_c2", "2104.08663", "text B"),
        ]
        result = self._build_with_hop(self._hop_llm_response(), self._cite_spans_with_beir(), hops)
        hop_edges = [e for e in result.edges if e.relation == EdgeRelation.HOP_EVIDENCE.value]
        assert len(hop_edges) == 2
        hop_targets = {e.target for e in hop_edges}
        assert hop_targets == {"hop_c1", "hop_c2"}

    # ── no-hop cases ─────────────────────────────────────────────────────────

    def test_hop_reason_none_skips_retrieval(self):
        mock_llm = mock.MagicMock()
        mock_llm.chat_text.return_value = json.dumps([{
            "text": "DeBERTa achieves 90.9 F1 on SQuAD 2.0.",
            "type": "claim",
            "subtype": "result",
            "linked_citations": [],
            "hop_reason": "none",
            "look_for": "",
        }])
        retriever = _make_mock_retriever([])
        embedder = _make_mock_embedder()
        agent = EvidenceGraphBuilderAgent(llm_client=mock_llm, retriever=retriever, embedder=embedder)
        doc = _doc("paper_A", "chunk_1", content="DeBERTa achieves 90.9 F1 on SQuAD 2.0.")
        agent.build(query="test", sub_queries=[], documents=[doc])
        retriever.retrieve_hop_chunks.assert_not_called()

    def test_empty_look_for_skips_retrieval(self):
        mock_llm = mock.MagicMock()
        mock_llm.chat_text.return_value = json.dumps([{
            "text": "Some claim.",
            "type": "claim",
            "subtype": "result",
            "linked_citations": [{"citation_raw": "Smith et al. BEIR 2021.", "alignment_score": 0.9, "alignment_reason": "x"}],
            "hop_reason": "missing_scope_context",
            "look_for": "",  # empty → skip
        }])
        retriever = _make_mock_retriever([])
        embedder = _make_mock_embedder()
        agent = EvidenceGraphBuilderAgent(llm_client=mock_llm, retriever=retriever, embedder=embedder)
        doc = _doc("paper_A", "chunk_1", content="Some claim.")
        agent.build(query="test", sub_queries=[], documents=[doc])
        retriever.retrieve_hop_chunks.assert_not_called()

    def test_unresolvable_citation_skips_retrieval(self):
        mock_llm = mock.MagicMock()
        mock_llm.chat_text.return_value = json.dumps([{
            "text": "Some claim.",
            "type": "claim",
            "subtype": "result",
            "linked_citations": [{"citation_raw": "Ghost Paper raw string not in spans.", "alignment_score": 0.9, "alignment_reason": "x"}],
            "hop_reason": "missing_comparison_baseline",
            "look_for": "baseline result",
        }])
        retriever = _make_mock_retriever([])
        embedder = _make_mock_embedder()
        agent = EvidenceGraphBuilderAgent(llm_client=mock_llm, retriever=retriever, embedder=embedder)
        doc = _doc("paper_A", "chunk_1",
                   content="Some claim.",
                   cite_spans={"cite_spans": [{"raw": "Completely different raw.", "arxiv_id": "9999.0000", "doi": ""}]})
        agent.build(query="test", sub_queries=[], documents=[doc])
        retriever.retrieve_hop_chunks.assert_not_called()

    def test_no_retriever_no_hop(self):
        mock_llm = mock.MagicMock()
        mock_llm.chat_text.return_value = json.dumps([{
            "text": "Claim requiring hop.",
            "type": "claim",
            "subtype": "result",
            "linked_citations": [{"citation_raw": "Smith et al. BEIR 2021.", "alignment_score": 0.9, "alignment_reason": "x"}],
            "hop_reason": "missing_scope_context",
            "look_for": "BEIR benchmark scope",
        }])
        # No retriever/embedder passed
        agent = EvidenceGraphBuilderAgent(llm_client=mock_llm)
        doc = _doc("paper_A", "chunk_1",
                   content="Claim.",
                   cite_spans={"cite_spans": [{"raw": "Smith et al. BEIR 2021.", "arxiv_id": "2104.08663", "doi": ""}]})
        result = agent.build(query="test", sub_queries=[], documents=[doc])
        hop_edges = [e for e in result.edges if e.relation == EdgeRelation.HOP_EVIDENCE.value]
        assert hop_edges == []

    def test_concept_nodes_never_trigger_hop(self):
        mock_llm = mock.MagicMock()
        mock_llm.chat_text.return_value = json.dumps([
            {"text": "BEIR", "type": "concept"},
        ])
        retriever = _make_mock_retriever([_hop_chunk_result("hop_c1", "2104.08663")])
        embedder = _make_mock_embedder()
        agent = EvidenceGraphBuilderAgent(llm_client=mock_llm, retriever=retriever, embedder=embedder)
        doc = _doc("paper_A", "chunk_1", content="See BEIR benchmark.")
        agent.build(query="test", sub_queries=[], documents=[doc])
        retriever.retrieve_hop_chunks.assert_not_called()

    # ── budget guard ─────────────────────────────────────────────────────────

    def test_hop_budget_limits_total_hops(self):
        """With MAX_HOPS_PER_BUILD=5, only 5 hop retrievals fire across all claims."""
        from config import settings as settings_module

        hop_claim = {
            "text": "Claim {i}.",
            "type": "claim",
            "subtype": "result",
            "linked_citations": [{"citation_raw": "Smith et al. BEIR 2021.", "alignment_score": 0.9, "alignment_reason": "x"}],
            "hop_reason": "missing_scope_context",
            "look_for": "scope",
        }
        # 8 distinct claims, each requesting a hop
        claims = [dict(hop_claim, text=f"Claim {i}.") for i in range(8)]
        mock_llm = mock.MagicMock()
        mock_llm.chat_text.return_value = json.dumps(claims)

        retriever = _make_mock_retriever([_hop_chunk_result(f"hop_{i}", "2104.08663") for i in range(2)])
        embedder = _make_mock_embedder()

        with mock.patch("agents.evidence_graph_builder.GRAPH_CONFIG") as mock_cfg:
            mock_cfg.hop_max_per_build = 3
            mock_cfg.hop_max_chunks_per_claim = 2
            agent = EvidenceGraphBuilderAgent(llm_client=mock_llm, retriever=retriever, embedder=embedder)
            doc = _doc("paper_A", "chunk_1",
                       content="Many claims.",
                       cite_spans={"cite_spans": [{"raw": "Smith et al. BEIR 2021.", "arxiv_id": "2104.08663", "doi": ""}]})
            agent.build(query="test", sub_queries=[], documents=[doc])

        # retrieve_hop_chunks called at most 3 times despite 8 eligible claims
        assert retriever.retrieve_hop_chunks.call_count <= 3

    # ── retriever failure resilience ─────────────────────────────────────────

    def test_retriever_exception_does_not_abort_build(self):
        mock_llm = mock.MagicMock()
        mock_llm.chat_text.return_value = json.dumps([{
            "text": "Some claim.",
            "type": "claim",
            "subtype": "result",
            "linked_citations": [{"citation_raw": "Smith et al. BEIR 2021.", "alignment_score": 0.9, "alignment_reason": "x"}],
            "hop_reason": "missing_scope_context",
            "look_for": "BEIR scope",
        }])
        retriever = mock.MagicMock()
        retriever.retrieve_hop_chunks.side_effect = RuntimeError("Qdrant down")
        embedder = _make_mock_embedder()
        agent = EvidenceGraphBuilderAgent(llm_client=mock_llm, retriever=retriever, embedder=embedder)
        doc = _doc("paper_A", "chunk_1",
                   content="Some claim.",
                   cite_spans={"cite_spans": [{"raw": "Smith et al. BEIR 2021.", "arxiv_id": "2104.08663", "doi": ""}]})
        result = agent.build(query="test", sub_queries=[], documents=[doc])
        # Build completes and returns a valid graph
        assert isinstance(result, EvidenceGraph)
        claim_nodes = [n for n in result.nodes if n.node_type == NodeType.CLAIM]
        assert len(claim_nodes) == 1

    def test_hop_retriever_returns_empty_no_hop_edges(self):
        mock_llm = mock.MagicMock()
        mock_llm.chat_text.return_value = json.dumps([{
            "text": "Some claim.",
            "type": "claim",
            "subtype": "result",
            "linked_citations": [{"citation_raw": "Smith et al. BEIR 2021.", "alignment_score": 0.9, "alignment_reason": "x"}],
            "hop_reason": "missing_scope_context",
            "look_for": "scope",
        }])
        retriever = _make_mock_retriever([])  # returns nothing
        embedder = _make_mock_embedder()
        agent = EvidenceGraphBuilderAgent(llm_client=mock_llm, retriever=retriever, embedder=embedder)
        doc = _doc("paper_A", "chunk_1",
                   content="Some claim.",
                   cite_spans={"cite_spans": [{"raw": "Smith et al. BEIR 2021.", "arxiv_id": "2104.08663", "doi": ""}]})
        result = agent.build(query="test", sub_queries=[], documents=[doc])
        hop_edges = [e for e in result.edges if e.relation == EdgeRelation.HOP_EVIDENCE.value]
        assert hop_edges == []


# ── Prompt: cited titles injected ────────────────────────────────────────────

class TestClaimExtractionPromptCiteTitles:

    def test_raw_injected_into_prompt(self):
        from config.prompts import build_claim_extraction_prompt
        doc = _doc(
            "paper_A", "chunk_1",
            content="See results from BEIR [Smith 2021].",
            cite_spans={"cite_spans": [
                {"raw": "Smith et al. BEIR: A Heterogeneous Benchmark. NeurIPS 2021.", "arxiv_id": "2104.08663", "doi": "", "start": 0, "end": 5}
            ]},
        )
        prompt = build_claim_extraction_prompt(doc)
        assert "Smith et al. BEIR: A Heterogeneous Benchmark. NeurIPS 2021." in prompt
        assert "Available citations:" in prompt

    def test_no_cite_spans_shows_none(self):
        from config.prompts import build_claim_extraction_prompt
        doc = _doc("paper_A", "chunk_1", content="Plain text.")
        prompt = build_claim_extraction_prompt(doc)
        assert "Available citations: none" in prompt

    def test_empty_cite_spans_shows_none(self):
        from config.prompts import build_claim_extraction_prompt
        doc = _doc("paper_A", "chunk_1", content="Plain text.", cite_spans={"cite_spans": []})
        prompt = build_claim_extraction_prompt(doc)
        assert "Available citations: none" in prompt

    def test_spans_without_raw_not_included(self):
        from config.prompts import build_claim_extraction_prompt
        doc = _doc(
            "paper_A", "chunk_1",
            content="Something.",
            cite_spans={"cite_spans": [{"raw": "", "arxiv_id": "1234.0000", "doi": ""}]},
        )
        prompt = build_claim_extraction_prompt(doc)
        assert "Available citations: none" in prompt

    def test_multiple_raws_all_listed(self):
        from config.prompts import build_claim_extraction_prompt
        doc = _doc(
            "paper_A", "chunk_1",
            content="Text.",
            cite_spans={"cite_spans": [
                {"raw": "Alpha et al. Paper A. 2020.", "arxiv_id": "0001.0001", "doi": ""},
                {"raw": "Beta et al. Paper B. 2021.", "arxiv_id": "0002.0002", "doi": ""},
            ]},
        )
        prompt = build_claim_extraction_prompt(doc)
        assert "Alpha et al. Paper A. 2020." in prompt
        assert "Beta et al. Paper B. 2021." in prompt
