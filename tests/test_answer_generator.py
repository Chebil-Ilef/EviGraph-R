from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.answer_generator import AnswerGeneratorAgent
from schemas.objects import (
    AnnotatedSentence,
    Citation,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    FinalAnswer,
    NodeType,
    RetrievedDocument,
)

def _doc(chunk_id: str, content: str, doc_id: str = "paper_1", section: str | None = "Methods") -> RetrievedDocument:
    return RetrievedDocument(doc_id=doc_id, chunk_id=chunk_id, content=content, score=0.9, section_title=section)


def _node(node_id: str, node_type: NodeType, text: str = "", chunk_id: str | None = None, **kw) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        node_type=node_type,
        text=text,
        chunk_id=chunk_id or node_id,
        metadata=kw,
    )


def _edge(src: str, tgt: str, relation: str = "extracted_from", score: float = 0.85) -> EvidenceEdge:
    return EvidenceEdge(source=src, target=tgt, relation=relation, score=score)


def _graph_with_claims() -> tuple[EvidenceGraph, list[RetrievedDocument]]:

    nodes = [
        _node("p1", NodeType.PAPER),
        _node("ch1", NodeType.CHUNK, "Dropout noise used as augmentation.", chunk_id="ch1"),
        _node("claim:ch1:0", NodeType.CLAIM, "Contrastive learning uses dropout as augmentation.", chunk_id="ch1"),
        _node("claim:ch1:1", NodeType.CLAIM, "Training improves out-of-domain retrieval.", chunk_id="ch1"),
    ]
    edges = [
        _edge("claim:ch1:0", "ch1", "METHOD", 0.91),
        _edge("claim:ch1:1", "ch1", "extracted_from", 0.80),
        _edge("ch1", "p1", "belongs_to"),
    ]
    graph = EvidenceGraph(nodes=nodes, edges=edges)
    docs = [_doc("ch1", "Dropout noise used as augmentation.", doc_id="arxiv:2104.08821")]
    return graph, docs


def _llm_response(sentences: list[dict]) -> str:
    return json.dumps({"sentences": sentences})


@pytest.fixture
def mock_llm():
    return mock.MagicMock()


@pytest.fixture
def agent(mock_llm):
    return AnswerGeneratorAgent(llm_client=mock_llm)


class TestCollectClaims:

    def test_returns_claim_and_chunk_nodes(self, agent):
        graph, docs = _graph_with_claims()
        claims = agent._collect_claims(graph, docs)
        texts = [c["text"] for c in claims]
        assert "Contrastive learning uses dropout as augmentation." in texts
        assert "Training improves out-of-domain retrieval." in texts

    def test_skips_paper_and_concept_nodes(self, agent):
        graph = EvidenceGraph(
            nodes=[
                _node("p1", NodeType.PAPER, "Some paper"),
                _node("con1", NodeType.CONCEPT, "contrastive learning"),
                _node("cl1", NodeType.CLAIM, "Real claim."),
            ],
            edges=[],
        )
        claims = agent._collect_claims(graph, [])
        assert len(claims) == 1
        assert claims[0]["text"] == "Real claim."

    def test_skips_empty_text_nodes(self, agent):
        graph = EvidenceGraph(
            nodes=[_node("cl1", NodeType.CLAIM, "   ")],
            edges=[],
        )
        assert agent._collect_claims(graph, []) == []

    def test_doc_metadata_attached(self, agent):
        graph, docs = _graph_with_claims()
        claims = agent._collect_claims(graph, docs)
        claim = next(c for c in claims if c["chunk_id"] == "ch1")
        assert claim["doc_id"] == "arxiv:2104.08821"
        assert claim["section_title"] == "Methods"

    def test_method_edge_sets_scicite_label(self, agent):
        graph, docs = _graph_with_claims()
        claims = agent._collect_claims(graph, docs)
        claim = next(c for c in claims if "dropout" in c["text"].lower())
        assert claim["scicite_label"] == "METHOD"
        assert claim["rel_score"] == pytest.approx(0.91)

    def test_conflict_flag_from_contradicts_edge(self, agent):
        graph = EvidenceGraph(
            nodes=[_node("cl1", NodeType.CLAIM, "X outperforms Y.", chunk_id="ch1")],
            edges=[_edge("cl1", "ch1", "CONTRADICTS", 0.5)],
        )
        claims = agent._collect_claims(graph, [])
        assert claims[0]["conflict"] is True

    def test_no_conflict_without_contradicts_edge(self, agent):
        graph, docs = _graph_with_claims()
        claims = agent._collect_claims(graph, docs)
        for c in claims:
            assert c["conflict"] is False

    def test_empty_graph_returns_empty(self, agent):
        assert agent._collect_claims(EvidenceGraph(), []) == []

    def test_none_graph_returns_empty(self, agent):
        assert agent._collect_claims(None, []) == []  # type: ignore[arg-type]



class TestParseSentencesJson:

    def test_parses_clean_json(self):
        payload = {"sentences": [{"text": "Hello.", "chunk_id": "c1", "doc_id": "p1",
                                   "section_title": None, "scicite_label": "METHOD",
                                   "rel_score": 0.9, "verdict": "Supported", "conflict": False}]}
        result = AnswerGeneratorAgent._parse_sentences_json(json.dumps(payload))
        assert len(result) == 1
        assert result[0]["text"] == "Hello."

    def test_strips_markdown_fence(self):
        payload = json.dumps({"sentences": [{"text": "Hi.", "chunk_id": "c1", "doc_id": "p1",
                                              "section_title": None, "scicite_label": "RESULT_CMP",
                                              "rel_score": 0.8, "verdict": "Supported", "conflict": False}]})
        fenced = f"```json\n{payload}\n```"
        result = AnswerGeneratorAgent._parse_sentences_json(fenced)
        assert result[0]["text"] == "Hi."

    def test_extracts_json_from_surrounding_text(self):
        inner = json.dumps({"sentences": [{"text": "Found.", "chunk_id": "c1", "doc_id": "p1",
                                            "section_title": None, "scicite_label": "BACKGROUND",
                                            "rel_score": 0.7, "verdict": "Supported", "conflict": False}]})
        wrapped = f"Here is the answer: {inner} That's it."
        result = AnswerGeneratorAgent._parse_sentences_json(wrapped)
        assert result[0]["text"] == "Found."

    def test_raises_on_no_json(self):
        with pytest.raises((ValueError, Exception)):
            AnswerGeneratorAgent._parse_sentences_json("no json here at all")

    def test_raises_if_sentences_not_list(self):
        bad = json.dumps({"sentences": "not a list"})
        with pytest.raises(ValueError):
            AnswerGeneratorAgent._parse_sentences_json(bad)

    def test_empty_sentences_list(self):
        result = AnswerGeneratorAgent._parse_sentences_json(json.dumps({"sentences": []}))
        assert result == []


class TestAssemble:

    def _sentence(self, text: str, chunk_id: str = "c1", doc_id: str = "p1",
                  section: str | None = "Results", score: float = 0.9) -> dict:
        return {"text": text, "chunk_id": chunk_id, "doc_id": doc_id,
                "section_title": section, "scicite_label": "METHOD",
                "rel_score": score, "verdict": "Supported", "conflict": False}

    def test_joins_sentence_texts(self):
        sentences = [self._sentence("First."), self._sentence("Second.", chunk_id="c2")]
        annotated, text = AnswerGeneratorAgent._assemble(sentences)
        assert text == "First. Second."
        assert len(annotated) == 2

    def test_per_sentence_citations(self):
        sentences = [self._sentence("A."), self._sentence("B.")]
        annotated, text = AnswerGeneratorAgent._assemble(sentences)
        assert len(annotated) == 2
        assert len(annotated[0].citations) == 1
        assert len(annotated[1].citations) == 1

    def test_unique_chunks_per_sentence(self):
        sentences = [self._sentence("A.", chunk_id="c1"), self._sentence("B.", chunk_id="c2")]
        annotated, text = AnswerGeneratorAgent._assemble(sentences)
        assert len(annotated) == 2
        assert annotated[0].citations[0].chunk_id == "c1"
        assert annotated[1].citations[0].chunk_id == "c2"

    def test_citation_fields(self):
        sentences = [self._sentence("Claim.", chunk_id="ch99", doc_id="arxiv:1234", section="Methods", score=0.77)]
        annotated, text = AnswerGeneratorAgent._assemble(sentences)
        c = annotated[0].citations[0]
        assert isinstance(c, Citation)
        assert c.doc_id == "arxiv:1234"
        assert c.chunk_id == "ch99"
        assert c.section_title == "Methods"
        assert c.rel_score == pytest.approx(0.77)

    def test_empty_sentences_returns_fallback(self):
        annotated, text = AnswerGeneratorAgent._assemble([])
        assert "Insufficient" in text
        assert annotated == []

    def test_skips_sentences_with_empty_text(self):
        sentences = [{"text": "  ", "chunk_id": "c1", "doc_id": "p1",
                      "section_title": None, "rel_score": 0.5}]
        annotated, text = AnswerGeneratorAgent._assemble(sentences)
        assert "Insufficient" in text


class TestGenerate:

    def _llm_sentence(self) -> dict:
        return {"text": "Contrastive learning uses dropout as augmentation.",
                "chunk_id": "ch1", "doc_id": "arxiv:2104.08821",
                "section_title": "Methods", "scicite_label": "METHOD",
                "rel_score": 0.91, "verdict": "Supported", "conflict": False}

    def test_returns_final_answer(self, agent, mock_llm):
        mock_llm.chat_text.return_value = _llm_response([self._llm_sentence()])
        graph, docs = _graph_with_claims()
        result = agent.generate("How does contrastive learning work?", [], graph, docs)
        assert isinstance(result, FinalAnswer)
        assert "dropout" in result.text.lower()

    def test_citations_attached(self, agent, mock_llm):
        mock_llm.chat_text.return_value = _llm_response([self._llm_sentence()])
        graph, docs = _graph_with_claims()
        result = agent.generate("query", [], graph, docs)
        assert len(result.sentences) >= 1
        assert result.sentences[0].citations[0].doc_id == "arxiv:2104.08821"

    def test_fallback_on_empty_graph(self, agent, mock_llm):
        result = agent.generate("query", [], EvidenceGraph(), [])
        assert "Insufficient" in result.text
        mock_llm.chat_text.assert_not_called()

    def test_fallback_on_llm_error(self, agent, mock_llm):
        mock_llm.chat_text.side_effect = RuntimeError("timeout")
        graph, docs = _graph_with_claims()
        result = agent.generate("query", [], graph, docs)
        assert "Insufficient" in result.text
        assert result.sentences == []

    def test_fallback_on_bad_json(self, agent, mock_llm):
        mock_llm.chat_text.return_value = "not json"
        graph, docs = _graph_with_claims()
        result = agent.generate("query", [], graph, docs)
        assert "Insufficient" in result.text

    def test_conflict_sentence_in_answer(self, agent, mock_llm):
        conflict_sentence = {
            "text": "However, paper X challenges this finding.",
            "chunk_id": "ch2", "doc_id": "arxiv:9999",
            "section_title": "Discussion", "scicite_label": "CONTRADICTS",
            "rel_score": 0.75, "verdict": "Contradicted", "conflict": True,
        }
        mock_llm.chat_text.return_value = _llm_response([self._llm_sentence(), conflict_sentence])
        graph, docs = _graph_with_claims()
        result = agent.generate("query", [], graph, docs)
        assert "However" in result.text

    def test_multiple_sentences_joined(self, agent, mock_llm):
        sentences = [
            {**self._llm_sentence(), "text": "First sentence."},
            {**self._llm_sentence(), "chunk_id": "ch2", "text": "Second sentence."},
        ]
        mock_llm.chat_text.return_value = _llm_response(sentences)
        graph, docs = _graph_with_claims()
        result = agent.generate("query", [], graph, docs)
        assert "First sentence." in result.text
        assert "Second sentence." in result.text

    def test_llm_called_with_correct_model(self, agent, mock_llm):
        mock_llm.chat_text.return_value = _llm_response([self._llm_sentence()])
        graph, docs = _graph_with_claims()
        agent.generate("query", [], graph, docs)
        call_kwargs = mock_llm.chat_text.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("model") == agent.config.model or \
               call_kwargs.args[0] == agent.config.model

    def test_none_evidence_graph_fallback(self, agent, mock_llm):
        result = agent.generate("query", [], None, [])  # type: ignore[arg-type]
        assert "Insufficient" in result.text
        mock_llm.chat_text.assert_not_called()
