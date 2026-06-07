from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evigraph.agents.answer_generator import AnswerGeneratorAgent
from evigraph.schemas.objects import (
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


def _edge(src: str, tgt: str, relation: str = "BACKGROUND", score: float = 0.85) -> EvidenceEdge:
    return EvidenceEdge(source=src, target=tgt, relation=relation, score=score)


def _graph_with_claims() -> tuple[EvidenceGraph, list[RetrievedDocument]]:

    nodes = [
        _node("p1", NodeType.PAPER),
        _node("ch1", NodeType.CHUNK, "Dropout noise used as augmentation.", chunk_id="ch1"),
        _node("claim:ch1:0", NodeType.CLAIM, "Contrastive learning uses dropout as augmentation.", chunk_id="ch1", verdict="Supported"),
        _node("claim:ch1:1", NodeType.CLAIM, "Training improves out-of-domain retrieval.", chunk_id="ch1", verdict="Supported"),
    ]
    edges = [
        _edge("claim:ch1:0", "ch1", "METHOD", 0.91),
        _edge("claim:ch1:1", "ch1", "BACKGROUND", 0.80),
        _edge("ch1", "p1", "CHUNK_OF"),
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

    def test_skips_non_claim_nodes(self, agent):
        graph = EvidenceGraph(
            nodes=[
                _node("p1", NodeType.PAPER, "Some paper"),
                _node("cl1", NodeType.CLAIM, "Real claim.", verdict="Supported"),
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

    def test_reads_scicite_label_from_edge_relation(self, agent):
        graph = EvidenceGraph(
            nodes=[
                _node("p1", NodeType.PAPER),
                _node("ch1", NodeType.CHUNK, "Dropout noise used as augmentation.", chunk_id="ch1"),
                _node("claim:ch1:0", NodeType.CLAIM, "Contrastive learning uses dropout as augmentation.", chunk_id="ch1", verdict="Supported"),
            ],
            edges=[
                EvidenceEdge(
                    source="claim:ch1:0",
                    target="ch1",
                    relation="METHOD",
                    score=0.91,
                    metadata={},
                ),
                _edge("ch1", "p1", "CHUNK_OF"),
            ],
        )
        docs = [_doc("ch1", "Dropout noise used as augmentation.", doc_id="arxiv:2104.08821")]

        claims = agent._collect_claims(graph, docs)

        assert claims[0]["scicite_label"] == "METHOD"
        assert claims[0]["rel_score"] == pytest.approx(0.91)

    def test_conflict_flag_from_contradicts_edge(self, agent):
        graph = EvidenceGraph(
            nodes=[_node("cl1", NodeType.CLAIM, "X outperforms Y.", chunk_id="ch1", verdict="Supported")],
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
        assert "First" in text and "Second" in text
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

    def test_claim_refs_attach_citations_from_claims(self):
        claims = [
            {
                "text": "Claim one.",
                "chunk_id": "ch1",
                "doc_id": "arxiv:1",
                "section_title": "Methods",
                "scicite_label": "METHOD",
                "rel_score": 0.91,
                "verdict": "Supported",
                "conflict": False,
            },
            {
                "text": "Claim two.",
                "chunk_id": "ch2",
                "doc_id": "arxiv:2",
                "section_title": "Results",
                "scicite_label": "RESULT_COMPARISON",
                "rel_score": 0.88,
                "verdict": "Supported",
                "conflict": True,
            },
        ]
        sentences = [{"text": "Combined answer sentence.", "claim_refs": [1, 2]}]
        annotated, text = AnswerGeneratorAgent._assemble(sentences, claims)
        assert "Combined answer sentence" in text
        assert len(annotated) == 1
        assert len(annotated[0].citations) == 2
        assert annotated[0].citations[0].doc_id == "arxiv:1"
        assert annotated[0].citations[1].doc_id == "arxiv:2"
        assert annotated[0].conflict_flag is True


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
        assert "insufficient verified evidence" in result.text.lower()
        mock_llm.chat_text.assert_not_called()

    def test_fallback_on_llm_error(self, agent, mock_llm):
        mock_llm.chat_text.side_effect = RuntimeError("timeout")
        graph, docs = _graph_with_claims()
        result = agent.generate("query", [], graph, docs)
        assert "insufficient verified evidence" in result.text.lower()
        assert result.sentences == []

    def test_fallback_on_bad_json(self, agent, mock_llm):
        mock_llm.chat_text.return_value = "not json"
        graph, docs = _graph_with_claims()
        result = agent.generate("query", [], graph, docs)
        assert "insufficient verified evidence" in result.text.lower()

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
        assert "First sentence" in result.text
        assert "Second sentence" in result.text

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
        assert "insufficient verified evidence" in result.text.lower()
        mock_llm.chat_text.assert_not_called()

    def test_generate_accepts_claim_refs_output(self, agent, mock_llm):
        mock_llm.chat_text.return_value = _llm_response([
            {"text": "Contrastive learning uses dropout as augmentation.", "claim_refs": [1]}
        ])
        graph, docs = _graph_with_claims()
        result = agent.generate("How does contrastive learning work?", [], graph, docs)
        assert isinstance(result, FinalAnswer)
        assert len(result.sentences) == 1
        assert result.sentences[0].citations[0].doc_id == "arxiv:2104.08821"


class TestLatexHandling:

    def _llm_sentence(self) -> dict:
        return {"text": "Contrastive learning uses dropout as augmentation.",
                "chunk_id": "ch1", "doc_id": "arxiv:2104.08821",
                "section_title": "Methods", "scicite_label": "METHOD",
                "rel_score": 0.91, "verdict": "Supported", "conflict": False}

    def test_parses_simple_latex_math_in_claim(self, agent):

        latex_text = r"The loss function is \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} loss_i"
        response = json.dumps({
            "sentences": [{
                **self._llm_sentence(),
                "text": latex_text
            }]
        })
        # Escape backslashes for JSON validity
        response = response.replace("\\", "\\\\")
        
        result = AnswerGeneratorAgent._parse_sentences_json(response)
        assert len(result) == 1
        # Result should preserve the LaTeX (unescaped)
        assert r"\mathcal{L}" in result[0]["text"]
        assert r"\frac{1}{N}" in result[0]["text"]

    def test_parses_complex_latex_equations(self, agent):

        latex_text = (
            r"The top quark production process is modified by "
            r"$\mathcal{L}_{cdm} = -ig_s\frac{\tilde{d}}{2}\bar{t}\sigma_{\mu\nu}\gamma_5 G^{\mu\nu}t$"
        )
        response = json.dumps({
            "sentences": [{
                **self._llm_sentence(),
                "text": latex_text
            }]
        })
        response = response.replace("\\", "\\\\")
        
        result = AnswerGeneratorAgent._parse_sentences_json(response)
        assert len(result) == 1
        text = result[0]["text"]
        # All LaTeX commands should be preserved
        assert r"\mathcal{L}_{cdm}" in text
        assert r"\frac{\tilde{d}}{2}" in text
        assert r"\bar{t}" in text
        assert r"\sigma_{\mu\nu}" in text
        assert r"\gamma_5" in text

    def test_parses_decay_vertex_equation(self, agent):

        latex_text = (
            r"The decay vertex for t\rightarrow b W^+ is "
            r"\Gamma^\mu_{Wtb} = -\frac{g}{\sqrt{2}} V_{tb}^\star "
            r"\bar{u}(p_b) [\gamma_\mu (f_1^L P_L + f_1^R P_R) - "
            r"i\sigma^{\mu\nu}(p_t - p_b)_\nu (f_2^L P_L + f_2^R P_R)] u(p_t)"
        )
        response = json.dumps({
            "sentences": [{
                **self._llm_sentence(),
                "text": latex_text
            }]
        })
        response = response.replace("\\", "\\\\")
        
        result = AnswerGeneratorAgent._parse_sentences_json(response)
        assert len(result) == 1
        text = result[0]["text"]
        assert r"\rightarrow" in text
        assert r"\Gamma^\mu_{Wtb}" in text
        assert r"\frac{g}{\sqrt{2}}" in text
        assert r"\bar{u}" in text
        assert r"\sigma^{\mu\nu}" in text

    def test_multiple_sentences_with_latex(self, agent):

        sentences_data = [
            {
                **self._llm_sentence(),
                "text": r"CP violation is parametrized by anomalous couplings $\mathcal{L}_{cdm}$.",
                "chunk_id": "ch1"
            },
            {
                **self._llm_sentence(),
                "text": r"The coupling $f_2^R = f\exp(i(\phi_f + \delta_f))$ has both phases.",
                "chunk_id": "ch2"
            },
            {
                **self._llm_sentence(),
                "text": r"We study correlations like $\bar{t}t \rightarrow b\mu^+\nu_\mu$.",
                "chunk_id": "ch3"
            }
        ]
        response = json.dumps({"sentences": sentences_data})
        response = response.replace("\\", "\\\\")
        
        result = AnswerGeneratorAgent._parse_sentences_json(response)
        assert len(result) == 3
        
        # Verify each sentence preserved its LaTeX
        assert r"\mathcal{L}_{cdm}" in result[0]["text"]
        assert r"f_2^R = f\exp" in result[1]["text"]
        assert r"\rightarrow" in result[2]["text"]

    def test_assembles_latex_sentences_correctly(self, agent):

        sentences = [
            {
                "text": r"The interaction $\mathcal{L}_{cdm} = -ig_s\frac{\tilde{d}}{2}\bar{t}\sigma_{\mu\nu}\gamma_5 G^{\mu\nu}t$.",
                "chunk_id": "ch1", "doc_id": "arxiv:0905.1074",
                "section_title": "Introduction", "scicite_label": "METHOD",
                "rel_score": 0.95, "verdict": "Supported", "conflict": False
            },
            {
                "text": r"Decay: $t\rightarrow bW^+$ with $\Gamma^\mu_{Wtb} = -\frac{g}{\sqrt{2}}V_{tb}^\star$.",
                "chunk_id": "ch2", "doc_id": "arxiv:0905.1074",
                "section_title": "Introduction", "scicite_label": "METHOD",
                "rel_score": 0.92, "verdict": "Supported", "conflict": False
            }
        ]
        
        annotated, text = AnswerGeneratorAgent._assemble(sentences)
        
        # Text should be joined
        assert len(annotated) == 2
        assert r"\mathcal{L}_{cdm}" in text
        assert r"\rightarrow" in text
        assert r"\frac{g}{\sqrt{2}}" in text
        
        # Citations should be attached
        assert annotated[0].citations[0].doc_id == "arxiv:0905.1074"
        assert annotated[1].citations[0].doc_id == "arxiv:0905.1074"

    def test_generate_with_latex_claims(self, agent, mock_llm):

        graph = EvidenceGraph(
            nodes=[
                _node("p1", NodeType.PAPER),
                _node("ch1", NodeType.CHUNK, 
                       r"$\mathcal{L}_{cdm} = -ig_s\frac{\tilde{d}}{2}$", 
                       chunk_id="ch1"),
                _node("claim:ch1:0", NodeType.CLAIM, 
                       r"The interaction $\mathcal{L}_{cdm}$ modifies the vertex.",
                       chunk_id="ch1", verdict="Supported"),
            ],
            edges=[
                _edge("claim:ch1:0", "ch1", "METHOD", 0.95),
                _edge("ch1", "p1", "CHUNK_OF"),
            ]
        )
        docs = [_doc("ch1", r"$\mathcal{L}_{cdm} = ...$", doc_id="arxiv:0905.1074")]
        
        llm_response = json.dumps({
            "sentences": [{
                "text": r"The Lagrangian $\mathcal{L}_{cdm} = -ig_s\frac{\tilde{d}}{2}\bar{t}\sigma_{\mu\nu}\gamma_5$ modifies production.",
                "chunk_id": "ch1", "doc_id": "arxiv:0905.1074",
                "section_title": "Introduction", "scicite_label": "METHOD",
                "rel_score": 0.95, "verdict": "Supported", "conflict": False
            }]
        })
        # Escape for JSON validity
        llm_response = llm_response.replace("\\", "\\\\")
        mock_llm.chat_text.return_value = llm_response
        
        result = agent.generate("How is top production modified?", [], graph, docs)
        
        assert isinstance(result, FinalAnswer)
        assert r"\mathcal{L}_{cdm}" in result.text
        assert r"\frac{\tilde{d}}{2}" in result.text
        assert len(result.sentences) > 0

    def test_handles_escaped_newlines_in_latex(self, agent):

        latex_text = r"Matrix form: $\begin{pmatrix} a \\ b \end{pmatrix}$"
        response = json.dumps({
            "sentences": [{
                **self._llm_sentence(),
                "text": latex_text
            }]
        })
        response = response.replace("\\", "\\\\")
        
        result = AnswerGeneratorAgent._parse_sentences_json(response)
        assert len(result) == 1
        # Should preserve double backslash for LaTeX line breaks
        assert r"\\" in result[0]["text"] or r"begin{pmatrix}" in result[0]["text"]

    def test_mixed_content_with_latex_and_text(self, agent):

        mixed_text = (
            "The top quark mass affects the Higgs potential stability. "
            "Recent measurements give $m_t = 173.24 \\pm 0.76$ GeV. "
            "The production cross section scales as $\\sigma \\propto \\alpha_s^2$. "
            "Three decay modes dominate: $t \\to bW^+$ with branching ratio $\\sim 100\\%$."
        )
        response = json.dumps({
            "sentences": [{
                **self._llm_sentence(),
                "text": mixed_text
            }]
        })
        response = response.replace("\\", "\\\\")
        
        result = AnswerGeneratorAgent._parse_sentences_json(response)
        assert len(result) == 1
        text = result[0]["text"]
        
        # All LaTeX should be preserved
        assert r"m_t = 173.24" in text
        assert r"\alpha_s^2" in text
        assert r"t \to bW^+" in text

    def test_error_handling_with_invalid_latex_json(self, agent):

        # Intentionally broken JSON (unmatched quotes)
        broken_json = r'{"sentences": [{"text": "Unclosed quote: \frac{x}{y}"]}'
        
        with pytest.raises((ValueError, json.JSONDecodeError)):
            AnswerGeneratorAgent._parse_sentences_json(broken_json)
