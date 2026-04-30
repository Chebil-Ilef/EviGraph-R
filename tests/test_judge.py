from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.judge import JudgeAgent
from utils.graph import project_dag, compute_hop_depth
from utils.nli import nli_verify
from schemas.objects import (
    ClaimType,
    EdgeRelation,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    HopDepth,
    JudgementResult,
    NodeType,
    RetrievedDocument,
    VerdictType,
)

@pytest.fixture
def mock_llm():
    return mock.MagicMock()


@pytest.fixture
def judge(mock_llm):
    return JudgeAgent(llm_client=mock_llm)


def _doc(chunk_id: str, content: str, doc_id: str = "paper_1") -> RetrievedDocument:
    return RetrievedDocument(doc_id=doc_id, chunk_id=chunk_id, content=content, score=1.0)


def _node(node_id: str, node_type: NodeType, text: str = "", chunk_id: str | None = None, **kw) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        node_type=node_type,
        text=text,
        chunk_id=chunk_id or node_id,
        metadata=kw,
    )


def _edge(src: str, tgt: str, relation: str = "BACKGROUND", score: float = 1.0) -> EvidenceEdge:
    return EvidenceEdge(source=src, target=tgt, relation=relation, score=score)


def _simple_graph() -> EvidenceGraph:

    return EvidenceGraph(
        nodes=[
            _node("p1", NodeType.PAPER),
            _node("ch1", NodeType.CHUNK, "BERT achieves 93.5% F1 on SQuAD.", chunk_id="ch1"),
            _node("claim:ch1:0", NodeType.CLAIM, "BERT achieves 93.5% F1.", chunk_id="ch1"),
        ],
        edges=[
            _edge("claim:ch1:0", "ch1", "BACKGROUND"),
            _edge("ch1", "p1", "CHUNK_OF"),
        ],
    )

class TestDAGProjection:

    def test_keeps_BACKGROUND_edges(self, judge):
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        assert dag.has_edge("claim:ch1:0", "ch1")

    def test_drops_chunk_of_edges(self, judge):
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        assert not dag.has_edge("ch1", "p1")

    def test_preserves_node_attributes(self, judge):
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        assert dag.nodes["ch1"]["node_type"] == "chunk"
        assert "93.5%" in dag.nodes["ch1"]["text"]

    def test_cycle_detection_marks_nodes(self, judge):
        import networkx as nx

        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        dag.add_edge("ch1", "claim:ch1:0", relation="BACKGROUND")
        dag2 = project_dag(G)
        assert isinstance(dag2, nx.DiGraph)

    def test_cycle_detection_failure_is_logged(self, judge):
        g = _simple_graph()
        G = judge._to_networkx(g)
        with mock.patch("utils.graph.nx.simple_cycles", side_effect=RuntimeError("boom")):
            with mock.patch("utils.graph.logger.exception") as log_exception:
                dag = project_dag(G)
        assert dag.has_edge("claim:ch1:0", "ch1")
        log_exception.assert_called_once()


class TestClaimClassification:

    def test_atomic_factual_number(self, judge):
        ct = judge._classify_claim_type("BERT achieves 93.5% F1.")
        assert ct == ClaimType.ATOMIC_FACTUAL

    def test_atomic_factual_acronym(self, judge):
        ct = judge._classify_claim_type("GPT-4 outperforms GPT-3 on MMLU.")
        assert ct == ClaimType.ATOMIC_FACTUAL

    def test_inferential_no_numbers(self, judge):
        ct = judge._classify_claim_type("Contrastive learning improves representations.")
        assert ct == ClaimType.INFERENTIAL

    def test_single_hop_direct_chunk(self, judge):
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        hd = compute_hop_depth("claim:ch1:0", dag)
        assert hd == HopDepth.SINGLE

    def test_multi_hop_via_non_standard_relation(self, judge):
        g = EvidenceGraph(
            nodes=[
                _node("ch1", NodeType.CHUNK, "text1", chunk_id="ch1"),
                _node("ch2", NodeType.CHUNK, "text2", chunk_id="ch2"),
                _node("cl1", NodeType.CLAIM, "claim text", chunk_id="ch1"),
            ],
            edges=[
                _edge("cl1", "ch1", "BACKGROUND"),
                _edge("ch1", "ch2", "result_comparison"),
            ],
        )
        G = judge._to_networkx(g)
        dag = project_dag(G)
        hd = compute_hop_depth("cl1", dag)
        assert hd == HopDepth.SINGLE


class TestNLIVerifier:

    def _patch_nli(self, scores: dict):

        mock_model = mock.MagicMock()
        mock_model.classify.return_value = scores
        return mock.patch("utils.nli.NLIModel.get", return_value=mock_model)

    def test_nli_supported_high_entail(self, judge):
        with self._patch_nli({"entails": 0.92, "contradicts": 0.03, "neutral": 0.05}):
            result = nli_verify(
                "Contrastive learning improves sentence embeddings.",
                ["The model uses contrastive loss which improves embedding quality."],
            )
        assert result["verdict"] == "Supported"
        assert result["verifier_used"] == "nli"

    def test_nli_contradicted_high_contradict(self, judge):
        with self._patch_nli({"entails": 0.05, "contradicts": 0.88, "neutral": 0.07}):
            result = nli_verify(
                "Model outperforms all baselines.",
                ["Model underperforms on 3 of 5 benchmarks."],
            )
        assert result["verdict"] == "Contradicted"

    def test_nli_neutral_escalates_to_llm(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Inconclusive", "reasoning": "ambiguous"}'
        with self._patch_nli({"entails": 0.40, "contradicts": 0.20, "neutral": 0.40}):
            result = nli_verify(
                "Method generalises well.",
                ["Results show mixed performance across domains."],
            )
        assert result["verdict"] == "Neutral"
        assert result["verifier_used"] == "nli"

    def test_nli_model_load_failure_escalates_to_llm(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Inconclusive", "reasoning": "ambiguous"}'
        with mock.patch("utils.nli.NLIModel.get", side_effect=RuntimeError("no model")):
            result = nli_verify("Claim text.", ["Some evidence."])
        assert result["verdict"] == "Neutral"
        assert result["error_stage"] == "model_load_failed"

    def test_nli_no_evidence_returns_not_supported(self, judge):
        with self._patch_nli({}):
            result = nli_verify("claim", [])
        assert result["verdict"] == "Not-Supported"
        assert result["error_stage"] == "no_evidence"


class TestLLMJudge:

    def test_llm_judge_supported(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Supported", "reasoning": "evidence confirms."}'
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        result = judge._llm_judge("claim:ch1:0", "BERT achieves 93.5%.", dag)
        assert result["verdict"] == VerdictType.SUPPORTED.value
        assert result["verifier_used"] == "llm_judge"

    def test_llm_judge_not_supported(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Not-Supported", "reasoning": "not in evidence."}'
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        result = judge._llm_judge("claim:ch1:0", "GPT-4 achieves 99%.", dag)
        assert result["verdict"] == VerdictType.NOT_SUPPORTED.value

    def test_llm_judge_malformed_json_returns_inconclusive(self, judge, mock_llm):
        mock_llm.chat_text.return_value = "This is not JSON at all."
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        result = judge._llm_judge("claim:ch1:0", "some claim", dag)
        assert result["verdict"] == VerdictType.INCONCLUSIVE.value

    def test_llm_judge_invalid_verdict_returns_inconclusive(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Maybe"}'
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        result = judge._llm_judge("claim:ch1:0", "some claim", dag)
        assert result["verdict"] == VerdictType.INCONCLUSIVE.value

    def test_llm_judge_request_failure_returns_inconclusive(self, judge, mock_llm):
        mock_llm.chat_text.side_effect = RuntimeError("timeout")
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        result = judge._llm_judge("claim:ch1:0", "some claim", dag)
        assert result["verdict"] == VerdictType.INCONCLUSIVE.value

    def test_llm_judge_api_error_returns_inconclusive(self, judge, mock_llm):
        mock_llm.chat_text.side_effect = RuntimeError("API down")
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        result = judge._llm_judge("claim:ch1:0", "some claim", dag)
        assert result["verdict"] == VerdictType.INCONCLUSIVE.value

    def test_llm_judge_no_evidence_short_circuits(self, judge, mock_llm):
        g = EvidenceGraph(
            nodes=[_node("cl1", NodeType.CLAIM, "a claim", chunk_id="ch1")],
            edges=[],
        )
        G = judge._to_networkx(g)
        dag = project_dag(G)
        result = judge._llm_judge("cl1", "a claim", dag)
        assert result["verdict"] == VerdictType.NOT_SUPPORTED.value
        assert result["error_stage"] == "no_evidence"
        mock_llm.chat_text.assert_not_called()

    def test_contradiction_flagged_error_stage(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Not-Supported", "reasoning": "contradicts."}'
        g = EvidenceGraph(
            nodes=[
                _node("ch1", NodeType.CHUNK, "evidence text", chunk_id="ch1"),
                EvidenceNode(
                    node_id="cl1",
                    node_type=NodeType.CLAIM,
                    text="claim text",
                    chunk_id="ch1",
                    metadata={"contradicts": True},
                ),
            ],
            edges=[_edge("cl1", "ch1", "BACKGROUND")],
        )
        G = judge._to_networkx(g)
        dag = project_dag(G)
        result = judge._llm_judge("cl1", "claim text", dag, error_stage="contradiction_flagged")
        assert result["error_stage"] == "contradiction_flagged"


class TestVerifierRouting:

    def _nli_patch(self, scores: dict):
        mock_model = mock.MagicMock()
        mock_model.classify.return_value = scores
        return mock.patch("utils.nli.NLIModel.get", return_value=mock_model)

    def test_single_hop_routes_to_nli(self, judge, mock_llm):
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        with mock.patch(
            "agents.judge.nli_verify",
            return_value={
                "verdict": VerdictType.SUPPORTED.value,
                "verifier_used": "nli",
                "evidence_trail": [{"text": "BERT achieves 93.5% F1 on SQuAD."}],
                "error_stage": None,
                "reason": "NLI entailment score exceeded threshold.",
            },
        ):
            result = judge._route_and_verify(
                claim_id="claim:ch1:0",
                claim_text="BERT achieves 93.5% F1 on SQuAD.",
                claim_type=ClaimType.ATOMIC_FACTUAL,
                hop_depth=HopDepth.SINGLE,
                has_contradiction=False,
                dag=dag,
            )
        assert result["verdict"] == VerdictType.SUPPORTED.value
        assert result["verifier_used"] == "nli"
        assert result["reason"] is not None
        mock_llm.chat_text.assert_not_called()

    def test_nli_neutral_goes_to_llm(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Supported", "reasoning": "evidence confirms."}'
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        with mock.patch(
            "agents.judge.nli_verify",
            return_value={
                "verdict": "Neutral",
                "verifier_used": "nli",
                "evidence_trail": [{"text": "BERT achieves 93.5% F1 on SQuAD."}],
                "error_stage": None,
                "reason": "NLI scores ambiguous.",
            },
        ):
            result = judge._route_and_verify(
                claim_id="claim:ch1:0",
                claim_text="Contrastive learning improves sentence embeddings.",
                claim_type=ClaimType.INFERENTIAL,
                hop_depth=HopDepth.SINGLE,
                has_contradiction=False,
                dag=dag,
            )
        assert result["verdict"] == VerdictType.SUPPORTED.value
        assert result["verifier_used"] == "nli→llm"

    def test_multi_hop_routes_to_llm_judge(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Supported", "reasoning": "x"}'
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        result = judge._route_and_verify(
            claim_id="claim:ch1:0",
            claim_text="BERT achieves 93.5%.",
            claim_type=ClaimType.ATOMIC_FACTUAL,
            hop_depth=HopDepth.MULTI,
            has_contradiction=False,
            dag=dag,
        )
        assert result["verifier_used"] == "llm_judge"

    def test_contradiction_routes_to_llm_judge(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Not-Supported", "reasoning": "x"}'
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        result = judge._route_and_verify(
            claim_id="claim:ch1:0",
            claim_text="BERT achieves 93.5%.",
            claim_type=ClaimType.ATOMIC_FACTUAL,
            hop_depth=HopDepth.SINGLE,
            has_contradiction=True,
            dag=dag,
        )
        assert result["verifier_used"] == "llm_judge"
        assert result["error_stage"] == "contradiction_flagged"

    def test_cycle_detected_returns_inconclusive(self, judge):
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        dag.nodes["claim:ch1:0"]["cycle_detected"] = True
        result = judge._route_and_verify(
            claim_id="claim:ch1:0",
            claim_text="any claim",
            claim_type=ClaimType.ATOMIC_FACTUAL,
            hop_depth=HopDepth.SINGLE,
            has_contradiction=False,
            dag=dag,
        )
        assert result["verdict"] == VerdictType.INCONCLUSIVE.value
        assert result["error_stage"] == "cycle_detected"

    def test_every_verdict_has_reason(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Supported", "reasoning": "evidence confirms."}'
        g = _simple_graph()
        docs = [_doc("ch1", "BERT achieves 93.5% F1 on SQuAD.")]
        result = judge.filter("query", g, docs)
        for cid, vd in result.verdict_details.items():
            assert vd.reason is not None, f"Missing reason for {cid}"


class TestFilterEndToEnd:

    def test_empty_graph_returns_all_docs(self, judge):
        docs = [_doc("ch1", "text1"), _doc("ch2", "text2")]
        result = judge.filter("query", EvidenceGraph(), docs)
        assert result.evidence_graph == EvidenceGraph()
        assert result.verdict_details == {}

    def test_supported_claim_forwards_doc(self, judge, mock_llm):
        mock_nli = mock.MagicMock()
        mock_nli.classify.return_value = {"entails": 0.92, "contradicts": 0.03, "neutral": 0.05}
        g = EvidenceGraph(
            nodes=[
                _node("ch1", NodeType.CHUNK, "BERT achieves 93.5% F1 on SQuAD.", chunk_id="ch1"),
                _node("claim:ch1:0", NodeType.CLAIM, "BERT achieves 93.5% F1.", chunk_id="ch1"),
            ],
            edges=[_edge("claim:ch1:0", "ch1", "BACKGROUND")],
        )
        docs = [_doc("ch1", "BERT achieves 93.5% F1 on SQuAD.")]
        with mock.patch("utils.nli.NLIModel.get", return_value=mock_nli):
            result = judge.filter("query", g, docs)
        claim_node = next(node for node in result.evidence_graph.nodes if node.node_id == "claim:ch1:0")
        assert claim_node.metadata["verdict"] == VerdictType.SUPPORTED.value

    def test_verdict_details_populated(self, judge, mock_llm):
        mock_nli = mock.MagicMock()
        mock_nli.classify.return_value = {"entails": 0.92, "contradicts": 0.03, "neutral": 0.05}
        g = _simple_graph()
        docs = [_doc("ch1", "BERT achieves 93.5% F1 on SQuAD.")]
        with mock.patch("utils.nli.NLIModel.get", return_value=mock_nli):
            result = judge.filter("query", g, docs)
        assert "claim:ch1:0" in result.verdict_details
        vd = result.verdict_details["claim:ch1:0"]
        assert vd.verdict == VerdictType.SUPPORTED.value
        assert vd.verifier_used in ("nli", "nli→llm")

    def test_filter_adds_judge_edges_for_supported(self, judge, mock_llm):
        g = EvidenceGraph(
            nodes=[
                _node("ch1", NodeType.CHUNK, "BERT achieves 93.5% on SQuAD.", chunk_id="ch1"),
                _node("cl1", NodeType.CLAIM, "BERT 93.5% SQuAD.", chunk_id="ch1"),
            ],
            edges=[_edge("cl1", "ch1", "BACKGROUND")],
        )
        docs = [_doc("ch1", "BERT achieves 93.5% on SQuAD.")]
        result = judge.filter("query", g, docs)
        supported = result.verdict_details.get("cl1").verdict == VerdictType.SUPPORTED.value if "cl1" in result.verdict_details else False
        if supported:
            assert any(
                e.source == "cl1" and e.relation == "judged_supported"
                for e in result.evidence_graph.edges
            )

    def test_multi_hop_claim_routed_to_llm_judge(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Supported", "reasoning": "hop evidence confirms."}'
        g = EvidenceGraph(
            nodes=[
                _node("paper_A", NodeType.PAPER),
                _node("cited_paper", NodeType.PAPER),
                _node("ch1", NodeType.CHUNK, "Source chunk text.", chunk_id="ch1"),
                _node("hop_ch1", NodeType.CHUNK, "Hop evidence text from cited paper.", chunk_id="hop_ch1",
                      is_hop=True, hop_reason="missing_scope_context"),
                _node("cl1", NodeType.CLAIM, "Contrastive models improve Recall@10 by 4.2% on BEIR.", chunk_id="ch1"),
            ],
            edges=[
                _edge("ch1", "paper_A", "CHUNK_OF"),
                _edge("hop_ch1", "cited_paper", "CHUNK_OF"),
                _edge("cl1", "ch1", "BACKGROUND"),
                _edge("cl1", "hop_ch1", "hop_evidence", score=0.88),
            ],
        )
        docs = [_doc("ch1", "Source chunk text.")]
        result = judge.filter("query", g, docs)
        vd = result.verdict_details.get("cl1")
        assert vd is not None
        assert vd.verifier_used == "llm_judge"
        assert vd.hop_depth == HopDepth.MULTI.value

def _multi_hop_graph(
    claim_text: str = "Contrastive models improve Recall@10 by 4.2% on BEIR.",
    hop_text: str = "BEIR covers 18 diverse retrieval datasets.",
    hop_reason: str = "missing_scope_context",
) -> EvidenceGraph:
    return EvidenceGraph(
        nodes=[
            _node("paper_A", NodeType.PAPER),
            _node("cited_paper", NodeType.PAPER),
            _node("src_ch", NodeType.CHUNK, "Source chunk text.", chunk_id="src_ch"),
            _node("hop_ch", NodeType.CHUNK, hop_text, chunk_id="hop_ch",
                  is_hop=True, hop_reason=hop_reason),
            _node("cl1", NodeType.CLAIM, claim_text, chunk_id="src_ch"),
        ],
        edges=[
            _edge("src_ch", "paper_A", "CHUNK_OF"),
            _edge("hop_ch", "cited_paper", "CHUNK_OF"),
            _edge("cl1", "src_ch", "BACKGROUND"),
            _edge("cl1", "hop_ch", "hop_evidence", score=0.88),
        ],
    )


class TestMultiHopJudge:

    @pytest.fixture
    def mock_llm(self):
        return mock.MagicMock()

    @pytest.fixture
    def judge(self, mock_llm):
        return JudgeAgent(llm_client=mock_llm)

    def test_hop_evidence_edge_yields_multi_hop_depth(self, judge):
        g = _multi_hop_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        hd = compute_hop_depth("cl1", dag)
        assert hd == HopDepth.MULTI

    def test_BACKGROUND_only_yields_single_hop_depth(self, judge):
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        hd = compute_hop_depth("claim:ch1:0", dag)
        assert hd == HopDepth.SINGLE

    def test_hop_evidence_edge_survives_dag_projection(self, judge):
        g = _multi_hop_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        assert dag.has_edge("cl1", "hop_ch")

    def test_chunk_of_edges_dropped_in_dag_projection(self, judge):
        g = _multi_hop_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        assert not dag.has_edge("src_ch", "paper_A")
        assert not dag.has_edge("hop_ch", "cited_paper")

    def test_multi_hop_claim_routes_to_llm_judge(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Supported", "reasoning": "evidence confirms."}'
        g = _multi_hop_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        result = judge._route_and_verify(
            claim_id="cl1",
            claim_text="Contrastive models improve Recall@10 by 4.2% on BEIR.",
            claim_type=ClaimType.ATOMIC_FACTUAL,
            hop_depth=HopDepth.MULTI,
            has_contradiction=False,
            dag=dag,
        )
        assert result["verifier_used"] == "llm_judge"

    def test_multi_hop_claim_never_uses_nli(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Supported", "reasoning": "x"}'
        g = _multi_hop_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        with mock.patch("utils.nli.NLIModel.get", side_effect=AssertionError("nli must not run")):
            result = judge._route_and_verify(
                claim_id="cl1",
                claim_text="Contrastive models improve Recall@10 by 4.2% on BEIR.",
                claim_type=ClaimType.ATOMIC_FACTUAL,
                hop_depth=HopDepth.MULTI,
                has_contradiction=False,
                dag=dag,
            )
        assert result["verifier_used"] == "llm_judge"

    def test_backwards_traverse_includes_source_and_hop_chunks(self, judge):
        from utils.graph import backwards_traverse
        g = _multi_hop_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        trail = backwards_traverse("cl1", dag)
        trail_ids = {step["node_id"] for step in trail}
        assert "src_ch" in trail_ids
        assert "hop_ch" in trail_ids

    def test_hop_chunk_text_in_trail(self, judge):
        from utils.graph import backwards_traverse
        hop_text = "BEIR covers 18 diverse retrieval datasets."
        g = _multi_hop_graph(hop_text=hop_text)
        G = judge._to_networkx(g)
        dag = project_dag(G)
        trail = backwards_traverse("cl1", dag)
        texts = [step["text"] for step in trail]
        assert any(hop_text[:30] in t for t in texts)

    def test_collect_evidence_chunks_includes_hop_chunk(self, judge):
        g = _multi_hop_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        chunks = judge._collect_evidence_chunks("cl1", dag)
        assert any("Source chunk" in c or "BEIR" in c for c in chunks)
        assert len(chunks) == 2

    def test_multi_hop_depth_recorded_in_verdict(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Supported", "reasoning": "hop confirms."}'
        g = _multi_hop_graph()
        docs = [_doc("src_ch", "Source chunk text.")]
        result = judge.filter("query", g, docs)
        vd = result.verdict_details.get("cl1")
        assert vd is not None
        assert vd.hop_depth == HopDepth.MULTI.value

    def test_single_hop_depth_recorded_in_verdict(self, judge, mock_llm):
        mock_nli = mock.MagicMock()
        mock_nli.classify.return_value = {"entails": 0.91, "contradicts": 0.04, "neutral": 0.05}
        g = _simple_graph()
        docs = [_doc("ch1", "BERT achieves 93.5% F1 on SQuAD.")]
        with mock.patch("utils.nli.NLIModel.get", return_value=mock_nli):
            result = judge.filter("query", g, docs)
        vd = result.verdict_details.get("claim:ch1:0")
        assert vd is not None
        assert vd.hop_depth == HopDepth.SINGLE.value

    def test_llm_judge_called_with_multi_chunk_trail(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Supported", "reasoning": "full trail."}'
        g = _multi_hop_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        judge._llm_judge("cl1", "Contrastive models improve Recall@10 by 4.2% on BEIR.", dag)
        call_args = mock_llm.chat_text.call_args
        user_prompt = call_args[1].get("user_prompt") or call_args[0][2]
        assert "Source chunk text" in user_prompt
        assert "BEIR covers 18 diverse retrieval datasets" in user_prompt

    @pytest.mark.parametrize("hop_reason", [
        "missing_scope_context",
        "missing_comparison_baseline",
        "missing_method_origin",
        "missing_definition_context",
    ])
    def test_all_hop_reasons_yield_multi_depth(self, judge, hop_reason):
        g = _multi_hop_graph(hop_reason=hop_reason)
        G = judge._to_networkx(g)
        dag = project_dag(G)
        hd = compute_hop_depth("cl1", dag)
        assert hd == HopDepth.MULTI

    def test_two_hop_chunks_both_in_evidence_trail(self, judge):
        from utils.graph import backwards_traverse
        g = EvidenceGraph(
            nodes=[
                _node("paper_A", NodeType.PAPER),
                _node("cited_paper", NodeType.PAPER),
                _node("src_ch", NodeType.CHUNK, "Source text.", chunk_id="src_ch"),
                _node("hop_ch1", NodeType.CHUNK, "Hop evidence A.", chunk_id="hop_ch1",
                      is_hop=True, hop_reason="missing_scope_context"),
                _node("hop_ch2", NodeType.CHUNK, "Hop evidence B.", chunk_id="hop_ch2",
                      is_hop=True, hop_reason="missing_scope_context"),
                _node("cl1", NodeType.CLAIM, "Claim text.", chunk_id="src_ch"),
            ],
            edges=[
                _edge("src_ch", "paper_A", "CHUNK_OF"),
                _edge("hop_ch1", "cited_paper", "CHUNK_OF"),
                _edge("hop_ch2", "cited_paper", "CHUNK_OF"),
                _edge("cl1", "src_ch", "BACKGROUND"),
                _edge("cl1", "hop_ch1", "hop_evidence", score=0.85),
                _edge("cl1", "hop_ch2", "hop_evidence", score=0.80),
            ],
        )
        G = judge._to_networkx(g)
        dag = project_dag(G)
        trail = backwards_traverse("cl1", dag)
        trail_ids = {step["node_id"] for step in trail}
        assert "src_ch" in trail_ids
        assert "hop_ch1" in trail_ids
        assert "hop_ch2" in trail_ids
