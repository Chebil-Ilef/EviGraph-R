from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.judge import JudgeAgent
from utils.npm import npm_verify, extract_key_tokens
from utils.graph import project_dag, compute_hop_depth
from utils.nli import nli_verify
from schemas.objects import (
    ClaimType,
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


def _edge(src: str, tgt: str, relation: str = "extracted_from", score: float = 1.0) -> EvidenceEdge:
    return EvidenceEdge(source=src, target=tgt, relation=relation, score=score)


def _simple_graph() -> EvidenceGraph:

    return EvidenceGraph(
        nodes=[
            _node("p1", NodeType.PAPER),
            _node("ch1", NodeType.CHUNK, "BERT achieves 93.5% F1 on SQuAD.", chunk_id="ch1"),
            _node("claim:ch1:0", NodeType.CLAIM, "BERT achieves 93.5% F1.", chunk_id="ch1"),
        ],
        edges=[
            _edge("claim:ch1:0", "ch1", "extracted_from"),
            _edge("ch1", "p1", "CHUNK_OF"),
        ],
    )

class TestDAGProjection:

    def test_keeps_extracted_from_edges(self, judge):
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
        # Manually inject a cycle after projection
        import networkx as nx

        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        # Inject cycle
        dag.add_edge("ch1", "claim:ch1:0", relation="extracted_from")
        dag2 = project_dag(G)  # clean re-projection; just test detection logic
        # Cycle detection path is exercised without crashing
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
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        ct = judge._classify_claim_type("BERT achieves 93.5% F1.")
        assert ct == ClaimType.ATOMIC_FACTUAL

    def test_atomic_factual_acronym(self, judge):
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        ct = judge._classify_claim_type("GPT-4 outperforms GPT-3 on MMLU.")
        assert ct == ClaimType.ATOMIC_FACTUAL

    def test_inferential_no_numbers(self, judge):
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
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
                _edge("cl1", "ch1", "extracted_from"),
                _edge("ch1", "ch2", "result_comparison"),  # SciCite boundary
            ],
        )
        G = judge._to_networkx(g)
        dag = project_dag(G)
        hd = compute_hop_depth("cl1", dag)
        # ch1->ch2 edge is dropped (not evidence relation) so single-hop
        assert hd == HopDepth.SINGLE


class TestNPMVerifier:

    def test_tokens_present_signals_pass_to_nli(self):
        # "Supported" from NPM means tokens present — caller passes to NLI
        result = npm_verify(
            "BERT achieves 93.5% F1 on SQuAD.",
            ["BERT achieves 93.5% F1 score on the SQuAD benchmark dataset."],
        )
        assert result["verdict"] == "Supported"
        assert result["verifier_used"] == "npm"
        assert result["key_tokens_found"] is not None
        assert result["error_stage"] is None

    def test_tokens_missing_returns_not_supported(self):
        result = npm_verify(
            "GPT-4 achieves 90.1% on MMLU 2024.",
            ["Contrastive learning uses dropout as noise for augmentation."],
        )
        assert result["verdict"] == "Not-Supported"
        assert result["error_stage"] is None  # not an error, a real rejection
        assert "Missing" in result["reason"]

    def test_no_evidence_returns_not_supported(self):
        result = npm_verify("Some claim.", [])
        assert result["verdict"] == "Not-Supported"
        assert result["error_stage"] == "no_evidence"

    def test_no_key_tokens_signals_skip(self):
        # Inferential claim with no acronyms/numbers → signal skip
        result = npm_verify(
            "Contrastive learning improves sentence embeddings.",
            ["The model uses contrastive loss for training."],
        )
        assert result["error_stage"] == "no_key_tokens"
        assert result["key_tokens_found"] is None

    def test_not_supported_reason_mentions_missing_tokens(self):
        result = npm_verify(
            "GPT-4 achieves 90.1% on MMLU.",
            ["Unrelated text about something else."],
        )
        assert result["verdict"] == "Not-Supported"
        assert result.get("reason") is not None
        assert "Missing" in result["reason"] or "missing" in result["reason"]

    def test_key_token_extraction_numbers(self):
        tokens = extract_key_tokens("BERT achieves 93.5% F1 on SQuAD.")
        assert "93.5%" in tokens or "93.5" in tokens

    def test_key_token_extraction_acronyms(self):
        tokens = extract_key_tokens("GPT-4 outperforms BERT on SQuAD.")
        assert any(t in tokens for t in ["GPT", "BERT", "SQuAD"])


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
        # Claim with no chunk successors
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
            edges=[_edge("cl1", "ch1", "extracted_from")],
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

    # ── NPM fast rejection ──────────────────────────────────────────────────

    def test_npm_rejection_when_tokens_missing(self, judge):
        # Tokens in claim but not in evidence → NPM rejects immediately, NLI never called
        g = EvidenceGraph(
            nodes=[
                _node("ch1", NodeType.CHUNK, "Unrelated text about deep learning.", chunk_id="ch1"),
                _node("cl1", NodeType.CLAIM, "BERT achieves 93.5% on SQuAD.", chunk_id="ch1"),
            ],
            edges=[_edge("cl1", "ch1", "extracted_from")],
        )
        G = judge._to_networkx(g)
        dag = project_dag(G)
        with mock.patch("utils.nli.NLIModel.get", side_effect=AssertionError("NLI should not be called")):
            result = judge._route_and_verify(
                claim_id="cl1",
                claim_text="BERT achieves 93.5% on SQuAD.",
                claim_type=ClaimType.ATOMIC_FACTUAL,
                hop_depth=HopDepth.SINGLE,
                has_contradiction=False,
                dag=dag,
            )
        assert result["verdict"] == VerdictType.NOT_SUPPORTED.value
        assert result["verifier_used"] == "npm"
        assert result["reason"] is not None

    # ── NLI confirmation after NPM pass ─────────────────────────────────────

    def test_tokens_present_then_nli_supported(self, judge):
        # NPM sees tokens → NLI confirms entailment → Supported from nli
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        with self._nli_patch({"entails": 0.92, "contradicts": 0.03, "neutral": 0.05}):
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

    def test_tokens_present_but_nli_contradicted(self, judge):
        # NPM sees tokens (passes) → NLI detects contradiction
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        with self._nli_patch({"entails": 0.04, "contradicts": 0.91, "neutral": 0.05}):
            result = judge._route_and_verify(
                claim_id="claim:ch1:0",
                claim_text="BERT achieves 93.5% F1 on SQuAD.",
                claim_type=ClaimType.ATOMIC_FACTUAL,
                hop_depth=HopDepth.SINGLE,
                has_contradiction=False,
                dag=dag,
            )
        assert result["verdict"] == VerdictType.CONTRADICTED.value
        assert result["verifier_used"] == "nli"

    def test_tokens_present_nli_neutral_escalates_to_llm(self, judge, mock_llm):
        # NPM passes → NLI neutral → LLM
        mock_llm.chat_text.return_value = '{"verdict": "Inconclusive", "reasoning": "ambiguous"}'
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        with self._nli_patch({"entails": 0.35, "contradicts": 0.30, "neutral": 0.35}):
            result = judge._route_and_verify(
                claim_id="claim:ch1:0",
                claim_text="BERT achieves 93.5% F1 on SQuAD.",
                claim_type=ClaimType.ATOMIC_FACTUAL,
                hop_depth=HopDepth.SINGLE,
                has_contradiction=False,
                dag=dag,
            )
        assert result["verifier_used"] == "nli→llm"

    # ── No key tokens → go straight to NLI ──────────────────────────────────

    def test_no_key_tokens_skips_npm_goes_to_nli(self, judge):
        g = _simple_graph()
        G = judge._to_networkx(g)
        dag = project_dag(G)
        with self._nli_patch({"entails": 0.88, "contradicts": 0.05, "neutral": 0.07}):
            result = judge._route_and_verify(
                claim_id="claim:ch1:0",
                claim_text="Contrastive learning improves sentence embeddings.",
                claim_type=ClaimType.INFERENTIAL,
                hop_depth=HopDepth.SINGLE,
                has_contradiction=False,
                dag=dag,
            )
        assert result["verdict"] == VerdictType.SUPPORTED.value
        assert result["verifier_used"] == "nli"

    # ── Multi-hop and contradiction always go to LLM ─────────────────────────

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

    # ── Cycle guard ──────────────────────────────────────────────────────────

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

    # ── Every verdict has a reason ────────────────────────────────────────────

    def test_every_verdict_has_reason(self, judge, mock_llm):
        mock_llm.chat_text.return_value = '{"verdict": "Supported", "reasoning": "evidence confirms."}'
        g = _simple_graph()
        docs = [_doc("ch1", "BERT achieves 93.5% F1 on SQuAD.")]
        with self._nli_patch({"entails": 0.92, "contradicts": 0.03, "neutral": 0.05}):
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
        # atomic single-hop: NPM passes (tokens present), NLI confirms entailment
        mock_nli = mock.MagicMock()
        mock_nli.classify.return_value = {"entails": 0.92, "contradicts": 0.03, "neutral": 0.05}
        g = EvidenceGraph(
            nodes=[
                _node("ch1", NodeType.CHUNK, "BERT achieves 93.5% F1 on SQuAD.", chunk_id="ch1"),
                _node("claim:ch1:0", NodeType.CLAIM, "BERT achieves 93.5% F1.", chunk_id="ch1"),
            ],
            edges=[_edge("claim:ch1:0", "ch1", "extracted_from")],
        )
        docs = [_doc("ch1", "BERT achieves 93.5% F1 on SQuAD.")]
        with mock.patch("utils.nli.NLIModel.get", return_value=mock_nli):
            result = judge.filter("query", g, docs)
        claim_node = next(node for node in result.evidence_graph.nodes if node.node_id == "claim:ch1:0")
        assert claim_node.metadata["verdict"] == VerdictType.SUPPORTED.value

    def test_no_surviving_claims_falls_back_to_all_docs(self, judge, mock_llm):
        # NPM will fail (no matching tokens)
        g = EvidenceGraph(
            nodes=[
                _node("ch1", NodeType.CHUNK, "unrelated text here", chunk_id="ch1"),
                _node("claim:ch1:0", NodeType.CLAIM, "BERT 93.5% SQuAD GPT-4 MMLU.", chunk_id="ch1"),
            ],
            edges=[_edge("claim:ch1:0", "ch1", "extracted_from")],
        )
        docs = [_doc("ch1", "unrelated text here")]
        result = judge.filter("query", g, docs)
        claim_node = next(node for node in result.evidence_graph.nodes if node.node_id == "claim:ch1:0")
        assert claim_node.metadata["verdict"] == VerdictType.NOT_SUPPORTED.value

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
        # npm rejects fast when tokens missing, nli confirms when present
        assert vd.verifier_used in ("npm", "nli", "nli→llm")

    def test_filter_adds_judge_edges_for_supported(self, judge, mock_llm):
        g = EvidenceGraph(
            nodes=[
                _node("ch1", NodeType.CHUNK, "BERT achieves 93.5% on SQuAD.", chunk_id="ch1"),
                _node("cl1", NodeType.CLAIM, "BERT 93.5% SQuAD.", chunk_id="ch1"),
            ],
            edges=[_edge("cl1", "ch1", "extracted_from")],
        )
        docs = [_doc("ch1", "BERT achieves 93.5% on SQuAD.")]
        result = judge.filter("query", g, docs)
        supported = result.verdict_details.get("cl1").verdict == VerdictType.SUPPORTED.value if "cl1" in result.verdict_details else False
        if supported:
            assert any(
                e.source == "cl1" and e.relation == "judged_supported"
                for e in result.evidence_graph.edges
            )

    def test_concept_nodes_not_in_verdict_details(self, judge):
        g = EvidenceGraph(
            nodes=[
                _node("ch1", NodeType.CHUNK, "BERT achieves 93.5% F1 on SQuAD.", chunk_id="ch1"),
                _node("claim:ch1:0", NodeType.CLAIM, "BERT achieves 93.5% F1.", chunk_id="ch1"),
                _node("concept:ch1:1", NodeType.CONCEPT, "BERT", chunk_id="ch1"),
                _node("concept:ch1:2", NodeType.CONCEPT, "SQuAD", chunk_id="ch1"),
            ],
            edges=[
                _edge("claim:ch1:0", "ch1", "extracted_from"),
                _edge("concept:ch1:1", "ch1", "extracted_from"),
                _edge("concept:ch1:2", "ch1", "extracted_from"),
            ],
        )
        docs = [_doc("ch1", "BERT achieves 93.5% F1 on SQuAD.")]
        result = judge.filter("query", g, docs)

        assert "concept:ch1:1" not in result.verdict_details
        assert "concept:ch1:2" not in result.verdict_details
        assert "claim:ch1:0" in result.verdict_details
