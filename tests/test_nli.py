from __future__ import annotations

from unittest import mock

import pytest

from utils.nli import NLIModel, _aggregate_chunk_scores, nli_verify, nli_verify_batch


class TestNLIModelLabelNormalization:

    def test_normalizes_standard_nli_labels(self):
        model = object.__new__(NLIModel)
        model._id2label = {}

        scores = model._normalize_raw_scores([
            {"label": "entailment", "score": 0.91},
            {"label": "neutral", "score": 0.07},
            {"label": "contradiction", "score": 0.02},
        ])

        assert scores == {
            "entails": 0.91,
            "neutral": 0.07,
            "contradicts": 0.02,
        }

    def test_normalizes_label_id_outputs_via_id2label(self):
        model = object.__new__(NLIModel)
        model._id2label = {
            0: "contradiction",
            1: "neutral",
            2: "entailment",
        }

        scores = model._normalize_raw_scores([
            {"label": "LABEL_0", "score": 0.15},
            {"label": "LABEL_1", "score": 0.05},
            {"label": "LABEL_2", "score": 0.80},
        ])

        assert scores == {
            "entails": 0.80,
            "neutral": 0.05,
            "contradicts": 0.15,
        }

    def test_warns_on_unknown_labels(self):
        model = object.__new__(NLIModel)
        model._id2label = {}

        with mock.patch("utils.nli.logger.warning") as warn:
            scores = model._normalize_raw_scores([
                {"label": "SUPPORTED", "score": 0.88},
                {"label": "MAYBE", "score": 0.12},
            ])

        assert scores["entails"] == 0.88
        assert scores["neutral"] == 0.0
        assert scores["contradicts"] == 0.0
        warn.assert_called_once()


class TestAggregateChunkScores:

    def test_single_chunk_returns_exact_score(self):
        chunk_scores = [{"entails": 0.80, "contradicts": 0.10, "neutral": 0.10}]
        agg = _aggregate_chunk_scores(chunk_scores)
        # 0.6*0.80 + 0.4*0.80 = 0.80
        assert abs(agg["entails"] - 0.80) < 1e-6

    def test_one_high_chunk_pulled_down_by_low_chunks(self):
        # Old pure-max would return 0.80; new formula should be lower
        chunk_scores = [
            {"entails": 0.80, "contradicts": 0.10, "neutral": 0.10},
            {"entails": 0.10, "contradicts": 0.05, "neutral": 0.85},
            {"entails": 0.15, "contradicts": 0.05, "neutral": 0.80},
        ]
        agg = _aggregate_chunk_scores(chunk_scores)
        mean_entails = (0.80 + 0.10 + 0.15) / 3
        expected = 0.6 * 0.80 + 0.4 * mean_entails
        assert abs(agg["entails"] - expected) < 1e-6
        # Must be strictly less than pure max (0.80)
        assert agg["entails"] < 0.80

    def test_all_chunks_high_entailment_stays_high(self):
        chunk_scores = [
            {"entails": 0.88, "contradicts": 0.05, "neutral": 0.07},
            {"entails": 0.85, "contradicts": 0.08, "neutral": 0.07},
            {"entails": 0.90, "contradicts": 0.04, "neutral": 0.06},
        ]
        agg = _aggregate_chunk_scores(chunk_scores)
        assert agg["entails"] > 0.65  # should still pass the new threshold

    def test_aggregation_threshold_boundary(self):
        # One chunk hits 0.70 entailment, two others are very low
        # Should NOT reach the 0.65 threshold after blending
        chunk_scores = [
            {"entails": 0.70, "contradicts": 0.10, "neutral": 0.20},
            {"entails": 0.05, "contradicts": 0.10, "neutral": 0.85},
            {"entails": 0.05, "contradicts": 0.10, "neutral": 0.85},
        ]
        agg = _aggregate_chunk_scores(chunk_scores)
        mean_e = (0.70 + 0.05 + 0.05) / 3  # ~0.267
        expected = 0.6 * 0.70 + 0.4 * mean_e  # ~0.527
        assert abs(agg["entails"] - expected) < 1e-6
        assert agg["entails"] < 0.65  # would have been "Supported" under pure-max

    def test_contradiction_aggregation(self):
        chunk_scores = [
            {"entails": 0.05, "contradicts": 0.80, "neutral": 0.15},
            {"entails": 0.10, "contradicts": 0.20, "neutral": 0.70},
        ]
        agg = _aggregate_chunk_scores(chunk_scores)
        mean_c = (0.80 + 0.20) / 2
        expected = 0.6 * 0.80 + 0.4 * mean_c
        assert abs(agg["contradicts"] - expected) < 1e-6


class TestNliVerifyWithMockedModel:

    def _make_mock_nli(self, per_chunk_scores: list[dict[str, float]]):
        """Return a mock NLIModel.get() that yields per_chunk_scores in order."""
        mock_model = mock.MagicMock()
        mock_model.classify.side_effect = per_chunk_scores
        return mock_model

    def test_supported_when_all_chunks_agree(self):
        scores = [
            {"entails": 0.90, "contradicts": 0.05, "neutral": 0.05},
            {"entails": 0.85, "contradicts": 0.08, "neutral": 0.07},
        ]
        with mock.patch("utils.nli.NLIModel.get", return_value=self._make_mock_nli(scores)):
            result = nli_verify("Transformers outperform RNNs on NLP benchmarks.", ["chunk1", "chunk2"])
        assert result["verdict"] == "Supported"

    def test_neutral_when_only_one_weak_chunk(self):
        # Max=0.70, mean=(0.70+0.05)/2=0.375 → agg=0.6*0.70+0.4*0.375=0.57 < 0.65
        scores = [
            {"entails": 0.70, "contradicts": 0.10, "neutral": 0.20},
            {"entails": 0.05, "contradicts": 0.10, "neutral": 0.85},
        ]
        with mock.patch("utils.nli.NLIModel.get", return_value=self._make_mock_nli(scores)), \
             mock.patch("utils.nli.GRAPH_CONFIG") as cfg:
            cfg.nli_threshold = 0.65
            cfg.nli_contradiction_threshold = 0.70
            result = nli_verify("Some claim.", ["chunk1", "chunk2"])
        assert result["verdict"] == "Neutral"

    def test_contradicted_when_contradiction_score_high(self):
        scores = [
            {"entails": 0.05, "contradicts": 0.82, "neutral": 0.13},
            {"entails": 0.08, "contradicts": 0.75, "neutral": 0.17},
        ]
        with mock.patch("utils.nli.NLIModel.get", return_value=self._make_mock_nli(scores)):
            result = nli_verify("Batch norm hurts performance.", ["chunk1", "chunk2"])
        assert result["verdict"] == "Contradicted"

    def test_not_supported_on_empty_chunks(self):
        result = nli_verify("Some claim.", [])
        assert result["verdict"] == "Not-Supported"
        assert result["error_stage"] == "no_evidence"

    def test_trail_has_correct_number_of_entries(self):
        scores = [
            {"entails": 0.88, "contradicts": 0.05, "neutral": 0.07},
            {"entails": 0.86, "contradicts": 0.06, "neutral": 0.08},
            {"entails": 0.84, "contradicts": 0.07, "neutral": 0.09},
        ]
        with mock.patch("utils.nli.NLIModel.get", return_value=self._make_mock_nli(scores)):
            result = nli_verify("claim", ["c1", "c2", "c3"])
        assert len(result["evidence_trail"]) == 3


class TestNliVerifyBatchWithMockedModel:

    def test_batch_matches_sequential_on_clear_support(self):
        all_scores = [
            {"entails": 0.88, "contradicts": 0.05, "neutral": 0.07},
            {"entails": 0.85, "contradicts": 0.06, "neutral": 0.09},
        ]
        mock_model = mock.MagicMock()
        mock_model.classify_batch.return_value = all_scores

        with mock.patch("utils.nli.NLIModel.get", return_value=mock_model):
            results = nli_verify_batch([("claim", ["chunk1", "chunk2"])])

        assert results[0]["verdict"] == "Supported"

    def test_batch_not_supported_for_empty_evidence(self):
        mock_model = mock.MagicMock()
        with mock.patch("utils.nli.NLIModel.get", return_value=mock_model):
            results = nli_verify_batch([("claim", [])])
        assert results[0]["verdict"] == "Not-Supported"

    def test_empty_input_returns_empty(self):
        results = nli_verify_batch([])
        assert results == []
