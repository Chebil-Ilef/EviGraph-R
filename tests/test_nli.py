from __future__ import annotations

from unittest import mock

from utils.nli import NLIModel


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
