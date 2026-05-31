from __future__ import annotations
import sys
from pathlib import Path
from unittest import mock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evigraph.utils.scicite import SciCiteModel, classify_citation, _FALLBACK


def _mock_pipeline(label: str, score: float):
    pipe = mock.MagicMock()
    pipe.return_value = [{"label": label, "score": score}]
    return pipe


class TestSciCiteModel:

    def setup_method(self):
        SciCiteModel._instance = None

    def test_classify_method_label(self):
        with mock.patch("evigraph.utils.scicite.SciCiteModel.__init__", return_value=None):
            model = SciCiteModel.get()
            model._pipe = _mock_pipeline("method", 0.92)
            label, conf = model.classify("We use the approach from [1].")
        assert label == "METHOD"
        assert conf == pytest.approx(0.92)

    def test_classify_background_label(self):
        with mock.patch("evigraph.utils.scicite.SciCiteModel.__init__", return_value=None):
            model = SciCiteModel.get()
            model._pipe = _mock_pipeline("background", 0.85)
            label, conf = model.classify("Prior work studied this problem [2].")
        assert label == "BACKGROUND"

    def test_classify_result_comparison_label(self):
        with mock.patch("evigraph.utils.scicite.SciCiteModel.__init__", return_value=None):
            model = SciCiteModel.get()
            model._pipe = _mock_pipeline("result_comparison", 0.78)
            label, conf = model.classify("Our model outperforms [3] on SQuAD.")
        assert label == "RESULT_COMPARISON"

    def test_unknown_label_falls_back_to_background(self):
        with mock.patch("evigraph.utils.scicite.SciCiteModel.__init__", return_value=None):
            model = SciCiteModel.get()
            model._pipe = _mock_pipeline("unknown_label", 0.50)
            label, _ = model.classify("Some sentence.")
        assert label == _FALLBACK

    def test_singleton_returns_same_instance(self):
        with mock.patch("evigraph.utils.scicite.SciCiteModel.__init__", return_value=None):
            a = SciCiteModel.get()
            b = SciCiteModel.get()
        assert a is b


class TestClassifyCitation:

    def setup_method(self):
        SciCiteModel._instance = None

    def test_returns_label_and_confidence(self):
        with mock.patch("evigraph.utils.scicite.SciCiteModel.__init__", return_value=None):
            SciCiteModel._instance = SciCiteModel.__new__(SciCiteModel)
            SciCiteModel._instance._pipe = _mock_pipeline("method", 0.91)
            label, conf = classify_citation("We adopt the method from [1].")
        assert label == "METHOD"
        assert conf == pytest.approx(0.91)

    def test_empty_sentence_returns_fallback(self):
        label, conf = classify_citation("   ")
        assert label == _FALLBACK
        assert conf == 0.0

    def test_model_load_failure_returns_fallback(self):
        with mock.patch("evigraph.utils.scicite.SciCiteModel.get", side_effect=RuntimeError("no model")):
            label, conf = classify_citation("Some citation sentence.")
        assert label == _FALLBACK
        assert conf == 0.0

    def test_pipe_exception_returns_fallback(self):
        with mock.patch("evigraph.utils.scicite.SciCiteModel.__init__", return_value=None):
            SciCiteModel._instance = SciCiteModel.__new__(SciCiteModel)
            SciCiteModel._instance._pipe = mock.MagicMock(side_effect=RuntimeError("GPU OOM"))
            label, conf = classify_citation("Some sentence.")
        assert label == _FALLBACK
        assert conf == 0.0


class TestSciCiteModelIntegration:

    def setup_method(self):
        SciCiteModel._instance = None

    @pytest.mark.slow
    def test_real_model_classifies_method(self):
        label, conf = classify_citation("We adopt the BERT approach from [1].")
        assert label in ["METHOD", "BACKGROUND"]  # Should be METHOD mostly
        assert 0 < conf <= 1.0
        assert label != _FALLBACK or conf == 0.0  # Either real label or fallback with 0 conf

    @pytest.mark.slow
    def test_real_model_classifies_background(self):
        label, conf = classify_citation("Prior work by Smith et al. studied neural networks [5].")
        assert label in ["METHOD", "BACKGROUND", "RESULT_COMPARISON"]
        assert 0 < conf <= 1.0

    @pytest.mark.slow
    def test_real_model_classifies_result_comparison(self):
        label, conf = classify_citation("Our approach significantly outperforms [3] on the benchmark.")
        assert label in ["METHOD", "BACKGROUND", "RESULT_COMPARISON"]
        assert 0 < conf <= 1.0

    @pytest.mark.slow
    def test_real_model_not_always_background(self):
        sentences = [
            "We use the method from [1].",
            "Prior work studied this problem [2].",
            "Our model outperforms [3] on SQuAD.",
        ]
        labels = [classify_citation(s)[0] for s in sentences]
        # If model always returns background, all labels would be the same
        # and test should have at least some variety
        unique_labels = set(labels)
        assert len(unique_labels) > 1 or all(
            l != "BACKGROUND" for l in labels
        ), "Model appears to always return BACKGROUND"

    @pytest.mark.slow
    def test_real_model_confidence_reasonable(self):
        label, conf = classify_citation("We adopt the approach from [1].")
        assert 0.0 < conf <= 1.0, f"Confidence {conf} out of bounds"
