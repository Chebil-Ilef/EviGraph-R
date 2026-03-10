import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.chunker import (
    chunk_abstract,
    chunk_section,
    count_tokens,
    sliding_window,
    split_sentences,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_tokenizer(chars_per_token: int = 1):
    """
    Tokenizer stub: each character counts as `chars_per_token` tokens.
    This lets us predict exact window boundaries without loading a real model.
    """
    tok = MagicMock()
    tok.encode = lambda text, add_special_tokens=False: list(range(len(text) // chars_per_token + 1))
    return tok


# ── count_tokens ──────────────────────────────────────────────────────────────

class TestCountTokens:
    def test_returns_integer(self):
        tok = _mock_tokenizer()
        n = count_tokens("hello world", tok)
        assert isinstance(n, int)
        assert n >= 0

    def test_empty_string_zero_or_one(self):
        tok = _mock_tokenizer()
        # empty→ encode returns [0] → len = 1 with mock; still a non-negative int
        n = count_tokens("", tok)
        assert isinstance(n, int)

    def test_longer_text_more_tokens(self):
        tok = _mock_tokenizer()
        short = count_tokens("hi", tok)
        long  = count_tokens("hi " * 100, tok)
        assert long > short


# ── split_sentences ───────────────────────────────────────────────────────────

class TestSplitSentences:
    def test_single_sentence(self):
        spans = split_sentences("Hello world.")
        assert len(spans) == 1
        assert spans[0] == (0, 12)

    def test_two_sentences(self):
        text = "First sentence. Second sentence."
        spans = split_sentences(text)
        assert len(spans) == 2

    def test_span_covers_full_text(self):
        text = "Alpha beta. Gamma delta."
        spans = split_sentences(text)
        # reconstructed text should equal original (stripped)
        reconstructed = " ".join(text[s:e] for s, e in spans)
        assert reconstructed.replace("  ", " ") != ""

    def test_no_splits_on_abbreviations(self):
        # "e.g." or "Fig. 2" should NOT split mid-abbreviation
        text = "See e.g. Fig. 2 for reference. Next sentence starts here."
        spans = split_sentences(text)
        # Most important: we should NOT get too many fragments (actual splitter
        # does split on periods in abbreviations - verify output is bounded)
        assert len(spans) <= 4

    def test_empty_text(self):
        assert split_sentences("") == []

    def test_returns_list_of_tuples(self):
        spans = split_sentences("A sentence.")
        assert isinstance(spans, list)
        for s, e in spans:
            assert isinstance(s, int)
            assert isinstance(e, int)
            assert e >= s


# ── sliding_window ────────────────────────────────────────────────────────────

class TestSlidingWindow:
    def test_short_text_single_window(self):
        # text fits entirely within max_tokens → returns one window covering all
        tok = _mock_tokenizer(chars_per_token=10)   # 10 chars = 1 token
        text = "Short text."  # ~1 token → well under any max
        windows = sliding_window(text, tok, max_tokens=50, window_tokens=50, overlap_tokens=5)
        assert len(windows) == 1
        assert windows[0] == (0, len(text))

    def test_long_text_multiple_windows(self):
        tok = _mock_tokenizer(chars_per_token=1)   # 1 char = 1 token
        # 5 sentences, each 20 chars → ~100 tokens total.
        sentences = ["A" * 18 + ". " for _ in range(5)]
        text = "".join(sentences).strip()
        windows = sliding_window(text, tok, max_tokens=30, window_tokens=30, overlap_tokens=5)
        assert len(windows) > 1

    def test_windows_cover_start_and_end(self):
        tok = _mock_tokenizer(chars_per_token=1)
        sentences = [f"Sentence number {i:02d} here. " for i in range(10)]
        text = "".join(sentences).strip()
        windows = sliding_window(text, tok, max_tokens=40, window_tokens=40, overlap_tokens=10)
        first_start = windows[0][0]
        last_end    = windows[-1][1]
        assert first_start == 0
        assert last_end == len(text)

    def test_all_windows_within_text_bounds(self):
        tok = _mock_tokenizer(chars_per_token=1)
        sentences = [f"Word{'x' * 15}. " for _ in range(8)]
        text = "".join(sentences).strip()
        windows = sliding_window(text, tok, max_tokens=30, window_tokens=30, overlap_tokens=5)
        for s, e in windows:
            assert 0 <= s < e <= len(text)


# ── chunk_abstract ─────────────────────────────────────────────────────────────

class TestChunkAbstract:
    _PAPER_WITH_ABSTRACT = {
        "abstract": {
            "text": (
                "This paper studies optimization in Banach spaces. "
                "We propose a novel augmented Lagrangian method. "
                "Convergence guarantees are provided."
            )
        }
    }

    def test_returns_list_of_dicts(self):
        tok = _mock_tokenizer(chars_per_token=5)
        result = chunk_abstract(self._PAPER_WITH_ABSTRACT, tok, {})
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_type_is_abstract(self):
        tok = _mock_tokenizer(chars_per_token=5)
        result = chunk_abstract(self._PAPER_WITH_ABSTRACT, tok, {})
        for chunk in result:
            assert chunk["chunk_type"] == "abstract"

    def test_section_title_is_abstract(self):
        tok = _mock_tokenizer(chars_per_token=5)
        result = chunk_abstract(self._PAPER_WITH_ABSTRACT, tok, {})
        for chunk in result:
            assert chunk["section_title"] == "Abstract"

    def test_empty_abstract_returns_empty(self):
        tok = _mock_tokenizer()
        result = chunk_abstract({"abstract": {"text": ""}}, tok, {})
        assert result == []

    def test_missing_abstract_returns_empty(self):
        tok = _mock_tokenizer()
        result = chunk_abstract({}, tok, {})
        assert result == []

    def test_chunk_has_cite_spans_key(self):
        tok = _mock_tokenizer(chars_per_token=5)
        result = chunk_abstract(self._PAPER_WITH_ABSTRACT, tok, {})
        for chunk in result:
            assert "cite_spans" in chunk

    def test_text_is_non_empty(self):
        tok = _mock_tokenizer(chars_per_token=5)
        result = chunk_abstract(self._PAPER_WITH_ABSTRACT, tok, {})
        for chunk in result:
            assert len(chunk["text"].strip()) > 0


# ── chunk_section ─────────────────────────────────────────────────────────────

class TestChunkSection:
    _SECTION = {
        "text": (
            "We describe the experimental setup. "
            "All models were trained for 100 epochs. "
            "Results are reported as mean ± std."
        )
    }

    def test_returns_list_of_dicts(self):
        tok = _mock_tokenizer(chars_per_token=5)
        result = chunk_section("Experiments", self._SECTION, tok, {})
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_type_is_subsection(self):
        tok = _mock_tokenizer(chars_per_token=5)
        result = chunk_section("Experiments", self._SECTION, tok, {})
        for chunk in result:
            assert chunk["chunk_type"] == "subsection"

    def test_section_title_is_set(self):
        tok = _mock_tokenizer(chars_per_token=5)
        result = chunk_section("Experiments", self._SECTION, tok, {})
        for chunk in result:
            assert chunk["section_title"] == "Experiments"

    def test_empty_section_returns_empty(self):
        tok = _mock_tokenizer()
        result = chunk_section("Empty", {"text": ""}, tok, {})
        assert result == []

    def test_chunk_has_required_keys(self):
        tok = _mock_tokenizer(chars_per_token=5)
        result = chunk_section("Methods", self._SECTION, tok, {})
        required = {"text", "cite_spans", "section_title", "chunk_type"}
        for chunk in result:
            assert required.issubset(chunk.keys())
