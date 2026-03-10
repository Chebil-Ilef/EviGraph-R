
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.builder import (
    build_chunk,
    build_paper_chunks,
    chunks_file_exists,
    load_chunks,
    save_chunks,
)

# ── Minimal fixtures ──────────────────────────────────────────────────────────

_WINDOW = {
    "text":          "We study optimization convergence.",
    "section_title": "Introduction",
    "chunk_type":    "subsection",
    "cite_spans":    [],
}

_PAPER_META = {
    "doi":            "10.9999/test",
    "title":          "A Great Paper",
    "authors":        ["Alice Smith", "Bob Jones"],
    "categories":     ["cs.LG"],
    "year":           2024,
    "cited_by_count": 10,
    "language":       "en",
    "discipline":     "CS",
}

_MINIMAL_PAPER = {
    "paper_id": "1234.56789",
    "metadata": {
        "doi":   "10.9999/test",
        "title": "A Great Paper",
        "authors_parsed": [["Smith", "Alice", ""]],
        "categories": "cs.LG",
        "versions": [{"created": "Mon, 01 Jan 2024 00:00:00 GMT"}],
        "update_date": "2024-01-01",
    },
    "abstract": {"text": "Short abstract text for testing. It should produce at least one chunk."},
    "bib_entries": {},
    "ref_entries": {},
    "sections": {},
}


# ── build_chunk ───────────────────────────────────────────────────────────────

class TestBuildChunk:
    def test_returns_dict_with_required_keys(self):
        chunk = build_chunk(_WINDOW, paper_id="1234.56789", paper_doi="10.9/t", paper_meta=_PAPER_META)
        required = {
            "chunk_uid", "chunk_type", "section_title", "embed_text",
            "spans", "paper_doi", "paper_id_arxiv",
            "title", "authors", "categories", "year",
        }
        assert required.issubset(chunk.keys())

    def test_chunk_uid_is_sha1(self):
        chunk = build_chunk(_WINDOW, paper_id="1234.56789", paper_doi="10.9/t", paper_meta=_PAPER_META)
        assert len(chunk["chunk_uid"]) == 40
        assert all(c in "0123456789abcdef" for c in chunk["chunk_uid"])

    def test_embed_text_includes_section_title(self):
        chunk = build_chunk(_WINDOW, paper_id="1234.56789", paper_doi="10.9/t", paper_meta=_PAPER_META)
        assert "Introduction" in chunk["embed_text"]

    def test_paper_id_arxiv_matches_input(self):
        chunk = build_chunk(_WINDOW, paper_id="1234.56789", paper_doi="10.9/t", paper_meta=_PAPER_META)
        assert chunk["paper_id_arxiv"] == "1234.56789"

    def test_cite_spans_wrapped_in_spans_dict(self):
        chunk = build_chunk(_WINDOW, paper_id="1234.56789", paper_doi="10.9/t", paper_meta=_PAPER_META)
        assert "cite_spans" in chunk["spans"]

    def test_deterministic_uid(self):
        c1 = build_chunk(_WINDOW, paper_id="1234.56789", paper_doi="10.9/t", paper_meta=_PAPER_META)
        c2 = build_chunk(_WINDOW, paper_id="1234.56789", paper_doi="10.9/t", paper_meta=_PAPER_META)
        assert c1["chunk_uid"] == c2["chunk_uid"]


# ── build_paper_chunks ────────────────────────────────────────────────────────

class TestBuildPaperChunks:
    """
    Mocks the tokenizer and bib-entry work-id resolution so no model
    download or network call happens.
    """

    def _run(self, paper=None):
        paper = paper or _MINIMAL_PAPER

        # Patch build_work_id_map to return empty (no network needed)
        # Patch get_tokenizer to return a fast char-counting stub
        fake_tok = MagicMock()
        fake_tok.encode = lambda text, add_special_tokens=False: list(range(max(1, len(text))))

        with patch("src.core.builder.get_tokenizer", return_value=fake_tok), \
             patch("src.core.builder.build_work_id_map", return_value={}), \
             patch("src.core.builder.build_ref_caption_map", return_value={}), \
             patch("src.core.preprocessor.resolve_title", return_value=None):
            return build_paper_chunks(paper)

    def test_returns_list(self):
        chunks = self._run()
        assert isinstance(chunks, list)

    def test_each_chunk_has_required_keys(self):
        chunks = self._run()
        required = {"chunk_uid", "chunk_type", "section_title", "embed_text", "paper_id_arxiv"}
        for chunk in chunks:
            assert required.issubset(chunk.keys())

    def test_paper_id_propagated(self):
        chunks = self._run()
        for chunk in chunks:
            assert chunk["paper_id_arxiv"] == "1234.56789"

    def test_abstract_chunk_present(self):
        chunks = self._run()
        types = {c["chunk_type"] for c in chunks}
        assert "abstract" in types

    def test_sections_processed(self):
        paper = {**_MINIMAL_PAPER, "sections": {
            "Methods": {"text": "We trained models for 100 epochs. Results are positive."}
        }}
        chunks = self._run(paper)
        types = {c["chunk_type"] for c in chunks}
        assert "subsection" in types

    def test_empty_paper_returns_no_section_chunks(self):
        paper = {**_MINIMAL_PAPER, "sections": {}}
        chunks = self._run(paper)
        section_chunks = [c for c in chunks if c["chunk_type"] == "subsection"]
        assert section_chunks == []


# ── save_chunks / load_chunks / chunks_file_exists ───────────────────────────

class TestChunkIO:
    def _sample_chunks(self):
        return [
            build_chunk(_WINDOW, paper_id="p1", paper_doi="10.1/x", paper_meta=_PAPER_META),
            build_chunk(
                {**_WINDOW, "section_title": "Methods", "text": "Another sentence."},
                paper_id="p1", paper_doi="10.1/x", paper_meta=_PAPER_META,
            ),
        ]

    def test_save_then_load_roundtrip(self, tmp_path):
        chunks = self._sample_chunks()
        stem = "test_batch"

        with patch("src.core.builder.PATHS") as mock_paths:
            mock_paths.chunks = tmp_path
            save_chunks(chunks, stem)
            loaded = load_chunks(stem)

        assert len(loaded) == len(chunks)
        assert loaded[0]["chunk_uid"] == chunks[0]["chunk_uid"]

    def test_load_nonexistent_returns_empty(self, tmp_path):
        with patch("src.core.builder.PATHS") as mock_paths:
            mock_paths.chunks = tmp_path
            result = load_chunks("nonexistent_batch")
        assert result == []

    def test_chunks_file_exists_true(self, tmp_path):
        stem = "exists_batch"
        (tmp_path / f"{stem}_chunks.jsonl").write_text('{"a":1}\n')
        with patch("src.core.builder.PATHS") as mock_paths:
            mock_paths.chunks = tmp_path
            assert chunks_file_exists(stem) is True

    def test_chunks_file_exists_false(self, tmp_path):
        with patch("src.core.builder.PATHS") as mock_paths:
            mock_paths.chunks = tmp_path
            assert chunks_file_exists("missing_batch") is False

    def test_saved_file_is_valid_jsonl(self, tmp_path):
        chunks = self._sample_chunks()
        stem = "jsonl_test"
        with patch("src.core.builder.PATHS") as mock_paths:
            mock_paths.chunks = tmp_path
            save_chunks(chunks, stem)
        lines = (tmp_path / f"{stem}_chunks.jsonl").read_text().splitlines()
        assert len(lines) == len(chunks)
        for line in lines:
            parsed = json.loads(line)
            assert "chunk_uid" in parsed
