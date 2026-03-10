import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.preprocessor import (
    build_arxiv_id_map,
    build_doi_map,
    build_paper_meta,
    build_ref_caption_map,
    build_work_id_map,
    clean_ref_markers,
    make_embed_text,
    make_uid,
    normalize_doi,
    process_text,
)



class TestNormalizeDoi:
    def test_strips_https_prefix(self):
        assert normalize_doi("https://doi.org/10.1000/xyz") == "10.1000/xyz"

    def test_strips_http_prefix(self):
        assert normalize_doi("http://doi.org/10.1000/xyz") == "10.1000/xyz"

    def test_strips_dx_prefix(self):
        assert normalize_doi("https://dx.doi.org/10.1000/xyz") == "10.1000/xyz"

    def test_bare_doi_is_unchanged(self):
        assert normalize_doi("10.1000/xyz") == "10.1000/xyz"

    def test_empty_string(self):
        assert normalize_doi("") == ""

    def test_none_like_empty(self):
        assert normalize_doi(None) == ""  # type: ignore[arg-type]



class TestBuildRefCaptionMap:
    def test_returns_caption_for_known_uuid(self):
        ref_entries = {"abc123": {"caption": "Figure 1: results"}}
        m = build_ref_caption_map(ref_entries)
        assert m["abc123"] == "Figure 1: results"

    def test_empty_caption_entry(self):
        ref_entries = {"abc123": {"caption": ""}}
        m = build_ref_caption_map(ref_entries)
        assert m["abc123"] == ""

    def test_missing_caption_key(self):
        ref_entries = {"abc123": {"other": "value"}}
        m = build_ref_caption_map(ref_entries)
        assert m["abc123"] == ""

    def test_none_entry_gives_empty(self):
        ref_entries = {"abc123": None}
        m = build_ref_caption_map(ref_entries)
        assert m["abc123"] == ""

    def test_empty_ref_entries(self):
        assert build_ref_caption_map({}) == {}



class TestCleanRefMarkers:
    def test_replaces_figure_with_caption(self):
        uid = "aaaa-bbbb"
        text = f"See {{{{figure:{uid}}}}} for details."
        caption_map = {uid: "Plot of accuracy"}
        result = clean_ref_markers(text, caption_map)
        assert "[Figure: Plot of accuracy]" in result
        assert f"{{{{figure:{uid}}}}}" not in result

    def test_replaces_table_with_caption(self):
        uid = "cccc-dddd"
        text = f"Table {{{{table:{uid}}}}} shows counts."
        caption_map = {uid: "Token counts"}
        result = clean_ref_markers(text, caption_map)
        assert "[Table: Token counts]" in result

    def test_removes_marker_when_no_caption(self):
        uid = "eeee-ffff"
        text = f"See {{{{figure:{uid}}}}}."
        caption_map = {uid: ""}
        result = clean_ref_markers(text, caption_map)
        assert f"{{{{figure:{uid}}}}}" not in result
        assert "[Figure" not in result

    def test_no_markers_unchanged(self):
        text = "Plain text without markers."
        assert clean_ref_markers(text, {}) == text



class TestBuildDoiMap:
    def test_uses_ids_doi(self):
        bib = {"ref1": {"ids": {"doi": "10.1000/xyz"}, "bib_entry_raw": ""}}
        m = build_doi_map(bib)
        assert m["ref1"] == "10.1000/xyz"

    def test_strips_url_prefix_from_ids_doi(self):
        bib = {"ref1": {"ids": {"doi": "https://doi.org/10.1000/xyz"}, "bib_entry_raw": ""}}
        m = build_doi_map(bib)
        assert m["ref1"] == "10.1000/xyz"

    def test_mines_doi_from_raw(self):
        bib = {"ref1": {"ids": {}, "bib_entry_raw": "doi: 10.9999/abc"}}
        m = build_doi_map(bib)
        assert m["ref1"] == "10.9999/abc"

    def test_empty_when_no_doi(self):
        bib = {"ref1": {"ids": {}, "bib_entry_raw": "no doi here"}}
        m = build_doi_map(bib)
        assert m["ref1"] == ""

    def test_none_entry(self):
        m = build_doi_map({"ref1": None})
        assert m["ref1"] == ""



class TestBuildArxivIdMap:
    def test_extracts_known_arxiv_id(self):
        bib = {"ref1": {"ids": {"arxiv_id": "1807.04467"}}}
        m = build_arxiv_id_map(bib)
        assert m["ref1"] == "1807.04467"

    def test_empty_when_absent(self):
        bib = {"ref1": {"ids": {}}}
        m = build_arxiv_id_map(bib)
        assert m["ref1"] == ""

    def test_none_entry(self):
        m = build_arxiv_id_map({"ref1": None})
        assert m["ref1"] == ""



class TestProcessText:
    def test_removes_cite_markers(self):
        work_id_map = {"aabbcc": {"work_id": "doi:10.1/x", "doi": "10.1/x", "openalex_id": "", "arxiv_id": ""}}
        text = "Hello {{cite:aabbcc}} world."
        cleaned, spans = process_text(text, work_id_map)
        assert "{{cite:" not in cleaned
        assert cleaned == "Hello  world."

    def test_span_has_correct_work_id(self):
        work_id_map = {"aabbcc": {"work_id": "doi:10.1/x", "doi": "10.1/x", "openalex_id": "", "arxiv_id": ""}}
        _, spans = process_text("See {{cite:aabbcc}}.", work_id_map)
        assert spans[0]["work_id"] == "doi:10.1/x"

    def test_no_cite_markers_unchanged(self):
        text = "No citations here."
        cleaned, spans = process_text(text, {})
        assert cleaned == text
        assert spans == []

    def test_unknown_ref_id_gets_unresolved(self):
        # ref IDs must be all hex digits ([0-9a-f]+) per _CITE_RE
        _, spans = process_text("See {{cite:aabbcc00}}.", {})
        assert spans[0]["work_id"].startswith("unresolved:")

    def test_multiple_citations(self):
        wmap = {
            "11223344": {"work_id": "doi:10.1/a", "doi": "10.1/a", "openalex_id": "", "arxiv_id": ""},
            "55667788": {"work_id": "doi:10.1/b", "doi": "10.1/b", "openalex_id": "", "arxiv_id": ""},
        }
        text = "A {{cite:11223344}} and B {{cite:55667788}}."
        cleaned, spans = process_text(text, wmap)
        assert "{{cite:" not in cleaned
        assert len(spans) == 2



class TestBuildPaperMeta:
    _PAPER = {
        "paper_id": "1234.56789",
        "metadata": {
            "doi":   "10.9999/test",
            "title": "A Great Paper",
            "authors_parsed": [["Smith", "Alice", ""], ["Jones", "Bob", ""]],
            "categories": "cs.AI cs.LG",
            "versions": [{"created": "Mon, 01 Jan 2024 00:00:00 GMT"}],
            "update_date": "2024-01-01",
            "cited_by_count": 42,
            "language": "en",
            "discipline": "Computer Science",
        },
    }

    def test_title_extracted(self):
        meta = build_paper_meta(self._PAPER)
        assert meta["title"] == "A Great Paper"

    def test_doi_extracted(self):
        meta = build_paper_meta(self._PAPER)
        assert meta["doi"] == "10.9999/test"

    def test_authors_parsed_correctly(self):
        meta = build_paper_meta(self._PAPER)
        assert "Alice Smith" in meta["authors"]
        assert "Bob Jones" in meta["authors"]

    def test_categories_split(self):
        meta = build_paper_meta(self._PAPER)
        assert "cs.AI" in meta["categories"]
        assert "cs.LG" in meta["categories"]

    def test_year_extracted(self):
        meta = build_paper_meta(self._PAPER)
        assert meta["year"] == 2024

    def test_missing_metadata_returns_defaults(self):
        meta = build_paper_meta({})
        assert meta["title"] == ""
        assert meta["doi"] == ""
        assert meta["authors"] == []



class TestMakeEmbedText:
    def test_prepends_section_title(self):
        result = make_embed_text("Introduction", "Some body text.")
        assert result == "Introduction: Some body text."

    def test_no_section_title(self):
        result = make_embed_text(None, "Some body text.")
        assert result == "Some body text."

    def test_collapses_double_spaces(self):
        result = make_embed_text("Methods", "Word  gap  here.")
        assert "  " not in result

    def test_empty_title(self):
        result = make_embed_text("", "Body.")
        # empty string is falsy → no prefix
        assert result == "Body."



class TestMakeUid:
    def test_deterministic(self):
        uid1 = make_uid("paper1", "Abstract", "some text")
        uid2 = make_uid("paper1", "Abstract", "some text")
        assert uid1 == uid2

    def test_different_inputs_different_uid(self):
        uid1 = make_uid("paper1", "Abstract", "some text")
        uid2 = make_uid("paper1", "Abstract", "other text")
        assert uid1 != uid2

    def test_returns_sha1_hex(self):
        uid = make_uid("paper1", "Abstract", "some text")
        # SHA-1 hex is 40 chars
        assert len(uid) == 40
        assert all(c in "0123456789abcdef" for c in uid)



class TestBuildWorkIdMapOffline:

    def test_resolves_doi_from_ids(self):
        bib = {
            "ref1": {
                "ids": {"doi": "10.1000/xyz"},
                "bib_entry_raw": "",
            }
        }
        with patch("src.core.preprocessor.verify_doi", return_value=True):
            result = build_work_id_map(bib)
        assert result["ref1"]["work_id"] == "doi:10.1000/xyz"

    def test_resolves_arxiv_id(self):
        bib = {
            "ref1": {
                "ids": {"arxiv_id": "1807.04467"},
                "bib_entry_raw": "",
            }
        }
        result = build_work_id_map(bib)
        assert result["ref1"]["work_id"] == "arxiv:1807.04467"

    def test_none_entry_gives_unresolved(self):
        result = build_work_id_map({"ref1": None})
        assert result["ref1"]["work_id"].startswith("unresolved:")
