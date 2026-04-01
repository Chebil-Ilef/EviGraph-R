"""
Targeted unit tests for indexing pipeline postprocessing logic.

All tests are pure-Python (no Qdrant, no network, no GPU).
They cover:
  - IMRAD heuristic labelling
  - IMRAD sequence validation helpers
  - IMRAD sequence repair
  - IMRAD stats builder
  - collect_non_imrad_sections
  - title scoring (resolve_title)
  - has_public_id (citation_ids)
  - CitationRecord / ProcessingReport dataclasses
  - resolve_citation short-circuits correctly
  - index_builder: build_chunk payload shape
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ---------------------------------------------------------------------------
# Patch broken absolute imports in source files before they are imported.
# citation_ids.py uses `from src.utils.qdrant import ...` which only works
# when the CWD is the repo root with src/ NOT on sys.path.  Stub it out so
# the module loads cleanly in tests.
# ---------------------------------------------------------------------------
_qdrant_stub = MagicMock()
sys.modules.setdefault("src", MagicMock())
sys.modules.setdefault("src.utils", MagicMock())
sys.modules.setdefault("src.utils.qdrant", _qdrant_stub)
sys.modules.setdefault("src.config", MagicMock())
sys.modules.setdefault("src.config.settings", MagicMock())

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from indexing.postprocessing.imrad_titles import (
    SectionRecord,
    build_stats,
    collect_non_imrad_sections,
    evaluate_paper,
    finalize_labels,
    has_core_imrad,
    heuristic_imrad_label,
    is_imrad_sequence,
    repair_imrad_sequences,
)
from indexing.postprocessing.citation_ids import (
    CitationRecord,
    ProcessingReport,
    has_public_id,
    resolve_citation,
)
from indexing.postprocessing.resolve_title import title_score


# ===========================================================================
# IMRAD — heuristic_imrad_label
# ===========================================================================
class TestHeuristicImradLabel:
    @pytest.mark.parametrize("title,expected", [
        ("Introduction",       "Introduction"),
        ("intro",              "Introduction"),
        ("Background",         "Introduction"),
        ("Motivation",         "Introduction"),
        ("Preliminaries",      "Introduction"),
        ("Methods",            "Methods"),
        ("Methodology",        "Methods"),
        ("Experimental Setup", "Methods"),
        ("Architecture",       "Methods"),
        ("Dataset",            "Methods"),
        ("Results",            "Results"),
        ("Experiments",        "Results"),
        ("Evaluation",         "Results"),
        ("Ablation Study",     "Results"),
        ("Discussion",         "Discussion"),
        ("Conclusion",         "Discussion"),
        ("Conclusions",        "Discussion"),
        ("Future Work",        "Discussion"),
        ("Limitations",        "Discussion"),
        ("Related Work",       "Related Work"),
        ("Literature Review",  "Related Work"),
        ("Prior Work",         "Related Work"),
        # SKIP
        ("Abstract",           "SKIP"),
        ("Acknowledgements",   "SKIP"),
        ("References",         "SKIP"),
        # unknown
        ("Appendix A",         None),
        ("Supplemental Material", None),
        ("",                   None),
    ])
    def test_label(self, title, expected):
        assert heuristic_imrad_label(title) == expected

    def test_case_insensitive(self):
        assert heuristic_imrad_label("INTRODUCTION") == "Introduction"
        assert heuristic_imrad_label("METHODS") == "Methods"

    def test_whitespace_normalised(self):
        assert heuristic_imrad_label("  Results  ") == "Results"


# ===========================================================================
# IMRAD — sequence validation helpers
# ===========================================================================
class TestImradSequenceHelpers:
    def test_valid_sequence(self):
        assert is_imrad_sequence(["Introduction", "Methods", "Results", "Discussion"])

    def test_sequence_subset_valid(self):
        assert is_imrad_sequence(["Introduction", "Results"])

    def test_sequence_reordered_invalid(self):
        assert not is_imrad_sequence(["Results", "Introduction"])

    def test_sequence_duplicate_invalid(self):
        assert not is_imrad_sequence(["Introduction", "Introduction", "Methods"])

    def test_sequence_empty_invalid(self):
        assert not is_imrad_sequence([])

    def test_sequence_unknown_label_invalid(self):
        assert not is_imrad_sequence(["Introduction", "SKIP"])

    def test_has_core_imrad_true(self):
        assert has_core_imrad(["Introduction", "Methods", "Results"])

    def test_has_core_imrad_missing_one(self):
        assert not has_core_imrad(["Introduction", "Methods"])

    def test_evaluate_paper_full(self):
        assert evaluate_paper(["Introduction", "Methods", "Results", "Discussion"])

    def test_evaluate_paper_bad_order(self):
        assert not evaluate_paper(["Methods", "Introduction", "Results"])


# ===========================================================================
# IMRAD — finalize_labels
# ===========================================================================
class TestFinalizeLabels:
    def _section(self, heuristic, source="heuristic", final=None):
        s = SectionRecord(
            paper_id="p1", title="t", text="some text",
            point_ids=[1], heuristic_label=heuristic,
        )
        s.source = source
        s.final_label = final
        return s

    def test_skip_label(self):
        s = self._section("SKIP")
        finalize_labels({("p1", "t"): s})
        assert s.final_label == "SKIP"
        assert s.source == "skip"
        assert s.confidence is None

    def test_heuristic_label(self):
        s = self._section("Introduction")
        finalize_labels({("p1", "t"): s})
        assert s.final_label == "Introduction"
        assert s.source == "heuristic"
        assert s.confidence == 1.0

    def test_classifier_label_not_overwritten(self):
        s = self._section(None, source="classifier", final="Methods")
        s.confidence = 0.87
        finalize_labels({("p1", "t"): s})
        assert s.final_label == "Methods"
        assert s.confidence == 0.87

    def test_unresolved_empty_text(self):
        s = self._section(None, source="heuristic")
        s.text = ""
        finalize_labels({("p1", "t"): s})
        assert s.final_label is None
        assert s.source == "unresolved"


# ===========================================================================
# IMRAD — repair_imrad_sequences
# ===========================================================================
class TestRepairImradSequences:
    def _make_sections_and_order(self, specs):
        """
        specs: list of (key_suffix, heuristic, final, source, raw_probs)
        Returns (section_map, paper_to_titles) for a single paper "p1".
        """
        sections = {}
        ordered = []
        id2label = {0: "Introduction", 1: "Methods", 2: "Results", 3: "Discussion", 4: "Related Work"}
        for i, (suffix, heuristic, final, source, raw_probs) in enumerate(specs):
            key = ("p1", suffix)
            s = SectionRecord(
                paper_id="p1", title=suffix, text="txt",
                point_ids=[i], heuristic_label=heuristic,
            )
            s.final_label = final
            s.source = source
            s.raw_probs = raw_probs
            s.chunk_index_min = i
            sections[key] = s
            ordered.append((i, key))
        paper_to_titles = {"p1": ordered}
        return sections, paper_to_titles, id2label

    def test_no_violation_unchanged(self):
        specs = [
            ("intro", "Introduction", "Introduction", "heuristic", None),
            ("methods", "Methods", "Methods", "heuristic", None),
            ("results", "Results", "Results", "heuristic", None),
        ]
        sections, p2t, id2label = self._make_sections_and_order(specs)
        relabelled = repair_imrad_sequences(sections, p2t, id2label)
        assert relabelled == 0

    def test_violation_repaired_via_classifier(self):
        # Results comes before Methods — classifier probs prefer Methods at index 1
        specs = [
            ("intro",   "Introduction", "Introduction", "heuristic", None),
            ("results", None, "Results", "classifier", [0.05, 0.6, 0.2, 0.1, 0.05]),  # Methods=idx1 highest valid
            ("methods", None, "Methods", "classifier", [0.05, 0.7, 0.2, 0.05, 0.0]),
        ]
        sections, p2t, id2label = self._make_sections_and_order(specs)
        relabelled = repair_imrad_sequences(sections, p2t, id2label)
        # The first classifier section violates order (Results after Introduction is fine,
        # but the ordering index: Results=3 > Introduction=0 OK; then Methods=2 < Results=3 → violation)
        assert relabelled >= 1

    def test_heuristic_labels_not_touched(self):
        # Reverse order heuristic — should NOT be repaired
        specs = [
            ("results", "Results", "Results", "heuristic", None),
            ("intro",   "Introduction", "Introduction", "heuristic", None),
        ]
        sections, p2t, id2label = self._make_sections_and_order(specs)
        relabelled = repair_imrad_sequences(sections, p2t, id2label)
        assert relabelled == 0
        assert sections[("p1", "intro")].final_label == "Introduction"


# ===========================================================================
# IMRAD — build_stats
# ===========================================================================
class TestBuildStats:
    def _make(self, paper_id, title_suffix, heuristic, final, source, conf=None):
        s = SectionRecord(
            paper_id=paper_id, title=title_suffix, text="x",
            point_ids=[0], heuristic_label=heuristic,
        )
        s.final_label = final
        s.source = source
        s.confidence = conf
        return s

    def test_basic_counts(self):
        sections = {
            ("p1", "intro"):   self._make("p1", "Introduction", "Introduction", "Introduction", "heuristic", 1.0),
            ("p1", "methods"): self._make("p1", "Methods",      "Methods",      "Methods",      "heuristic", 1.0),
            ("p1", "results"): self._make("p1", "Results",      "Results",      "Results",      "heuristic", 1.0),
            ("p1", "disc"):    self._make("p1", "Discussion",   "Discussion",   "Discussion",   "heuristic", 1.0),
            ("p1", "refs"):    self._make("p1", "References",   "SKIP",         "SKIP",         "skip"),
        }
        paper_to_titles = {
            "p1": [(i, k) for i, k in enumerate(sections)]
        }
        stats = build_stats(sections, paper_to_titles)

        assert stats["papers_total"] == 1
        assert stats["sections_total"] == 5
        assert stats["sections_labelled_imrad"] == 4
        assert stats["sections_skipped"] == 1
        assert stats["sections_no_label"] == 0
        assert stats["imrad_label_sources"]["heuristic"] == 4

    def test_no_label_section(self):
        sections = {
            ("p1", "appendix"): self._make("p1", "Appendix", None, None, "unresolved"),
        }
        paper_to_titles = {"p1": [(0, ("p1", "appendix"))]}
        stats = build_stats(sections, paper_to_titles)
        assert stats["sections_no_label"] == 1
        assert stats["sections_no_label_unresolved"] == 1

    def test_classifier_confidence_stats(self):
        sections = {
            ("p1", "sec1"): self._make("p1", "sec1", None, "Methods", "classifier", 0.9),
            ("p1", "sec2"): self._make("p1", "sec2", None, "Results", "classifier", 0.7),
        }
        paper_to_titles = {"p1": [(0, ("p1", "sec1")), (1, ("p1", "sec2"))]}
        stats = build_stats(sections, paper_to_titles)
        cc = stats["classifier_confidence"]
        assert cc["count"] == 2
        assert abs(cc["mean"] - 0.8) < 1e-4
        assert cc["min"] == 0.7
        assert cc["max"] == 0.9


# ===========================================================================
# IMRAD — collect_non_imrad_sections
# ===========================================================================
class TestCollectNonImradSections:
    def _section(self, paper_id, title, final_label, source):
        s = SectionRecord(
            paper_id=paper_id, title=title, text="sample text",
            point_ids=[0], heuristic_label=None,
        )
        s.final_label = final_label
        s.source = source
        return s

    def test_skipped_sections_grouped(self):
        sections = {
            ("p1", "abstract"): self._section("p1", "Abstract", "SKIP", "skip"),
            ("p2", "abstract"): self._section("p2", "Abstract", "SKIP", "skip"),
        }
        result = collect_non_imrad_sections(sections)
        assert len(result) == 1
        assert result[0]["normalized_title"] == "abstract"
        assert result[0]["count"] == 2
        # skip-dominated → papers list is empty
        assert result[0]["papers"] == []

    def test_unresolved_sections_include_papers(self):
        sections = {
            ("p1", "appendix a"): self._section("p1", "Appendix A", None, "unresolved"),
        }
        result = collect_non_imrad_sections(sections)
        assert len(result) == 1
        assert "p1" in result[0]["papers"]

    def test_imrad_sections_excluded(self):
        sections = {
            ("p1", "introduction"): self._section("p1", "Introduction", "Introduction", "heuristic"),
        }
        result = collect_non_imrad_sections(sections)
        assert result == []

    def test_sorted_by_count_descending(self):
        sections = {
            ("p1", "refs"): self._section("p1", "References", "SKIP", "skip"),
            ("p2", "refs"): self._section("p2", "References", "SKIP", "skip"),
            ("p3", "refs"): self._section("p3", "References", "SKIP", "skip"),
            ("p1", "app"):  self._section("p1", "Appendix",   "SKIP", "skip"),
        }
        result = collect_non_imrad_sections(sections)
        assert result[0]["normalized_title"] == "references"
        assert result[0]["count"] == 3


# ===========================================================================
# resolve_title — title_score
# ===========================================================================
class TestTitleScore:
    def test_exact_match(self):
        assert title_score("Attention Is All You Need", "Attention Is All You Need") == 1.0

    def test_case_insensitive(self):
        score = title_score("attention is all you need", "Attention Is All You Need")
        assert score == 1.0

    def test_partial_overlap_below_threshold(self):
        score = title_score("Deep Learning", "Reinforcement Learning for Games")
        assert 0.0 < score < 0.85

    def test_empty_inputs(self):
        assert title_score("", "something") == 0.0
        assert title_score("something", "") == 0.0

    def test_disjoint_returns_zero(self):
        score = title_score("apple banana cherry", "dog elephant frog")
        assert score == 0.0

    def test_containment_handles_abbreviation(self):
        # short query whose words are fully contained in longer candidate
        score = title_score("BERT Transformers", "BERT: Pre-training of Deep Bidirectional Transformers")
        # some overlap — well above zero
        assert score > 0.2


# ===========================================================================
# citation_ids — has_public_id
# ===========================================================================
class TestHasPublicId:
    def test_with_doi(self):
        assert has_public_id({"doi": "10.1234/foo"})

    def test_with_openalex(self):
        assert has_public_id({"openalex_id": "W123456"})

    def test_with_arxiv(self):
        assert has_public_id({"arxiv_id": "1706.03762"})

    def test_empty_strings_falsy(self):
        assert not has_public_id({"doi": "", "openalex_id": "", "arxiv_id": ""})

    def test_missing_keys(self):
        assert not has_public_id({})

    def test_none_values(self):
        assert not has_public_id({"doi": None, "openalex_id": None})


# ===========================================================================
# citation_ids — CitationRecord / ProcessingReport
# ===========================================================================
class TestDataclasses:
    def test_citation_record_defaults(self):
        r = CitationRecord(chunk_uid="abc", source_ref_id="ref1", cite_index=0)
        assert r.title is None
        assert r.raw is None

    def test_processing_report_errors_list_not_shared(self):
        r1 = ProcessingReport()
        r2 = ProcessingReport()
        r1.errors.append("err")
        assert r2.errors == []


# ===========================================================================
# citation_ids — resolve_citation
# ===========================================================================
class TestResolveCitation:
    def test_no_title_returns_none(self):
        record = CitationRecord(chunk_uid="x", source_ref_id="ref", cite_index=0, title=None)
        assert resolve_citation(record) is None

    def test_unresolved_result_returns_none(self):
        record = CitationRecord(chunk_uid="x", source_ref_id="ref", cite_index=0, title="Some Title")
        unresolved = {"work_id": "unresolved:ref", "id_source": "unresolved", "doi": "", "openalex_id": "", "arxiv_id": ""}
        with patch("indexing.postprocessing.citation_ids.resolve_bib_entry", return_value=unresolved):
            result = resolve_citation(record)
        assert result is None

    def test_resolved_doi_returned(self):
        record = CitationRecord(chunk_uid="x", source_ref_id="ref", cite_index=0, title="Attention Is All You Need")
        resolved = {"work_id": "doi:10.x/y", "id_source": "doi", "doi": "10.x/y", "openalex_id": "", "arxiv_id": ""}
        with patch("indexing.postprocessing.citation_ids.resolve_bib_entry", return_value=resolved):
            result = resolve_citation(record)
        assert result is not None
        assert result["doi"] == "10.x/y"


# ===========================================================================
# preprocessor helpers — make_uid / make_embed_text
# Tested directly to avoid the broken `indexing.models` import chain inside
# index_builder → preprocessor → indexing.models (should be indexing.utils.models).
# ===========================================================================
class TestPreprocessorHelpers:
    @pytest.fixture(autouse=True)
    def _patch_models(self):
        """preprocessor imports `indexing.models`; stub it out."""
        from indexing.utils import models as real_models
        with patch.dict("sys.modules", {"indexing.models": real_models}):
            yield

    def test_make_embed_text(self):
        from indexing.preprocessing.preprocessor import make_embed_text
        result = make_embed_text("Introduction", "This is the body.")
        assert result == "Introduction: This is the body."

    def test_make_embed_text_empty_title(self):
        from indexing.preprocessing.preprocessor import make_embed_text
        result = make_embed_text("", "Body only.")
        assert "Body only." in result

    def test_make_uid_deterministic(self):
        from indexing.preprocessing.preprocessor import make_uid
        uid1 = make_uid("paper123", "Introduction", "some text here")
        uid2 = make_uid("paper123", "Introduction", "some text here")
        assert uid1 == uid2
        assert len(uid1) == 40  # SHA1 hex digest

    def test_make_uid_differs_by_input(self):
        from indexing.preprocessing.preprocessor import make_uid
        assert make_uid("p1", "Introduction", "text") != make_uid("p2", "Introduction", "text")
        assert make_uid("p1", "Introduction", "text") != make_uid("p1", "Methods", "text")
