from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_qdrant_stub = MagicMock()
sys.modules.setdefault("src", MagicMock())
sys.modules.setdefault("src.utils", MagicMock())
sys.modules.setdefault("src.utils.qdrant", _qdrant_stub)
sys.modules.setdefault("src.config", MagicMock())
sys.modules.setdefault("src.config.settings", MagicMock())

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
from utils.qdrant import (
    POSTPROCESSED_SUFFIX,
    PREVIOUS_SUFFIX,
    backup_previous_qdrant_state,
    capture_postprocessed_qdrant_state,
    path_with_suffix,
    snapshot_name_with_suffix,
)
from indexing.postprocessing.resolve_title import title_score
from indexing.preprocessing.preprocessor import build_paper_meta, make_embed_text, make_uid, process_text
from config.settings import get_qdrant_profile
from indexing.utils import models as real_models
from utils.qdrant import setup_collection

# IMRAD — heuristic_imrad_label
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


# IMRAD — sequence validation helpers
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


# IMRAD — finalize_labels
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


# IMRAD — repair_imrad_sequences
class TestRepairImradSequences:
    def _make_sections_and_order(self, specs):

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


# IMRAD — build_stats
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
        assert stats["sections_labelled_imrad_before"] == 4
        assert stats["sections_labelled_imrad_after"] == 4
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


# IMRAD — collect_non_imrad_sections
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


# resolve_title — title_score
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


# citation_ids — has_public_id
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


# citation_ids — CitationRecord / ProcessingReport
class TestDataclasses:
    def test_citation_record_defaults(self):
        r = CitationRecord(chunk_uid="abc", source_ref_id="ref1", cite_index=0)
        assert r.raw is None

    def test_processing_report_errors_list_not_shared(self):
        r1 = ProcessingReport()
        r2 = ProcessingReport()
        r1.errors.append("err")
        assert r2.errors == []


# citation_ids — resolve_citation
class TestResolveCitation:
    def test_no_raw_returns_none(self):
        record = CitationRecord(chunk_uid="x", source_ref_id="ref", cite_index=0, raw=None)
        assert resolve_citation(record) is None

    def test_unresolved_result_returns_none(self):
        record = CitationRecord(
            chunk_uid="x",
            source_ref_id="ref",
            cite_index=0,
            raw="Some Title. Journal name, 2020.",
        )
        unresolved = {"work_id": "unresolved:ref", "id_source": "unresolved", "doi": "", "openalex_id": "", "arxiv_id": ""}
        with patch("indexing.postprocessing.citation_ids.resolve_bib_entry", return_value=unresolved):
            result = resolve_citation(record)
        assert result is None

    def test_resolved_doi_returned(self):
        record = CitationRecord(
            chunk_uid="x",
            source_ref_id="ref",
            cite_index=0,
            raw="Attention Is All You Need. NIPS 2017.",
        )
        resolved = {"work_id": "doi:10.x/y", "id_source": "doi", "doi": "10.x/y", "openalex_id": "", "arxiv_id": ""}
        with patch("indexing.postprocessing.citation_ids.resolve_bib_entry", return_value=resolved):
            result = resolve_citation(record)
        assert result is not None
        assert result["doi"] == "10.x/y"


# preprocessor 
class TestPreprocessorHelpers:
    @pytest.fixture(autouse=True)
    def _patch_models(self):        
        with patch.dict("sys.modules", {"indexing.models": real_models}):
            yield

    def test_make_embed_text(self):
        result = make_embed_text("Introduction", "This is the body.")
        assert result == "Introduction: This is the body."

    def test_make_embed_text_empty_title(self):
        result = make_embed_text("", "Body only.")
        assert "Body only." in result

    def test_make_embed_text_omits_synthetic_body_title(self):
        result = make_embed_text("Body", "Actual section text.")
        assert result == "Actual section text."

    def test_make_uid_deterministic(self):
        uid1 = make_uid("paper123", "Introduction", "some text here", chunk_index=0)
        uid2 = make_uid("paper123", "Introduction", "some text here", chunk_index=0)
        assert uid1 == uid2
        assert len(uid1) == 40  # SHA1 hex digest

    def test_make_uid_differs_by_input(self):
        assert make_uid("p1", "Introduction", "text", chunk_index=0) != make_uid("p2", "Introduction", "text", chunk_index=0)
        assert make_uid("p1", "Introduction", "text", chunk_index=0) != make_uid("p1", "Methods", "text", chunk_index=0)
        assert make_uid("p1", "Introduction", "text", chunk_index=0) != make_uid("p1", "Introduction", "text", chunk_index=1)

    def test_build_paper_meta_omits_cited_by_count(self):
        meta = build_paper_meta(
            {
                "metadata": {
                    "title": "Paper title",
                    "authors": "Alice, Bob",
                    "categories": "cs.AI",
                    "cited_by_count": 42,
                    "language": "en",
                    "discipline": "Computer Science",
                }
            }
        )
        assert "cited_by_count" not in meta
        assert meta["language"] == "en"
        assert meta["discipline"] == "Computer Science"

    def test_process_text_maps_citation_to_full_sentence(self):
        cleaned, cite_spans = process_text(
            "First sentence. Claim with support {{cite:abc123}} continues here. Last sentence.",
            {"abc123": {"source_ref_id": "abc123", "raw": "Citation raw"}},
        )
        assert cleaned == "First sentence. Claim with support  continues here. Last sentence."
        assert len(cite_spans) == 1
        span = cite_spans[0]
        assert cleaned[span["start"]:span["end"]] == "Claim with support  continues here."

    def test_process_text_keeps_multiple_citations_in_same_sentence(self):
        cleaned, cite_spans = process_text(
            "Same sentence {{cite:abc123}} and again {{cite:def456}}.",
            {
                "abc123": {"source_ref_id": "abc123", "raw": "First raw"},
                "def456": {"source_ref_id": "def456", "raw": "Second raw"},
            },
        )
        assert len(cite_spans) == 2
        expected_sentence = "Same sentence  and again ."
        assert all(cleaned[span["start"]:span["end"]] == expected_sentence for span in cite_spans)

    def test_process_text_ignores_common_abbreviations_for_sentence_bounds(self):
        cleaned, cite_spans = process_text(
            "This follows Phys. Rev. Lett. closely {{cite:abc123}} and stays in one sentence. Next sentence.",
            {"abc123": {"source_ref_id": "abc123", "raw": "Citation raw"}},
        )
        assert len(cite_spans) == 1
        span = cite_spans[0]
        assert cleaned[span["start"]:span["end"]] == "This follows Phys. Rev. Lett. closely  and stays in one sentence."


class TestQdrantCollectionSetup:
    def test_hpc_profile_enables_int8_quantization(self):
        profile = get_qdrant_profile("hpc")
        assert profile.quantize is True
        assert profile.quantize_scalar_type == "int8"
        assert profile.quantize_always_ram is True
        assert 15 <= profile.optimizer.flush_interval_sec <= 30

    def test_setup_collection_uses_profile_configs(self):
        client = MagicMock()
        client.get_collections.return_value.collections = []

        profile = get_qdrant_profile("hpc")

        setup_collection(client, model_key="bge-m3", profile=profile, recreate=False)

        create_kwargs = client.create_collection.call_args.kwargs
        dense_cfg = create_kwargs["vectors_config"][profile.dense_vector_name]
        sparse_cfg = create_kwargs["sparse_vectors_config"][profile.sparse_vector_name]
        optimizers_cfg = create_kwargs["optimizers_config"]
        wal_cfg = create_kwargs["wal_config"]

        assert create_kwargs["on_disk_payload"] is profile.payload_on_disk
        assert dense_cfg.on_disk is profile.vectors_on_disk
        assert dense_cfg.hnsw_config.ef_construct == profile.hnsw.ef_construct
        assert dense_cfg.quantization_config is not None
        assert dense_cfg.quantization_config.scalar.type.name.lower() == profile.quantize_scalar_type
        assert dense_cfg.quantization_config.scalar.always_ram is profile.quantize_always_ram
        assert sparse_cfg.index.on_disk is profile.vectors_on_disk
        assert optimizers_cfg.memmap_threshold == profile.optimizer.memmap_threshold
        assert optimizers_cfg.flush_interval_sec == profile.optimizer.flush_interval_sec
        assert wal_cfg.wal_capacity_mb == profile.wal.wal_capacity_mb

    def test_local_profile_disables_quantization(self):
        client = MagicMock()
        client.get_collections.return_value.collections = []

        profile = get_qdrant_profile("local")

        setup_collection(client, model_key="e5-base-v2", profile=profile, recreate=False)

        create_kwargs = client.create_collection.call_args.kwargs
        dense_cfg = create_kwargs["vectors_config"][profile.dense_vector_name]

        assert profile.quantize is False
        assert dense_cfg.quantization_config is None


class TestPostprocessingQdrantArtifacts:

    def test_path_with_suffix_for_snapshot_file(self, tmp_path):
        assert path_with_suffix(tmp_path / "snapshot-1.snapshot", PREVIOUS_SUFFIX).name == "snapshot-1_previous.snapshot"

    def test_snapshot_name_with_suffix(self):
        assert snapshot_name_with_suffix("collection-123.snapshot", POSTPROCESSED_SUFFIX) == "collection-123_postprocessed.snapshot"

    def test_backup_previous_qdrant_state_records_existing_snapshot(self, tmp_path):
        snapshots_dir = tmp_path / "qdrant_snapshots"
        progress_dir = tmp_path / "progress"
        snapshots_dir.mkdir()
        progress_dir.mkdir()

        (snapshots_dir / "initial.snapshot").write_text("snapshot-before")
        metadata_path = progress_dir / "snapshot.json"
        metadata_path.write_text('{"snapshot_name":"initial.snapshot","collection_name":"papers"}')

        result = backup_previous_qdrant_state(
            metadata_path=metadata_path,
            progress_dir=progress_dir,
        )

        metadata_backup = progress_dir / "snapshot_previous.json"

        assert result["snapshot_name"] == "initial.snapshot"
        assert result["metadata_backup"] == str(metadata_backup)
        assert (snapshots_dir / "initial.snapshot").read_text() == "snapshot-before"
        assert '"snapshot_name": "initial.snapshot"' in metadata_backup.read_text()

    def test_capture_postprocessed_qdrant_state_renames_snapshot(self, tmp_path):
        snapshots_dir = tmp_path / "qdrant_snapshots"
        collection_dir = snapshots_dir / "unarxive_chunks"
        progress_dir = tmp_path / "progress"
        snapshots_dir.mkdir()
        collection_dir.mkdir()
        progress_dir.mkdir()

        (collection_dir / "fresh.snapshot").write_text("snapshot-after")

        result = capture_postprocessed_qdrant_state(
            profile_name="hpc",
            snapshots_dir=snapshots_dir,
            progress_dir=progress_dir,
            snapshot_creator=lambda _profile: "fresh.snapshot",
        )

        snapshot_path = collection_dir / "fresh_postprocessed.snapshot"
        metadata_path = progress_dir / "snapshot_postprocessed.json"

        assert result["snapshot_path"] == str(snapshot_path)
        assert result["metadata_path"] == str(metadata_path)
        assert not (collection_dir / "fresh.snapshot").exists()
        assert snapshot_path.read_text() == "snapshot-after"
        metadata = metadata_path.read_text()
        assert '"snapshot_name": "fresh_postprocessed.snapshot"' in metadata
        assert '"source_snapshot_name": "fresh.snapshot"' in metadata


class TestPostprocessingScripts:
    @pytest.mark.parametrize(
        "script_name",
        [
            "run_postprocessing_ids_capella.sh",
            "run_postprocessing_imrad_capella.sh",
        ],
    )
    def test_scripts_capture_previous_and_postprocessed_qdrant_state(self, script_name):
        script_path = Path(__file__).resolve().parent.parent / "scripts" / script_name
        script_text = script_path.read_text()

        assert "--artifact-mode backup-previous" in script_text
        assert "--artifact-mode capture-postprocessed" in script_text
        assert "_previous" in script_text
        assert "_postprocessed" in script_text
