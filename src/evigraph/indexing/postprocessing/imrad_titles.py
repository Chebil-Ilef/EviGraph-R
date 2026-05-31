from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from evigraph.config.settings import PATHS, QDRANT_ACTIVE
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import ExtendedPointId
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from evigraph.utils.qdrant import ensure_qdrant_runtime, qdrant_client

logger = logging.getLogger(__name__)

IMRAD_ORDER = {
    "Introduction": 0,
    "Methods": 1,
    "Results": 2,
    "Discussion": 3,
}

SKIP_LABEL = "SKIP"

INTRO_PATTERNS = [
    r"^intro(duction)?$",
    r"^introduction section$",
    r"^background$",
    r"^background and motivation$",
    r"^motivation$",
    r"^overview$",
    r"^problem (statement|formulation|definition|setup)$",
    r"^preliminaries$",
    r"^setup$",
    r"^related work$",
    r"^related work and (prior|background)$",
    r"^literature review$",
    r"^prior work$",
    r"^related (research|literature|studies)$",
    r"^background and related work$",
    r"^state of the art$",
    r"^previous work$",
]
METHODS_PATTERNS = [
    r"^methods?$",
    r"^materials? and methods?$",
    r"^methodology$",
    r"^approach$",
    r"^proposed (method|approach|model|framework|algorithm)$",
    r"^model$",
    r"^framework$",
    r"^algorithm(s)?$",
    r"^implementation$",
    r"^experimental setup$",
    r"^dataset(s)?$",
    r"^data$",
    r"^data (collection|preparation|preprocessing)$",
    r"^training (procedure|details|setup)$",
    r"^(our )?system$",
    r"^architecture$",
    r"^technical (approach|details)$",
]
RESULTS_PATTERNS = [
    r"^results?$",
    r"^experiments?$",
    r"^evaluation$",
    r"^experimental results?$",
    r"^results? and (analysis|discussion|evaluation)$",
    r"^analysis$",
    r"^ablation(s| study| studies)?$",
    r"^performance$",
    r"^benchmarks?$",
    r"^quantitative (results?|evaluation|analysis)$",
    r"^numerical (results?|experiments?|evaluation)$",
    r"^case stud(y|ies)$",
    r"^empirical (results?|evaluation|study)$",
]
DISCUSSION_PATTERNS = [
    r"^discussion$",
    r"^conclusion(s)?$",
    r"^conclusions and future work$",
    r"^summary( and conclusion(s)?)?$",
    r"^future (work|directions?)$",
    r"^limitations?$",
    r"^discussion and (conclusion(s)?|future work|summary)$",
    r"^concluding remarks?$",
]
SKIP_PATTERNS = [
    r"^abstract$",
    r"^acknowledg(e?ments?|ements?)$",
    r"^references$",
    r"^bibliography$",
    r"^index$",
    r"^table of contents$",
    r"^list of (figures|tables|algorithms)$",
]


@dataclass
class SectionRecord:
    paper_id: str
    title: str
    text: str
    point_ids: list[Any]
    heuristic_label: str | None
    chunk_index_min: int = 0
    final_label: str | None = None
    source: str = "heuristic"
    confidence: float | None = None
    # Per-class probability vector from the classifier; None for heuristic/skip sections.
    raw_probs: list[float] | None = None
    # True when imrad_section_title was already present in Qdrant — skip classify + write.
    already_written: bool = False


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def _first_pattern_match(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def heuristic_imrad_label(title: str) -> str | None:
    norm = _normalize_title(title)
    if not norm:
        return None
    if _first_pattern_match(norm, SKIP_PATTERNS):
        return SKIP_LABEL
    if _first_pattern_match(norm, INTRO_PATTERNS):
        return "Introduction"
    if _first_pattern_match(norm, METHODS_PATTERNS):
        return "Methods"
    if _first_pattern_match(norm, RESULTS_PATTERNS):
        return "Results"
    if _first_pattern_match(norm, DISCUSSION_PATTERNS):
        return "Discussion"
    return None


def is_imrad_sequence(labels: list[str]) -> bool:
    if not labels:
        return False
    if any(label not in IMRAD_ORDER for label in labels):
        return False
    order_values = [IMRAD_ORDER[label] for label in labels]
    return all(next_value > value for value, next_value in zip(order_values, order_values[1:]))


def has_core_imrad(labels: list[str]) -> bool:
    required = {"Introduction", "Methods", "Results"}
    return required.issubset(set(labels))


def evaluate_paper(labels: list[str]) -> bool:
    return is_imrad_sequence(labels) and has_core_imrad(labels)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(device)


# ---------------------------------------------------------------------------
# Point → section accumulation helpers (shared by scroll paths)
# ---------------------------------------------------------------------------

def _accumulate_point(
    point,
    section_map: dict[tuple[str, str], SectionRecord],
    section_min_idx: dict[tuple[str, str], int],
    paper_sections: dict[str, set[tuple[str, str]]],
) -> str | None:
    """Fold one Qdrant point into the running section_map. Returns paper_id or None."""
    payload = point.payload or {}
    paper_id = str(payload.get("paper_id_arxiv") or "")
    if not paper_id:
        return None

    title = str(payload.get("section_title") or "")
    embed_text = str(payload.get("embed_text") or "")
    chunk_index = int(payload.get("chunk_index") or 0)
    already_written = payload.get("imrad_section_title") is not None

    title_key = _normalize_title(title)
    section_key = (paper_id, title_key)
    hl = heuristic_imrad_label(title)
    needs_text = hl is None and not already_written

    if section_key not in section_map:
        section_map[section_key] = SectionRecord(
            paper_id=paper_id,
            title=title,
            text=embed_text if needs_text else "",
            point_ids=[point.id],
            heuristic_label=hl,
            chunk_index_min=chunk_index,
            already_written=already_written,
        )
        section_min_idx[section_key] = chunk_index
        paper_sections[paper_id].add(section_key)
    else:
        record = section_map[section_key]
        record.point_ids.append(point.id)
        if not already_written:
            record.already_written = False
        if chunk_index < section_min_idx[section_key]:
            section_min_idx[section_key] = chunk_index
            record.chunk_index_min = chunk_index
            if record.heuristic_label is None and not record.already_written:
                record.text = embed_text

    return paper_id


def _extract_ready_papers(
    section_map: dict[tuple[str, str], SectionRecord],
    section_min_idx: dict[tuple[str, str], int],
    paper_sections: dict[str, set[tuple[str, str]]],
    active_paper_ids: set[str],
) -> tuple[dict[tuple[str, str], SectionRecord], dict[str, list[tuple[int, tuple[str, str]]]]]:
    """
    Extract all papers that are NOT in active_paper_ids (i.e. complete).
    Mutates section_map / section_min_idx / paper_sections in place (removes extracted papers).
    """
    done_paper_ids = set(paper_sections.keys()) - active_paper_ids
    if not done_paper_ids:
        return {}, {}

    batch_sections: dict[tuple[str, str], SectionRecord] = {}
    batch_paper_to_titles: dict[str, list[tuple[int, tuple[str, str]]]] = {}

    for pid in done_paper_ids:
        keys = paper_sections.pop(pid)
        ordered = sorted(((section_min_idx[k], k) for k in keys), key=lambda it: it[0])
        batch_paper_to_titles[pid] = ordered
        for _, key in ordered:
            batch_sections[key] = section_map.pop(key)
            section_min_idx.pop(key, None)

    return batch_sections, batch_paper_to_titles


# ---------------------------------------------------------------------------
# Classify + finalize + repair + write one batch
# ---------------------------------------------------------------------------

def _process_batch(
    batch_sections: dict[tuple[str, str], SectionRecord],
    batch_paper_to_titles: dict[str, list[tuple[int, tuple[str, str]]]],
    model,
    tokenizer,
    id2label: dict[int, str],
    device: torch.device,
    inference_batch_size: int,
    client,
    collection_name: str,
    update_batch_size: int,
    n_write_workers: int,
    connection,
    dry_run: bool,
) -> dict[str, Any]:
    """Classify, finalize, repair and (optionally) write a batch of complete papers."""
    # Classify
    unknown_keys = [
        key
        for key, section in batch_sections.items()
        if section.heuristic_label is None and section.text.strip() and not section.already_written
    ]
    if unknown_keys and model is not None:
        for start in range(0, len(unknown_keys), inference_batch_size):
            batch_keys = unknown_keys[start: start + inference_batch_size]
            texts = [f"{batch_sections[k].title} [SEP] {batch_sections[k].text}" for k in batch_keys]
            encoded = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)
                confs, preds = torch.max(probs, dim=-1)
            for key, pred_idx, conf, prob_row in zip(batch_keys, preds.tolist(), confs.tolist(), probs.tolist()):
                sec = batch_sections[key]
                sec.confidence = float(conf)
                sec.raw_probs = prob_row
                sec.final_label = id2label[int(pred_idx)]
                sec.source = "classifier"

    finalize_labels(batch_sections)

    # Free embed_text — no longer needed.
    for s in batch_sections.values():
        s.text = ""

    repaired = repair_imrad_sequences(batch_sections, batch_paper_to_titles, id2label)

    # Partial stats
    batch_stats = _partial_stats(batch_sections, batch_paper_to_titles)
    batch_stats["sequence_repair_count"] = repaired

    # Write
    updated_points = 0
    if not dry_run:
        updated_points = update_qdrant_payloads(
            client,
            collection_name=collection_name,
            sections=batch_sections,
            batch_size=update_batch_size,
            n_write_workers=n_write_workers,
            connection=connection,
        )

    batch_stats["updated_points"] = updated_points
    return batch_stats


def _partial_stats(
    sections: dict[tuple[str, str], SectionRecord],
    paper_to_titles: dict[str, list[tuple[int, tuple[str, str]]]],
) -> dict[str, Any]:
    papers_imrad_before = 0
    papers_imrad_after = 0
    for _, ordered in paper_to_titles.items():
        before_labels: list[str] = []
        after_labels: list[str] = []
        for _, key in ordered:
            sec = sections[key]
            if sec.heuristic_label and sec.heuristic_label != SKIP_LABEL:
                if not before_labels or before_labels[-1] != sec.heuristic_label:
                    before_labels.append(sec.heuristic_label)
            if sec.final_label and sec.final_label != SKIP_LABEL:
                if not after_labels or after_labels[-1] != sec.final_label:
                    after_labels.append(sec.final_label)
        if evaluate_paper(before_labels):
            papers_imrad_before += 1
        if evaluate_paper(after_labels):
            papers_imrad_after += 1

    label_counts_before = {lbl: 0 for lbl in IMRAD_ORDER}
    label_counts_after = {lbl: 0 for lbl in IMRAD_ORDER}
    for sec in sections.values():
        if sec.heuristic_label in label_counts_before:
            label_counts_before[sec.heuristic_label] += 1
        if sec.final_label in label_counts_after:
            label_counts_after[sec.final_label] += 1

    classifier_confs = [
        s.confidence
        for s in sections.values()
        if s.source in ("classifier", "sequence_repair") and s.confidence is not None
    ]

    return {
        "papers_total": len(paper_to_titles),
        "papers_respecting_imrad_before": papers_imrad_before,
        "papers_respecting_imrad_after": papers_imrad_after,
        "sections_total": len(sections),
        "sections_labelled_imrad_before": sum(1 for s in sections.values() if s.heuristic_label in IMRAD_ORDER),
        "sections_labelled_imrad_after": sum(1 for s in sections.values() if s.final_label in IMRAD_ORDER),
        "sections_skipped": sum(1 for s in sections.values() if s.final_label == SKIP_LABEL),
        "sections_no_label": sum(1 for s in sections.values() if s.final_label is None),
        "sections_already_written": sum(1 for s in sections.values() if s.already_written),
        "imrad_label_sources": {
            "heuristic": sum(1 for s in sections.values() if s.final_label in IMRAD_ORDER and s.source == "heuristic"),
            "classifier": sum(1 for s in sections.values() if s.final_label in IMRAD_ORDER and s.source == "classifier"),
            "sequence_repair": sum(1 for s in sections.values() if s.final_label in IMRAD_ORDER and s.source == "sequence_repair"),
        },
        "sections_no_label_unresolved": sum(1 for s in sections.values() if s.final_label is None and s.source == "unresolved"),
        "imrad_label_counts": {
            lbl: {"before": label_counts_before[lbl], "after": label_counts_after[lbl]}
            for lbl in IMRAD_ORDER
        },
        "classifier_confs": classifier_confs,
    }


def _merge_stats(accumulated: dict[str, Any], batch: dict[str, Any]) -> None:
    """Add batch partial stats into the accumulated totals (in-place)."""
    for key in (
        "papers_total", "papers_respecting_imrad_before", "papers_respecting_imrad_after",
        "sections_total", "sections_labelled_imrad_before", "sections_labelled_imrad_after",
        "sections_skipped", "sections_no_label", "sections_already_written",
        "sections_no_label_unresolved", "updated_points", "sequence_repair_count",
    ):
        accumulated[key] = accumulated.get(key, 0) + batch.get(key, 0)

    for src in ("heuristic", "classifier", "sequence_repair"):
        accumulated.setdefault("imrad_label_sources", {src: 0 for src in ("heuristic", "classifier", "sequence_repair")})
        accumulated["imrad_label_sources"][src] = accumulated["imrad_label_sources"].get(src, 0) + batch.get("imrad_label_sources", {}).get(src, 0)

    for lbl in IMRAD_ORDER:
        accumulated.setdefault("imrad_label_counts", {lbl: {"before": 0, "after": 0} for lbl in IMRAD_ORDER})
        accumulated["imrad_label_counts"][lbl]["before"] += batch.get("imrad_label_counts", {}).get(lbl, {}).get("before", 0)
        accumulated["imrad_label_counts"][lbl]["after"] += batch.get("imrad_label_counts", {}).get(lbl, {}).get("after", 0)

    accumulated.setdefault("classifier_confs", [])
    accumulated["classifier_confs"].extend(batch.get("classifier_confs", []))


def _finalize_accumulated_stats(accumulated: dict[str, Any]) -> dict[str, Any]:
    confs = accumulated.pop("classifier_confs", [])
    repaired = accumulated.pop("sequence_repair_count", 0)
    accumulated["classifier_confidence"] = {
        "count": len(confs),
        "mean": round(mean(confs), 4) if confs else None,
        "min": round(min(confs), 4) if confs else None,
        "max": round(max(confs), 4) if confs else None,
    }
    accumulated["sequence_repair_count"] = repaired
    return accumulated


# ---------------------------------------------------------------------------
# Streaming scroll-classify-write pipeline
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_id: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, tokenizer


def run_streaming_pipeline(
    client,
    collection_name: str,
    page_size: int,
    max_points: int | None,
    model_id: str,
    device: torch.device,
    inference_batch_size: int,
    update_batch_size: int,
    n_write_workers: int,
    connection,
    dry_run: bool,
    paper_batch_size: int,
    scroll_workers: int,
) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
    """
    Stream Qdrant → classify → write without ever holding all points in RAM.

    Memory is bounded to O(paper_batch_size × avg_chunks_per_paper).
    Returns (aggregated_stats, total_updated_points, non_imrad_sections).
    """
    model, tokenizer = _load_model_and_tokenizer(model_id, device)
    id2label: dict[int, str] = dict(model.config.id2label)

    section_map: dict[tuple[str, str], SectionRecord] = {}
    section_min_idx: dict[tuple[str, str], int] = {}
    paper_sections: dict[str, set[tuple[str, str]]] = defaultdict(set)
    accumulated: dict[str, Any] = {}
    non_imrad_accumulator: dict[str, dict[str, Any]] = {}
    total_points_seen = 0
    total_batches = 0
    t_classify = 0.0
    t_write = 0.0

    def _flush_batch(batch_sections, batch_paper_to_titles):
        nonlocal t_classify, t_write
        tc = time.monotonic()
        # Classify
        unknown_keys = [
            key
            for key, section in batch_sections.items()
            if section.heuristic_label is None and section.text.strip() and not section.already_written
        ]
        if unknown_keys:
            for start in range(0, len(unknown_keys), inference_batch_size):
                bk = unknown_keys[start: start + inference_batch_size]
                texts = [f"{batch_sections[k].title} [SEP] {batch_sections[k].text}" for k in bk]
                encoded = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                with torch.no_grad():
                    logits = model(**encoded).logits
                    probs = torch.softmax(logits, dim=-1)
                    confs, preds = torch.max(probs, dim=-1)
                for key, pred_idx, conf, prob_row in zip(bk, preds.tolist(), confs.tolist(), probs.tolist()):
                    sec = batch_sections[key]
                    sec.confidence = float(conf)
                    sec.raw_probs = prob_row
                    sec.final_label = id2label[int(pred_idx)]
                    sec.source = "classifier"
        t_classify += time.monotonic() - tc

        finalize_labels(batch_sections)
        for s in batch_sections.values():
            s.text = ""
        repair_imrad_sequences(batch_sections, batch_paper_to_titles, id2label)

        batch_stats = _partial_stats(batch_sections, batch_paper_to_titles)

        # Accumulate non-IMRaD info
        _NON_IMRAD = {SKIP_LABEL, None}
        for sec in batch_sections.values():
            if sec.final_label not in _NON_IMRAD:
                continue
            norm_title = _normalize_title(sec.title)
            if norm_title not in non_imrad_accumulator:
                non_imrad_accumulator[norm_title] = {
                    "normalized_title": norm_title,
                    "original_title": sec.title,
                    "count": 0,
                    "source_counts": defaultdict(int),
                    "papers": set(),
                    "sample_text": sec.text[:300] if sec.text else "",
                }
            non_imrad_accumulator[norm_title]["count"] += 1
            non_imrad_accumulator[norm_title]["source_counts"][sec.source] += 1
            non_imrad_accumulator[norm_title]["papers"].add(sec.paper_id)

        tw = time.monotonic()
        updated = 0
        if not dry_run:
            updated = update_qdrant_payloads(
                client,
                collection_name=collection_name,
                sections=batch_sections,
                batch_size=update_batch_size,
                n_write_workers=n_write_workers,
                connection=connection,
            )
        t_write += time.monotonic() - tw
        batch_stats["updated_points"] = updated
        return batch_stats

    # --- Single-worker synchronous scroll (memory-bounded streaming) ---
    offset = None
    stop = False

    while not stop:
        try:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=page_size,
                offset=offset,
                with_vectors=False,
                with_payload=["paper_id_arxiv", "section_title", "embed_text", "chunk_index", "imrad_section_title"],
            )
        except UnexpectedResponse as exc:
            logger.warning("Qdrant scroll error (treating as end): %s", exc)
            break

        if not points:
            break

        active_this_page: set[str] = set()
        for point in points:
            pid = _accumulate_point(point, section_map, section_min_idx, paper_sections)
            if pid:
                active_this_page.add(pid)
            total_points_seen += 1
            if max_points is not None and total_points_seen >= max_points:
                stop = True
                break

        if total_points_seen % 500_000 == 0 or (stop and total_points_seen % 100_000 == 0):
            logger.info(
                "  … scrolled %d points | %d papers buffered | %d unique sections in buffer",
                total_points_seen, len(paper_sections), len(section_map),
            )

        # Papers seen in this page may still have chunks in the next page — keep them.
        # Papers NOT seen in this page are complete — flush them when buffer is large enough.
        if stop or next_offset is None:
            # Final flush: all remaining papers are complete.
            active_this_page = set()

        ready_sections, ready_paper_to_titles = _extract_ready_papers(
            section_map, section_min_idx, paper_sections, active_this_page
        )

        if ready_sections and (len(ready_paper_to_titles) >= paper_batch_size or stop or next_offset is None):
            batch_stats = _flush_batch(ready_sections, ready_paper_to_titles)
            _merge_stats(accumulated, batch_stats)
            total_batches += 1
            logger.info(
                "Flushed batch %d: %d papers / %d sections | total scrolled %d",
                total_batches, batch_stats["papers_total"], batch_stats["sections_total"], total_points_seen,
            )
        elif ready_sections:
            # Not enough to flush yet — put them back. This happens rarely (small ready set).
            for key, sec in ready_sections.items():
                section_map[key] = sec
                section_min_idx[key] = sec.chunk_index_min
                paper_sections[sec.paper_id].add(key)

        if next_offset is None or stop:
            break
        offset = next_offset

    # Flush any remaining papers (shouldn't happen after the stop path above, but be safe).
    if paper_sections:
        all_sections, all_paper_to_titles = _extract_ready_papers(
            section_map, section_min_idx, paper_sections, set()
        )
        if all_sections:
            batch_stats = _flush_batch(all_sections, all_paper_to_titles)
            _merge_stats(accumulated, batch_stats)
            total_batches += 1
            logger.info(
                "Final flush batch %d: %d papers / %d sections",
                total_batches, batch_stats["papers_total"], batch_stats["sections_total"],
            )

    stats = _finalize_accumulated_stats(accumulated)
    stats["total_points_scrolled"] = total_points_seen
    stats["timing_classify_s"] = round(t_classify, 2)
    stats["timing_write_s"] = round(t_write, 2)

    # Build non-IMRaD list
    _PAPERS_LIST_SKIP_SOURCES = {"skip"}
    _PAPERS_LIST_CAP = 50
    non_imrad_sections = []
    for item in non_imrad_accumulator.values():
        papers_sorted = sorted(item["papers"])
        dominant_reason = max(item["source_counts"], key=lambda s: item["source_counts"][s])
        include_papers = dominant_reason not in _PAPERS_LIST_SKIP_SOURCES
        non_imrad_sections.append({
            "normalized_title": item["normalized_title"],
            "original_title": item["original_title"],
            "count": item["count"],
            "reason": dominant_reason,
            "sample_text": item["sample_text"],
            "paper_count": len(papers_sorted),
            "papers": papers_sorted[:_PAPERS_LIST_CAP] if include_papers else [],
        })
    non_imrad_sections.sort(key=lambda x: x["count"], reverse=True)

    return stats, accumulated.get("updated_points", 0), non_imrad_sections


# ---------------------------------------------------------------------------
# Legacy full-collect path (kept for reference / small runs)
# ---------------------------------------------------------------------------

async def _scroll_worker(
    async_client: AsyncQdrantClient,
    collection_name: str,
    page_size: int,
    max_points: int | None,
    worker_id: int,
    n_workers: int,
    result_queue: asyncio.Queue,
) -> None:
    total_uuid_space = 2**128
    stride = total_uuid_space // n_workers
    start_int = worker_id * stride
    end_int = (worker_id + 1) * stride if worker_id < n_workers - 1 else total_uuid_space

    def _int_to_uuid(i: int) -> str:
        h = f"{i:032x}"
        return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

    offset: ExtendedPointId | None = _int_to_uuid(start_int) if start_int > 0 else None
    end_uuid: str | None = _int_to_uuid(end_int) if worker_id < n_workers - 1 else None
    seen = 0

    while True:
        try:
            points, next_offset = await async_client.scroll(
                collection_name=collection_name,
                limit=page_size,
                offset=offset,
                with_vectors=False,
                with_payload=["paper_id_arxiv", "section_title", "embed_text", "chunk_index", "imrad_section_title"],
            )
        except Exception as exc:
            logger.warning("Worker %d scroll error (treating as end): %s", worker_id, exc)
            break

        if not points:
            break

        for point in points:
            if end_uuid is not None and str(point.id) >= end_uuid:
                return
            await result_queue.put(point)
            seen += 1
            if max_points is not None and seen >= max_points:
                return

        if next_offset is None:
            break
        if end_uuid is not None and str(next_offset) >= end_uuid:
            break
        offset = next_offset

    logger.debug("Worker %d finished after %d points", worker_id, seen)


async def _collect_sections_async(
    collection_name: str,
    page_size: int,
    max_points: int | None,
    n_workers: int,
    connection,
) -> tuple[dict[tuple[str, str], SectionRecord], dict[str, list[tuple[int, tuple[str, str]]]]]:
    async_client = AsyncQdrantClient(
        host=connection.host,
        port=connection.port,
        grpc_port=connection.grpc_port,
        prefer_grpc=connection.prefer_grpc,
        timeout=600,
    )

    result_queue: asyncio.Queue = asyncio.Queue(maxsize=n_workers * page_size * 4)
    section_map: dict[tuple[str, str], SectionRecord] = {}
    section_min_idx: dict[tuple[str, str], int] = {}
    paper_sections: dict[str, set[tuple[str, str]]] = defaultdict(set)
    total_seen = 0
    per_worker_max = (max_points + n_workers - 1) // n_workers if max_points else None

    async def consumer():
        nonlocal total_seen
        while True:
            item = await result_queue.get()
            if item is None:
                result_queue.task_done()
                break
            _accumulate_point(item, section_map, section_min_idx, paper_sections)
            total_seen += 1
            if total_seen % 500_000 == 0:
                logger.info("  … scrolled %d points so far (%d unique sections)", total_seen, len(section_map))
            result_queue.task_done()

    workers = [
        asyncio.create_task(
            _scroll_worker(async_client, collection_name, page_size, per_worker_max, i, n_workers, result_queue)
        )
        for i in range(n_workers)
    ]
    consumer_task = asyncio.create_task(consumer())
    await asyncio.gather(*workers)
    await result_queue.put(None)
    await consumer_task
    await async_client.close()

    paper_to_titles: dict[str, list[tuple[int, tuple[str, str]]]] = {
        paper_id: sorted(
            ((section_min_idx[k], k) for k in keys),
            key=lambda it: it[0],
        )
        for paper_id, keys in paper_sections.items()
    }
    return section_map, paper_to_titles


def collect_sections(
    client,
    collection_name: str,
    page_size: int,
    max_points: int | None,
    n_workers: int = 1,
    connection=None,
) -> tuple[dict[tuple[str, str], SectionRecord], dict[str, list[tuple[int, tuple[str, str]]]]]:
    if n_workers > 1 and connection is not None:
        return asyncio.run(
            _collect_sections_async(collection_name, page_size, max_points, n_workers, connection)
        )

    section_map: dict[tuple[str, str], SectionRecord] = {}
    section_min_idx: dict[tuple[str, str], int] = {}
    paper_sections: dict[str, set[tuple[str, str]]] = defaultdict(set)

    offset = None
    seen = 0
    while True:
        try:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=page_size,
                offset=offset,
                with_vectors=False,
                with_payload=["paper_id_arxiv", "section_title", "embed_text", "chunk_index", "imrad_section_title"],
            )
        except UnexpectedResponse as exc:
            logger.warning("Qdrant scroll error (treating as end): %s", exc)
            break

        if not points:
            break

        for point in points:
            _accumulate_point(point, section_map, section_min_idx, paper_sections)
            seen += 1
            if max_points is not None and seen >= max_points:
                break

        if (max_points is not None and seen >= max_points) or offset is None:
            break

    paper_to_titles: dict[str, list[tuple[int, tuple[str, str]]]] = {
        paper_id: sorted(
            ((section_min_idx[k], k) for k in keys),
            key=lambda it: it[0],
        )
        for paper_id, keys in paper_sections.items()
    }
    return section_map, paper_to_titles


def classify_unknown_sections(
    sections: dict[tuple[str, str], SectionRecord],
    model_id: str,
    device: torch.device,
    batch_size: int,
) -> dict[int, str]:
    unknown_keys = [
        key
        for key, section in sections.items()
        if section.heuristic_label is None and section.text.strip() and not section.already_written
    ]
    if not unknown_keys:
        logger.info("No unknown section titles to classify")
        return {}

    logger.info(
        "Classifying %d unresolved section titles with %s (heuristic saved %d model calls)",
        len(unknown_keys),
        model_id,
        sum(1 for s in sections.values() if s.heuristic_label is not None),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()

    for start in range(0, len(unknown_keys), batch_size):
        batch_keys = unknown_keys[start: start + batch_size]
        texts = [f"{sections[key].title} [SEP] {sections[key].text}" for key in batch_keys]
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

        for key, pred_idx, conf, prob_row in zip(
            batch_keys, preds.tolist(), confs.tolist(), probs.tolist()
        ):
            section = sections[key]
            section.confidence = float(conf)
            section.raw_probs = prob_row
            section.final_label = model.config.id2label[int(pred_idx)]
            section.source = "classifier"

    return dict(model.config.id2label)


def finalize_labels(sections: dict[tuple[str, str], SectionRecord]) -> None:
    for section in sections.values():
        if section.heuristic_label == SKIP_LABEL:
            section.final_label = SKIP_LABEL
            section.source = "skip"
            section.confidence = None
        elif section.heuristic_label is not None:
            section.final_label = section.heuristic_label
            section.source = "heuristic"
            section.confidence = 1.0
        elif section.source == "classifier":
            pass
        else:
            section.final_label = None
            section.source = "unresolved"
            section.confidence = None


def repair_imrad_sequences(
    sections: dict[tuple[str, str], SectionRecord],
    paper_to_titles: dict[str, list[tuple[int, tuple[str, str]]]],
    id2label: dict[int, str],
) -> int:
    relabelled = 0

    for _, ordered in paper_to_titles.items():
        last_order = -1
        for _, key in ordered:
            section = sections[key]
            label = section.final_label
            if label is None or label not in IMRAD_ORDER:
                continue
            current_order = IMRAD_ORDER[label]
            if current_order > last_order:
                last_order = current_order
                continue
            if section.source != "classifier" or section.raw_probs is None:
                continue
            candidates = sorted(
                (
                    (section.raw_probs[idx], lbl)
                    for idx, lbl in id2label.items()
                    if lbl in IMRAD_ORDER and IMRAD_ORDER[lbl] > last_order
                ),
                reverse=True,
            )
            if candidates:
                best_prob, best_label = candidates[0]
                section.final_label = best_label
                section.confidence = best_prob
                section.source = "sequence_repair"
                last_order = IMRAD_ORDER[best_label]
                relabelled += 1

    return relabelled


def build_stats(
    sections: dict[tuple[str, str], SectionRecord],
    paper_to_titles: dict[str, list[tuple[int, tuple[str, str]]]],
) -> dict[str, Any]:
    papers_imrad_before = 0
    papers_imrad_after = 0

    for _, ordered in paper_to_titles.items():
        before_labels: list[str] = []
        after_labels: list[str] = []

        for _, key in ordered:
            sec = sections[key]
            if sec.heuristic_label and sec.heuristic_label != SKIP_LABEL:
                if not before_labels or before_labels[-1] != sec.heuristic_label:
                    before_labels.append(sec.heuristic_label)
            if sec.final_label and sec.final_label != SKIP_LABEL:
                if not after_labels or after_labels[-1] != sec.final_label:
                    after_labels.append(sec.final_label)

        if evaluate_paper(before_labels):
            papers_imrad_before += 1
        if evaluate_paper(after_labels):
            papers_imrad_after += 1

    n_imrad_before = sum(1 for s in sections.values() if s.heuristic_label in IMRAD_ORDER)
    n_imrad_after = sum(1 for s in sections.values() if s.final_label in IMRAD_ORDER)
    n_skipped = sum(1 for s in sections.values() if s.final_label == SKIP_LABEL)
    n_no_label = sum(1 for s in sections.values() if s.final_label is None)

    n_from_heuristic = sum(1 for s in sections.values() if s.final_label in IMRAD_ORDER and s.source == "heuristic")
    n_from_model = sum(1 for s in sections.values() if s.final_label in IMRAD_ORDER and s.source == "classifier")
    n_from_repair = sum(1 for s in sections.values() if s.final_label in IMRAD_ORDER and s.source == "sequence_repair")
    n_no_label_unresolved = sum(1 for s in sections.values() if s.final_label is None and s.source == "unresolved")

    label_counts_before = {lbl: 0 for lbl in IMRAD_ORDER}
    label_counts_after = {lbl: 0 for lbl in IMRAD_ORDER}
    for sec in sections.values():
        if sec.heuristic_label in label_counts_before:
            label_counts_before[sec.heuristic_label] += 1
        if sec.final_label in label_counts_after:
            label_counts_after[sec.final_label] += 1

    classifier_confs = [
        section.confidence
        for section in sections.values()
        if section.source in ("classifier", "sequence_repair") and section.confidence is not None
    ]

    return {
        "papers_total": len(paper_to_titles),
        "papers_respecting_imrad_before": papers_imrad_before,
        "papers_respecting_imrad_after": papers_imrad_after,
        "sections_total": len(sections),
        "sections_labelled_imrad_before": n_imrad_before,
        "sections_labelled_imrad_after": n_imrad_after,
        "sections_skipped": n_skipped,
        "sections_no_label": n_no_label,
        "imrad_label_sources": {
            "heuristic": n_from_heuristic,
            "classifier": n_from_model,
            "sequence_repair": n_from_repair,
        },
        "sections_no_label_unresolved": n_no_label_unresolved,
        "imrad_label_counts": {
            lbl: {"before": label_counts_before[lbl], "after": label_counts_after[lbl]}
            for lbl in IMRAD_ORDER
        },
        "classifier_confidence": {
            "count": len(classifier_confs),
            "mean": round(mean(classifier_confs), 4) if classifier_confs else None,
            "min": round(min(classifier_confs), 4) if classifier_confs else None,
            "max": round(max(classifier_confs), 4) if classifier_confs else None,
        },
    }


# ---------------------------------------------------------------------------
# Async parallel Qdrant writes
# ---------------------------------------------------------------------------

async def _update_qdrant_payloads_async(
    connection,
    collection_name: str,
    sections: dict[tuple[str, str], SectionRecord],
    batch_size: int,
    n_workers: int,
) -> int:
    async_client = AsyncQdrantClient(
        host=connection.host,
        port=connection.port,
        grpc_port=connection.grpc_port,
        prefer_grpc=connection.prefer_grpc,
        timeout=600,
    )

    work: list[tuple[dict, list]] = []
    for section in sections.values():
        if not section.point_ids or section.already_written:
            continue
        payload = {
            "imrad_section_title": section.final_label,
            "imrad_label_source": section.source,
            "imrad_classifier_confidence": section.confidence,
            "imrad_section_title_norm": _normalize_title(section.title),
        }
        for start in range(0, len(section.point_ids), batch_size):
            work.append((payload, section.point_ids[start: start + batch_size]))

    semaphore = asyncio.Semaphore(n_workers)
    total_updates = 0
    lock = asyncio.Lock()

    async def _do_update(payload: dict, point_ids: list) -> None:
        nonlocal total_updates
        async with semaphore:
            for attempt in range(5):
                try:
                    await async_client.set_payload(
                        collection_name=collection_name,
                        payload=payload,
                        points=point_ids,
                        wait=True,
                    )
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    await asyncio.sleep(2 ** attempt)
            async with lock:
                total_updates += len(point_ids)

    await asyncio.gather(*(_do_update(p, ids) for p, ids in work))
    await async_client.close()
    return total_updates


def update_qdrant_payloads(
    client,
    collection_name: str,
    sections: dict[tuple[str, str], SectionRecord],
    batch_size: int,
    n_write_workers: int = 1,
    connection=None,
) -> int:
    if n_write_workers > 1 and connection is not None:
        return asyncio.run(
            _update_qdrant_payloads_async(connection, collection_name, sections, batch_size, n_write_workers)
        )

    updates = 0
    for section in sections.values():
        if not section.point_ids or section.already_written:
            continue
        payload = {
            "imrad_section_title": section.final_label,
            "imrad_label_source": section.source,
            "imrad_classifier_confidence": section.confidence,
            "imrad_section_title_norm": _normalize_title(section.title),
        }
        point_ids = section.point_ids
        for start in range(0, len(point_ids), batch_size):
            batch_ids = point_ids[start: start + batch_size]
            client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=batch_ids,
                wait=True,
            )
            updates += len(batch_ids)
    return updates


_NON_IMRAD_FINAL_LABELS = {SKIP_LABEL, None}
_PAPERS_LIST_SKIP_SOURCES = {"skip"}
_PAPERS_LIST_CAP = 50


def collect_non_imrad_sections(sections: dict[tuple[str, str], SectionRecord]) -> list[dict[str, Any]]:
    non_imrad_by_title: dict[str, dict[str, Any]] = {}

    for section in sections.values():
        if section.final_label not in _NON_IMRAD_FINAL_LABELS:
            continue
        norm_title = _normalize_title(section.title)
        if norm_title not in non_imrad_by_title:
            non_imrad_by_title[norm_title] = {
                "normalized_title": norm_title,
                "original_title": section.title,
                "count": 0,
                "source_counts": defaultdict(int),
                "papers": set(),
                "sample_text": section.text[:300] if section.text else "",
            }
        non_imrad_by_title[norm_title]["count"] += 1
        non_imrad_by_title[norm_title]["source_counts"][section.source] += 1
        non_imrad_by_title[norm_title]["papers"].add(section.paper_id)

    result = []
    for item in non_imrad_by_title.values():
        papers_sorted = sorted(item["papers"])
        dominant_reason = max(item["source_counts"], key=lambda s: item["source_counts"][s])
        include_papers = dominant_reason not in _PAPERS_LIST_SKIP_SOURCES
        result.append({
            "normalized_title": item["normalized_title"],
            "original_title": item["original_title"],
            "count": item["count"],
            "reason": dominant_reason,
            "sample_text": item["sample_text"],
            "paper_count": len(papers_sorted),
            "papers": papers_sorted[:_PAPERS_LIST_CAP] if include_papers else [],
        })

    return sorted(result, key=lambda x: x["count"], reverse=True)


def write_non_imrad_report(output_path: Path, non_imrad: list[dict[str, Any]]) -> None:
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_unique_non_imrad": len(non_imrad),
        "top_50": non_imrad[:50],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_report(output_path: Path, stats: dict[str, Any], sections: dict[tuple[str, str], SectionRecord] | None = None, timings: dict[str, float] | None = None) -> None:
    low_conf: list[dict] = []
    if sections is not None:
        low_conf = sorted(
            (
                {
                    "paper_id_arxiv": section.paper_id,
                    "section_title": section.title,
                    "predicted_label": section.final_label,
                    "confidence": section.confidence,
                }
                for section in sections.values()
                if section.source == "classifier" and section.confidence is not None
            ),
            key=lambda row: row["confidence"] or 0.0,
        )[:50]

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "lowest_confidence_predictions": low_conf,
    }
    if timings:
        report["timings_seconds"] = {k: round(v, 2) for k, v in timings.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit IMRaD section-title compliance and classify unresolved sections in Qdrant.",
    )
    parser.add_argument("--profile", choices=["local", "hpc"], default=QDRANT_ACTIVE.profile)
    parser.add_argument("--collection-name", default=QDRANT_ACTIVE.collection_name)
    parser.add_argument("--model-id", default="lostelf/section-classifier-imrad")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--scroll-page-size", type=int, default=2048)
    parser.add_argument("--scroll-workers", type=int, default=8,
                        help="Concurrent async scroll workers. Only used in legacy --no-streaming mode.")
    parser.add_argument("--write-workers", type=int, default=8,
                        help="Concurrent async Qdrant set_payload calls.")
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument("--qdrant-update-batch-size", type=int, default=512)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--paper-batch-size", type=int, default=50_000,
        help=(
            "Streaming mode: flush classify+write after this many complete papers. "
            "Controls peak RAM usage. Lower = less RAM, more Qdrant round-trips. "
            "Default 50 000 keeps peak well under 10 GB even for large collections."
        ),
    )
    parser.add_argument(
        "--no-streaming", action="store_true",
        help="Disable streaming pipeline and use the legacy full-collect approach (high RAM).",
    )
    parser.add_argument(
        "--report-path",
        default=str(PATHS.progress / "imrad_postprocessing_report.json"),
        help="Where to write the main JSON report.",
    )
    parser.add_argument(
        "--non-imrad-report-path",
        default=None,
        help=(
            "Where to write the non-IMRaD sections JSON report. "
            "Defaults to <report-path-stem>_non_imrad.json next to --report-path."
        ),
    )
    return parser


def main() -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ensure_qdrant_runtime(args.profile)
    client = qdrant_client(timeout=600)

    from config.settings import QDRANT_CONNECTION
    connection = QDRANT_CONNECTION

    report_path = Path(args.report_path)
    if args.non_imrad_report_path:
        non_imrad_report_path = Path(args.non_imrad_report_path)
    else:
        non_imrad_report_path = report_path.with_name(report_path.stem + "_non_imrad.json")

    timings: dict[str, float] = {}

    # ── Streaming pipeline (default) ──────────────────────────────────────────
    if not args.no_streaming:
        logger.info(
            "Streaming pipeline: collection=%s  page_size=%d  paper_batch=%d  write_workers=%d",
            args.collection_name, args.scroll_page_size, args.paper_batch_size, args.write_workers,
        )
        device = _resolve_device(args.device)
        t0 = time.monotonic()
        stats, updated_points, non_imrad_sections = run_streaming_pipeline(
            client=client,
            collection_name=args.collection_name,
            page_size=args.scroll_page_size,
            max_points=args.max_points,
            model_id=args.model_id,
            device=device,
            inference_batch_size=args.inference_batch_size,
            update_batch_size=args.qdrant_update_batch_size,
            n_write_workers=args.write_workers,
            connection=connection,
            dry_run=args.dry_run,
            paper_batch_size=args.paper_batch_size,
            scroll_workers=args.scroll_workers,
        )
        timings["total"] = time.monotonic() - t0
        timings["classify"] = stats.pop("timing_classify_s", 0.0)
        timings["write"] = stats.pop("timing_write_s", 0.0)
        timings["scroll"] = timings["total"] - timings["classify"] - timings["write"]

        write_report(report_path, stats, sections=None, timings=timings)
        write_non_imrad_report(non_imrad_report_path, non_imrad_sections)

        if args.max_points:
            total_points = 69_022_454
            scale = total_points / args.max_points
            logger.info(
                "EXTRAPOLATION (scale=%.0fx): total=%.0fs (%.1fh)",
                scale, timings["total"] * scale, timings["total"] * scale / 3600,
            )

        logger.info("IMRaD postprocessing complete (streaming)")
        logger.info(
            "Papers respecting IMRaD (before -> after): %d -> %d",
            stats.get("papers_respecting_imrad_before", 0), stats.get("papers_respecting_imrad_after", 0),
        )
        logger.info("Updated points in Qdrant: %d", updated_points)
        logger.info("Timings: %s", {k: f"{v:.1f}s" for k, v in timings.items()})
        logger.info("Report written to %s", report_path)
        logger.info("Non-IMRaD report written to %s", non_imrad_report_path)
        return {"stats": stats, "updated_points": updated_points, "report_path": str(report_path), "timings": timings}

    # ── Legacy full-collect path ───────────────────────────────────────────────
    logger.info(
        "Legacy mode: collecting all points from %s (scroll_workers=%d, page_size=%d)",
        args.collection_name, args.scroll_workers, args.scroll_page_size,
    )
    t0 = time.monotonic()
    sections, paper_to_titles = collect_sections(
        client,
        collection_name=args.collection_name,
        page_size=args.scroll_page_size,
        max_points=args.max_points,
        n_workers=args.scroll_workers,
        connection=connection,
    )
    timings["scroll"] = time.monotonic() - t0
    logger.info(
        "Collected %d unique section titles across %d papers in %.1fs",
        len(sections), len(paper_to_titles), timings["scroll"],
    )

    n_already_written = sum(1 for s in sections.values() if s.already_written)
    n_heuristic = sum(1 for s in sections.values() if s.heuristic_label is not None and not s.already_written)
    n_needs_model = sum(1 for s in sections.values() if s.heuristic_label is None and s.text.strip() and not s.already_written)
    logger.info(
        "Checkpoint resume: %d already written | %d heuristic | %d needs model | %d empty text",
        n_already_written, n_heuristic, n_needs_model,
        sum(1 for s in sections.values() if s.heuristic_label is None and not s.text.strip() and not s.already_written),
    )

    device = _resolve_device(args.device)
    t1 = time.monotonic()
    id2label = classify_unknown_sections(sections, model_id=args.model_id, device=device, batch_size=args.inference_batch_size)
    timings["classify"] = time.monotonic() - t1
    logger.info("Classification done in %.1fs", timings["classify"])

    finalize_labels(sections)
    for s in sections.values():
        s.text = ""
    repaired = repair_imrad_sequences(sections, paper_to_titles, id2label)
    logger.info("Sequence repair: %d sections relabelled", repaired)

    stats = build_stats(sections, paper_to_titles)
    updated_points = 0
    timings["write"] = 0.0
    if not args.dry_run:
        logger.info("Writing payloads to Qdrant (write_workers=%d, update_batch=%d)", args.write_workers, args.qdrant_update_batch_size)
        t2 = time.monotonic()
        updated_points = update_qdrant_payloads(
            client,
            collection_name=args.collection_name,
            sections=sections,
            batch_size=args.qdrant_update_batch_size,
            n_write_workers=args.write_workers,
            connection=connection,
        )
        timings["write"] = time.monotonic() - t2
        logger.info("Wrote %d points in %.1fs", updated_points, timings["write"])

    timings["total"] = sum(timings.values())

    if args.max_points:
        total_points = 69_022_454
        scale = total_points / args.max_points
        logger.info(
            "EXTRAPOLATION (scale=%.0fx): scroll=%.0fs  classify=%.0fs  write=%.0fs  total=%.0fs  (%.1fh)",
            scale, timings["scroll"] * scale, timings["classify"] * scale,
            timings["write"] * scale, timings["total"] * scale, timings["total"] * scale / 3600,
        )

    write_report(report_path, stats, sections, timings=timings)
    non_imrad_sections = collect_non_imrad_sections(sections)
    write_non_imrad_report(non_imrad_report_path, non_imrad_sections)

    logger.info("IMRaD postprocessing complete (legacy)")
    logger.info(
        "Papers respecting IMRaD (before -> after): %d -> %d",
        stats["papers_respecting_imrad_before"], stats["papers_respecting_imrad_after"],
    )
    logger.info("Classifier confidence: %s", stats["classifier_confidence"])
    src = stats["imrad_label_sources"]
    logger.info(
        "Sections total:%d  labelled_imrad(before->after):%d->%d  skipped:%d  no_label:%d",
        stats["sections_total"], stats["sections_labelled_imrad_before"],
        stats["sections_labelled_imrad_after"], stats["sections_skipped"], stats["sections_no_label"],
    )
    logger.info(
        "IMRaD sources — heuristic:%d  classifier:%d  sequence_repair:%d",
        src["heuristic"], src["classifier"], src["sequence_repair"],
    )
    logger.info("No-label (no body text): %d", stats["sections_no_label_unresolved"])
    logger.info("Non-IMRaD unique sections: %d", len(non_imrad_sections))
    logger.info("Updated points in Qdrant: %d", updated_points)
    logger.info("Timings: %s", {k: f"{v:.1f}s" for k, v in timings.items()})
    logger.info("Report written to %s", report_path)
    logger.info("Non-IMRaD report written to %s", non_imrad_report_path)

    return {
        "stats": stats,
        "updated_points": updated_points,
        "report_path": str(report_path),
        "non_imrad_report_path": str(non_imrad_report_path),
        "timings": timings,
    }


if __name__ == "__main__":
    main()
