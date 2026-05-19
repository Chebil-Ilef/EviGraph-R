from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import PATHS, QDRANT_ACTIVE
from qdrant_client.http.exceptions import UnexpectedResponse
from utils.qdrant import ensure_qdrant_runtime, qdrant_client

from indexing.postprocessing.imrad_titles import (
    IMRAD_ORDER,
    SKIP_LABEL,
    _normalize_title,
    evaluate_paper,
    heuristic_imrad_label,
)

logger = logging.getLogger(__name__)


@dataclass
class ExistingSection:
    paper_id: str
    title: str
    chunk_index_min: int
    point_count: int = 0
    imrad_label: str | None = None
    source: str | None = None
    sample_text: str = ""


def _extract_ready_papers(
    section_map: dict[tuple[str, str], ExistingSection],
    section_min_idx: dict[tuple[str, str], int],
    paper_sections: dict[str, set[tuple[str, str]]],
    active_paper_ids: set[str],
) -> tuple[dict[tuple[str, str], ExistingSection], dict[str, list[tuple[int, tuple[str, str]]]]]:
    done_paper_ids = set(paper_sections.keys()) - active_paper_ids
    if not done_paper_ids:
        return {}, {}
    batch_sections: dict[tuple[str, str], ExistingSection] = {}
    batch_paper_to_titles: dict[str, list[tuple[int, tuple[str, str]]]] = {}
    for pid in done_paper_ids:
        keys = paper_sections.pop(pid)
        ordered = sorted(((section_min_idx[k], k) for k in keys), key=lambda it: it[0])
        batch_paper_to_titles[pid] = ordered
        for _, key in ordered:
            batch_sections[key] = section_map.pop(key)
            section_min_idx.pop(key, None)
    return batch_sections, batch_paper_to_titles


def _label_bucket(label: str | None) -> str:
    if label is None:
        return "<missing>"
    if label in IMRAD_ORDER:
        return label
    return f"<non_imrad:{label}>"


def _accumulate_point(
    point,
    section_map: dict[tuple[str, str], ExistingSection],
    section_min_idx: dict[tuple[str, str], int],
    paper_sections: dict[str, set[tuple[str, str]]],
    chunk_label_counts: Counter[str],
    chunk_title_heuristic_counts: Counter[str],
    source_chunk_counts: Counter[str],
    sample_chars: int,
) -> str | None:
    payload = point.payload or {}
    paper_id = str(payload.get("paper_id_arxiv") or "")
    if not paper_id:
        return None

    title = str(payload.get("section_title") or "")
    title_key = _normalize_title(title)
    section_key = (paper_id, title_key)
    chunk_index = int(payload.get("chunk_index") or 0)
    label = payload.get("imrad_section_title")
    label = str(label) if label is not None else None
    source = payload.get("imrad_label_source")
    source = str(source) if source is not None else None
    sample_text = str(payload.get("embed_text") or "")[:sample_chars] if sample_chars > 0 else ""

    chunk_label_counts[_label_bucket(label)] += 1
    chunk_title_heuristic_counts[_label_bucket(heuristic_imrad_label(title))] += 1
    source_chunk_counts[source or "<missing>"] += 1

    if section_key not in section_map:
        section_map[section_key] = ExistingSection(
            paper_id=paper_id,
            title=title,
            chunk_index_min=chunk_index,
            point_count=1,
            imrad_label=label,
            source=source,
            sample_text=sample_text,
        )
        section_min_idx[section_key] = chunk_index
        paper_sections[paper_id].add(section_key)
        return paper_id

    section = section_map[section_key]
    section.point_count += 1
    if section.imrad_label is None and label is not None:
        section.imrad_label = label
    if section.source is None and source is not None:
        section.source = source
    if not section.sample_text and sample_text:
        section.sample_text = sample_text
    if chunk_index < section_min_idx[section_key]:
        section_min_idx[section_key] = chunk_index
        section.chunk_index_min = chunk_index
    return paper_id


def _summarize_batch(
    batch_sections: dict[tuple[str, str], ExistingSection],
    batch_paper_to_titles: dict[str, list[tuple[int, tuple[str, str]]]],
    sample_per_title: int,
) -> dict[str, Any]:
    section_label_counts: Counter[str] = Counter()
    section_title_heuristic_counts: Counter[str] = Counter()
    source_section_counts: Counter[str] = Counter()
    non_imrad_title_counts: Counter[str] = Counter()
    imrad_title_counts: Counter[str] = Counter()
    non_imrad_title_details: dict[str, dict[str, Any]] = {}
    papers_with_any_imrad = 0
    papers_with_core_imrad = 0
    papers_respecting_imrad = 0

    for section in batch_sections.values():
        bucket = _label_bucket(section.imrad_label)
        section_label_counts[bucket] += 1
        section_title_heuristic_counts[_label_bucket(heuristic_imrad_label(section.title))] += 1
        source_section_counts[section.source or "<missing>"] += 1
        if section.imrad_label in IMRAD_ORDER:
            imrad_title_counts[section.title or "<empty>"] += 1
        else:
            title = section.title or "<empty>"
            norm_title = _normalize_title(title)
            non_imrad_title_counts[norm_title] += 1
            if norm_title not in non_imrad_title_details:
                non_imrad_title_details[norm_title] = {
                    "normalized_title": norm_title,
                    "original_titles": Counter(),
                    "label_counts": Counter(),
                    "source_counts": Counter(),
                    "sample_texts": [],
                }
            detail = non_imrad_title_details[norm_title]
            detail["original_titles"][title] += 1
            detail["label_counts"][_label_bucket(section.imrad_label)] += 1
            detail["source_counts"][section.source or "<missing>"] += 1
            if section.sample_text and len(detail["sample_texts"]) < sample_per_title:
                detail["sample_texts"].append(section.sample_text)

    required = {"Introduction", "Methods", "Results"}
    for _, ordered in batch_paper_to_titles.items():
        labels: list[str] = []
        for _, key in ordered:
            label = batch_sections[key].imrad_label
            if label and label in IMRAD_ORDER:
                if not labels or labels[-1] != label:
                    labels.append(label)
        if labels:
            papers_with_any_imrad += 1
        if required.issubset(set(labels)):
            papers_with_core_imrad += 1
        if evaluate_paper(labels):
            papers_respecting_imrad += 1

    return {
        "papers_total": len(batch_paper_to_titles),
        "papers_with_any_imrad": papers_with_any_imrad,
        "papers_with_core_imrad": papers_with_core_imrad,
        "papers_respecting_imrad": papers_respecting_imrad,
        "sections_total": len(batch_sections),
        "section_label_counts": section_label_counts,
        "section_title_heuristic_counts": section_title_heuristic_counts,
        "source_section_counts": source_section_counts,
        "top_imrad_section_titles": imrad_title_counts,
        "top_non_imrad_section_titles": non_imrad_title_counts,
        "top_non_imrad_section_title_details": non_imrad_title_details,
    }


def _merge_counter(dst: Counter[str], src: Counter[str]) -> None:
    for key, value in src.items():
        dst[key] += value


def _merge_stats(accumulated: dict[str, Any], batch: dict[str, Any]) -> None:
    for key in (
        "papers_total",
        "papers_with_any_imrad",
        "papers_with_core_imrad",
        "papers_respecting_imrad",
        "sections_total",
    ):
        accumulated[key] = accumulated.get(key, 0) + batch.get(key, 0)

    for key in (
        "section_label_counts",
        "section_title_heuristic_counts",
        "source_section_counts",
        "top_imrad_section_titles",
        "top_non_imrad_section_titles",
    ):
        accumulated.setdefault(key, Counter())
        _merge_counter(accumulated[key], batch[key])

    accumulated.setdefault("top_non_imrad_section_title_details", {})
    for title, detail in batch["top_non_imrad_section_title_details"].items():
        if title not in accumulated["top_non_imrad_section_title_details"]:
            accumulated["top_non_imrad_section_title_details"][title] = {
                "normalized_title": title,
                "original_titles": Counter(),
                "label_counts": Counter(),
                "source_counts": Counter(),
                "sample_texts": [],
            }
        acc_detail = accumulated["top_non_imrad_section_title_details"][title]
        _merge_counter(acc_detail["original_titles"], detail["original_titles"])
        _merge_counter(acc_detail["label_counts"], detail["label_counts"])
        _merge_counter(acc_detail["source_counts"], detail["source_counts"])
        for sample in detail["sample_texts"]:
            if sample not in acc_detail["sample_texts"]:
                acc_detail["sample_texts"].append(sample)


def _top(counter: Counter[str], n: int) -> list[dict[str, Any]]:
    return [{"value": key, "count": count} for key, count in counter.most_common(n)]


def _top_non_imrad_details(
    counter: Counter[str],
    details: dict[str, dict[str, Any]],
    *,
    top_n: int,
    sample_per_title: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for title, count in counter.most_common(top_n):
        detail = details.get(title, {})
        rows.append({
            "normalized_title": title,
            "count": count,
            "original_titles": _top(detail.get("original_titles", Counter()), 5),
            "label_counts": dict(detail.get("label_counts", Counter())),
            "source_counts": dict(detail.get("source_counts", Counter())),
            "sample_texts": list(detail.get("sample_texts", []))[:sample_per_title],
        })
    return rows


def collect_imrad_stats(
    *,
    collection_name: str,
    page_size: int,
    max_points: int | None,
    paper_batch_size: int,
    top_n: int,
    include_samples: bool,
    sample_chars: int,
    sample_per_title: int,
) -> dict[str, Any]:
    client = qdrant_client(timeout=600)
    section_map: dict[tuple[str, str], ExistingSection] = {}
    section_min_idx: dict[tuple[str, str], int] = {}
    paper_sections: dict[str, set[tuple[str, str]]] = defaultdict(set)
    accumulated: dict[str, Any] = {}
    chunk_label_counts: Counter[str] = Counter()
    chunk_title_heuristic_counts: Counter[str] = Counter()
    source_chunk_counts: Counter[str] = Counter()
    total_points_seen = 0
    total_batches = 0
    offset = None
    stop = False
    started = time.monotonic()

    while not stop:
        try:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=page_size,
                offset=offset,
                with_vectors=False,
                with_payload=[
                    "paper_id_arxiv",
                    "section_title",
                    "chunk_index",
                    "imrad_section_title",
                    "imrad_label_source",
                ] + (["embed_text"] if include_samples else []),
            )
        except UnexpectedResponse as exc:
            logger.warning("Qdrant scroll error (treating as end): %s", exc)
            break

        if not points:
            break

        active_this_page: set[str] = set()
        for point in points:
            pid = _accumulate_point(
                point,
                section_map,
                section_min_idx,
                paper_sections,
                chunk_label_counts,
                chunk_title_heuristic_counts,
                source_chunk_counts,
                sample_chars if include_samples else 0,
            )
            if pid:
                active_this_page.add(pid)
            total_points_seen += 1
            if max_points is not None and total_points_seen >= max_points:
                stop = True
                break

        if total_points_seen % 500_000 == 0:
            logger.info(
                "Scrolled %d points | %d papers buffered | %d sections buffered",
                total_points_seen,
                len(paper_sections),
                len(section_map),
            )

        if stop or next_offset is None:
            active_this_page = set()

        ready_sections, ready_paper_to_titles = _extract_ready_papers(
            section_map,
            section_min_idx,
            paper_sections,
            active_this_page,
        )
        if ready_sections and (len(ready_paper_to_titles) >= paper_batch_size or stop or next_offset is None):
            batch_stats = _summarize_batch(ready_sections, ready_paper_to_titles, sample_per_title)
            _merge_stats(accumulated, batch_stats)
            total_batches += 1
            logger.info(
                "Flushed batch %d: %d papers / %d sections | total scrolled %d",
                total_batches,
                batch_stats["papers_total"],
                batch_stats["sections_total"],
                total_points_seen,
            )
        elif ready_sections:
            for key, section in ready_sections.items():
                section_map[key] = section
                section_min_idx[key] = section.chunk_index_min
                paper_sections[section.paper_id].add(key)

        if next_offset is None or stop:
            break
        offset = next_offset

    if paper_sections:
        ready_sections, ready_paper_to_titles = _extract_ready_papers(
            section_map,
            section_min_idx,
            paper_sections,
            set(),
        )
        if ready_sections:
            batch_stats = _summarize_batch(ready_sections, ready_paper_to_titles, sample_per_title)
            _merge_stats(accumulated, batch_stats)
            total_batches += 1
            logger.info(
                "Final flush batch %d: %d papers / %d sections",
                total_batches,
                batch_stats["papers_total"],
                batch_stats["sections_total"],
            )

    section_label_counts = accumulated.pop("section_label_counts", Counter())
    section_title_heuristic_counts = accumulated.pop("section_title_heuristic_counts", Counter())
    source_section_counts = accumulated.pop("source_section_counts", Counter())
    top_imrad_titles = accumulated.pop("top_imrad_section_titles", Counter())
    top_non_imrad_titles = accumulated.pop("top_non_imrad_section_titles", Counter())
    top_non_imrad_details = accumulated.pop("top_non_imrad_section_title_details", {})

    imrad_sections = sum(section_label_counts.get(label, 0) for label in IMRAD_ORDER)
    heuristic_imrad_sections = sum(section_title_heuristic_counts.get(label, 0) for label in IMRAD_ORDER)
    missing_sections = section_label_counts.get("<missing>", 0)
    skipped_by_title = section_title_heuristic_counts.get(f"<non_imrad:{SKIP_LABEL}>", 0)
    total_sections = accumulated.get("sections_total", 0)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "collection_name": collection_name,
        "max_points": max_points,
        "total_points_scrolled": total_points_seen,
        "total_batches": total_batches,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "stats": {
            **accumulated,
            "sections_with_imrad_label": imrad_sections,
            "sections_missing_imrad_label": missing_sections,
            "sections_whose_raw_title_matches_imrad_heuristic": heuristic_imrad_sections,
            "sections_whose_raw_title_is_skip_heuristic": skipped_by_title,
            "section_imrad_coverage": round(imrad_sections / total_sections, 6) if total_sections else None,
            "chunk_label_counts": dict(chunk_label_counts),
            "chunk_title_heuristic_counts": dict(chunk_title_heuristic_counts),
            "section_label_counts": dict(section_label_counts),
            "section_title_heuristic_counts": dict(section_title_heuristic_counts),
            "source_chunk_counts": dict(source_chunk_counts),
            "source_section_counts": dict(source_section_counts),
        },
        "top_imrad_section_titles": _top(top_imrad_titles, top_n),
        "top_non_imrad_section_titles": _top_non_imrad_details(
            top_non_imrad_titles,
            top_non_imrad_details,
            top_n=top_n,
            sample_per_title=sample_per_title if include_samples else 0,
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only IMRaD payload statistics for Qdrant.")
    parser.add_argument("--profile", default="hpc")
    parser.add_argument("--collection-name", default=QDRANT_ACTIVE.collection_name)
    parser.add_argument("--scroll-page-size", type=int, default=4096)
    parser.add_argument("--paper-batch-size", type=int, default=50_000)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Fetch embed_text and include small examples for top non-IMRaD titles.",
    )
    parser.add_argument("--sample-chars", type=int, default=300)
    parser.add_argument("--sample-per-title", type=int, default=3)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=PATHS.progress / "imrad_current_stats.json",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ensure_qdrant_runtime(args.profile)
    report = collect_imrad_stats(
        collection_name=args.collection_name,
        page_size=args.scroll_page_size,
        max_points=args.max_points,
        paper_batch_size=args.paper_batch_size,
        top_n=args.top_n,
        include_samples=args.include_samples,
        sample_chars=args.sample_chars,
        sample_per_title=args.sample_per_title,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    stats = report["stats"]
    logger.info(
        "IMRaD section labels: %d/%d (coverage %.4f)",
        stats["sections_with_imrad_label"],
        stats["sections_total"],
        stats["section_imrad_coverage"] or 0.0,
    )
    logger.info(
        "Papers with any/core/ordered IMRaD: %d / %d / %d",
        stats["papers_with_any_imrad"],
        stats["papers_with_core_imrad"],
        stats["papers_respecting_imrad"],
    )
    logger.info("Report written to %s", args.report_path)
    return report


if __name__ == "__main__":
    main()
