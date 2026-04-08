from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from config.settings import PATHS, QDRANT_ACTIVE
from qdrant_client.http.exceptions import UnexpectedResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.qdrant import ensure_qdrant_runtime, qdrant_client

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
    # Used by the sequence repair pass to pick the next-best valid label.
    raw_probs: list[float] | None = None


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


def _iter_points(
    client,
    collection_name: str,
    page_size: int,
    max_points: int | None,
    truncation_flag: list[bool] | None = None,
):

    offset = None
    seen = 0
    while True:
        try:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=page_size,
                offset=offset,
                with_vectors=False,
                with_payload=["paper_id_arxiv", "section_title", "embed_text", "chunk_index"],
            )
        except UnexpectedResponse as exc:
            logger.warning("Qdrant scroll returned error (treating as end): %s", exc)
            if truncation_flag is not None:
                truncation_flag[0] = True
            break

        if not points:
            break

        for point in points:
            yield point
            seen += 1
            if max_points is not None and seen >= max_points:
                return

        if offset is None:
            break


def collect_sections(client, collection_name: str, page_size: int, max_points: int | None):
    section_map: dict[tuple[str, str], SectionRecord] = {}
    # Tracks the minimum chunk_index seen per section — used to select representative text
    # and to order sections within a paper.  Avoids the O(n) scan of the old implementation.
    section_min_idx: dict[tuple[str, str], int] = {}
    paper_sections: dict[str, set[tuple[str, str]]] = defaultdict(set)

    truncation_flag: list[bool] = [False]
    for point in _iter_points(
        client,
        collection_name,
        page_size=page_size,
        max_points=max_points,
        truncation_flag=truncation_flag,
    ):
        payload = point.payload or {}
        paper_id = str(payload.get("paper_id_arxiv") or "")
        title = str(payload.get("section_title") or "")
        embed_text = str(payload.get("embed_text") or "")
        chunk_index = int(payload.get("chunk_index") or 0)

        if not paper_id:
            continue

        title_key = _normalize_title(title)
        section_key = (paper_id, title_key)

        if section_key not in section_map:
            section_map[section_key] = SectionRecord(
                paper_id=paper_id,
                title=title,
                text=embed_text,
                point_ids=[point.id],
                heuristic_label=heuristic_imrad_label(title),
                chunk_index_min=chunk_index,
            )
            section_min_idx[section_key] = chunk_index
            paper_sections[paper_id].add(section_key)
        else:
            record = section_map[section_key]
            record.point_ids.append(point.id)
            if chunk_index < section_min_idx[section_key]:
                # Use the earliest chunk's text as the representative for classification.
                section_min_idx[section_key] = chunk_index
                record.chunk_index_min = chunk_index
                record.text = embed_text

    # Build ordered paper_to_titles sorted by min chunk_index.
    paper_to_titles: dict[str, list[tuple[int, tuple[str, str]]]] = {
        paper_id: sorted(
            ((section_min_idx[k], k) for k in keys),
            key=lambda it: it[0],
        )
        for paper_id, keys in paper_sections.items()
    }

    if truncation_flag[0]:
        logger.warning(
            "SCROLL TRUNCATED by Qdrant 500 error — only %d points were read "
            "(%d unique sections across %d papers). "
            "The collection contains more data. Re-run or investigate the Qdrant error above.",
            sum(len(s.point_ids) for s in section_map.values()),
            len(section_map),
            len(paper_to_titles),
        )

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
        if section.heuristic_label is None and section.text.strip()
    ]
    if not unknown_keys:
        logger.info("No unknown section titles to classify")
        return {}

    logger.info(
        "Classifying %d unresolved section titles with %s",
        len(unknown_keys), model_id,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()

    for start in range(0, len(unknown_keys), batch_size):
        batch_keys = unknown_keys[start : start + batch_size]
        # Prepend the section title so the model sees title + body, matching how
        # academic-text models best distinguish section roles.
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
            # Already handled by classify_unknown_sections; do not overwrite.
            pass
        else:
            # No heuristic match, no text to classify (empty embed_text).
            section.final_label = None
            section.source = "unresolved"
            section.confidence = None


def repair_imrad_sequences(
    sections: dict[tuple[str, str], SectionRecord],
    paper_to_titles: dict[str, list[tuple[int, tuple[str, str]]]],
    id2label: dict[int, str],
) -> int:

    relabelled = 0
    imrad_labels = list(IMRAD_ORDER.keys())  # ordered by IMRAD_ORDER value

    for _, ordered in paper_to_titles.items():
        last_order = -1  # tracks the highest IMRAD_ORDER value committed so far

        for _, key in ordered:
            section = sections[key]
            label = section.final_label

            # Skip anything that is not an IMRaD content label.
            if label is None or label not in IMRAD_ORDER:
                continue

            current_order = IMRAD_ORDER[label]

            if current_order > last_order:
                last_order = current_order
                continue

            # This section violates order — try to find a valid reassignment.
            if section.source != "classifier" or section.raw_probs is None:
                # Heuristic-derived labels are not touched.
                continue

            # Candidates: IMRaD labels with order > last_order, sorted by their
            # classifier probability (descending), so we pick the most confident valid label.
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
            # If no valid forward label exists, keep the original label as-is.
            # Forcing a drop here cascades and strips valid downstream sections.

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
    n_imrad_after  = sum(1 for s in sections.values() if s.final_label in IMRAD_ORDER)
    n_skipped      = sum(1 for s in sections.values() if s.final_label == SKIP_LABEL)
    n_no_label     = sum(1 for s in sections.values() if s.final_label is None)

    n_from_heuristic = sum(1 for s in sections.values() if s.final_label in IMRAD_ORDER and s.source == "heuristic")
    n_from_model     = sum(1 for s in sections.values() if s.final_label in IMRAD_ORDER and s.source == "classifier")
    n_from_repair    = sum(1 for s in sections.values() if s.final_label in IMRAD_ORDER and s.source == "sequence_repair")

    n_no_label_unresolved = sum(1 for s in sections.values() if s.final_label is None and s.source == "unresolved")

    label_counts_before = {lbl: 0 for lbl in IMRAD_ORDER}
    label_counts_after  = {lbl: 0 for lbl in IMRAD_ORDER}
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
        "papers_respecting_imrad_after":  papers_imrad_after,

        "sections_total": len(sections),
        "sections_labelled_imrad_before": n_imrad_before,
        "sections_labelled_imrad_after":  n_imrad_after,
        "sections_skipped":               n_skipped,   # abstract, refs, appendix — intentionally excluded
        "sections_no_label":              n_no_label,  # could not be classified

        "imrad_label_sources": {
            "heuristic":       n_from_heuristic,
            "classifier":      n_from_model,
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


def update_qdrant_payloads(
    client,
    collection_name: str,
    sections: dict[tuple[str, str], SectionRecord],
    batch_size: int,
) -> int:
    updates = 0
    for section in sections.values():
        if not section.point_ids:
            continue
        payload = {
            "imrad_label": section.final_label,
            "imrad_label_source": section.source,
            "imrad_classifier_confidence": section.confidence,
            "imrad_section_title_norm": _normalize_title(section.title),
        }
        point_ids = section.point_ids
        for start in range(0, len(point_ids), batch_size):
            batch_ids = point_ids[start : start + batch_size]
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
        # Dominant source = the one that appears most often for this title.
        dominant_reason = max(item["source_counts"], key=lambda s: item["source_counts"][s])
        # Omit paper list for skip-dominated entries (abstract, references, etc.) — always
        # universal and not actionable.  Keep capped list for low_confidence/unresolved.
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


def write_report(output_path: Path, stats: dict[str, Any], sections: dict[tuple[str, str], SectionRecord]) -> None:
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
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--qdrant-update-batch-size", type=int, default=512)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
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

    logger.info("Collecting points from collection %s", args.collection_name)
    sections, paper_to_titles = collect_sections(
        client,
        collection_name=args.collection_name,
        page_size=args.scroll_page_size,
        max_points=args.max_points,
    )
    logger.info("Collected %d unique section titles across %d papers", len(sections), len(paper_to_titles))

    device = _resolve_device(args.device)
    id2label = classify_unknown_sections(
        sections,
        model_id=args.model_id,
        device=device,
        batch_size=args.inference_batch_size,
    )
    finalize_labels(sections)

    repaired = repair_imrad_sequences(sections, paper_to_titles, id2label)
    logger.info("Sequence repair: %d sections relabelled", repaired)

    stats = build_stats(sections, paper_to_titles)

    updated_points = 0
    if not args.dry_run:
        updated_points = update_qdrant_payloads(
            client,
            collection_name=args.collection_name,
            sections=sections,
            batch_size=args.qdrant_update_batch_size,
        )

    report_path = Path(args.report_path)
    write_report(report_path, stats, sections)

    # Derive non-IMRaD report path; explicit arg takes precedence over auto-derivation.
    if args.non_imrad_report_path:
        non_imrad_report_path = Path(args.non_imrad_report_path)
    else:
        non_imrad_report_path = report_path.with_name(report_path.stem + "_non_imrad.json")

    non_imrad_sections = collect_non_imrad_sections(sections)
    write_non_imrad_report(non_imrad_report_path, non_imrad_sections)

    logger.info("IMRaD postprocessing complete")
    logger.info(
        "Papers respecting IMRaD (before -> after): %d -> %d",
        stats["papers_respecting_imrad_before"], stats["papers_respecting_imrad_after"],
    )
    logger.info("Classifier confidence: %s", stats["classifier_confidence"])
    src = stats["imrad_label_sources"]
    logger.info(
        "Sections total:%d  labelled_imrad(before->after):%d->%d  skipped:%d  no_label:%d",
        stats["sections_total"],
        stats["sections_labelled_imrad_before"],
        stats["sections_labelled_imrad_after"],
        stats["sections_skipped"],
        stats["sections_no_label"],
    )
    logger.info(
        "IMRaD sources — heuristic:%d  classifier:%d  sequence_repair:%d",
        src["heuristic"], src["classifier"], src["sequence_repair"],
    )
    logger.info("No-label (no body text): %d", stats["sections_no_label_unresolved"])
    logger.info("Non-IMRaD unique sections: %d", len(non_imrad_sections))
    logger.info("Updated points in Qdrant: %d", updated_points)
    logger.info("Report written to %s", report_path)
    logger.info("Non-IMRaD report written to %s", non_imrad_report_path)

    return {
        "stats": stats,
        "updated_points": updated_points,
        "report_path": str(report_path),
        "non_imrad_report_path": str(non_imrad_report_path),
    }


if __name__ == "__main__":
    main()
