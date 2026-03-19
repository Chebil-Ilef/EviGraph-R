"""
unarXive Data Quality Analysis
================================
Streams N papers from the unarXive_2024 HuggingFace dataset and produces a
comprehensive quality report covering every dimension relevant to the
EviGraph-R pipeline.

Usage:
    uv run experiments/unarxiv_dataset/analyze_quality.py --n 5000
    uv run experiments/unarxiv_dataset/analyse_quality.py --n 5000 --out report.txt

Runtime: ~3–6 min for 5 000 papers on a typical laptop (streaming, no GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from time import time

from dotenv import load_dotenv

load_dotenv()

_CITE_RE        = re.compile(r"\{\{cite:([0-9a-f]+)\}\}")
_FIGURE_RE      = re.compile(r"\{\{figure:[0-9a-f\-]+\}\}")
_TABLE_RE       = re.compile(r"\{\{table:[0-9a-f\-]+\}\}")
_YEAR_RE        = re.compile(r"\b(19|20)\d{2}\b")
_ARXIV_RE       = re.compile(r"(?:arxiv[^\d]*)?([0-9]{4}\.[0-9]{4,5}|(?:hep-ph|hep-th|cs|math|eess|stat|physics)/[0-9]{7})", re.I)
_DOI_RE         = re.compile(r"10\.\d{4,}/\S+")

# IMRAD canonical section names (lowercase, partial match is fine)
_IMRAD_KEYWORDS = [
    "introduction", "related work", "background", "method", "approach",
    "experiment", "result", "evaluation", "discussion", "conclusion",
    "abstract", "acknowledgement", "reference", "appendix",
]

# Noise patterns that indicate a broken/unhelpful section title
_NOISE_TITLE_RE = re.compile(
    r"^(\d+\.?\s*)*$"          # pure numbers / numbering
    r"|^\s*$"                   # empty / whitespace
    r"|^[ivxIVX]+\.?\s*$"      # roman numerals alone
    r"|^\W+$"                   # punctuation only
    r"|^(nan|none|null)$",      # literal null strings
    re.I,
)


@dataclass
class Stats:
    # Paper counts
    total: int = 0
    has_abstract: int = 0
    abstract_empty: int = 0
    has_body: int = 0
    body_empty: int = 0

    # Metadata
    has_year: int = 0
    has_authors: int = 0
    has_doi: int = 0            # paper-level DOI
    has_arxiv_id: int = 0       # paper-level arXiv ID
    has_categories: int = 0
    has_title: int = 0

    # Sections
    total_sections: int = 0
    sections_with_empty_title: int = 0
    sections_with_noise_title: int = 0
    sections_imrad_matched: int = 0
    section_title_counter: Counter = field(default_factory=Counter)  # normalised titles
    papers_with_no_section_titles: int = 0   # body present but ALL titles missing/noise

    # In-text citations
    total_cite_markers: int = 0   # all {{cite:xxx}} occurrences
    papers_with_no_citations: int = 0

    # Bib entries
    total_bib_entries: int = 0
    bib_has_doi: int = 0
    bib_has_arxiv: int = 0
    bib_has_title: int = 0
    bib_has_year: int = 0
    bib_has_any_id: int = 0      # doi OR arxiv
    bib_sha_only: int = 0        # no doi, no arxiv → fallback to sha:ref_id

    # Citation marker → bib resolution
    total_cite_ref_ids: int = 0       # unique ref_ids seen in markers
    cite_ref_resolved_doi: int = 0    # resolved to doi in bib
    cite_ref_resolved_arxiv: int = 0  # resolved to arxiv in bib
    cite_ref_sha_only: int = 0        # marker points to bib with no external ID

    # Figure / table markers
    total_figure_markers: int = 0
    total_table_markers: int = 0

    # Body text lengths (token proxy: whitespace-split words)
    body_word_counts: list = field(default_factory=list)

    # Section count distribution
    section_counts: list = field(default_factory=list)

    # Category / discipline breakdown
    category_counter: Counter = field(default_factory=Counter)


def _runtime_profile() -> str:
    profile = os.getenv("INDEXING_PROFILE", "local").strip().lower()
    if profile not in {"local", "hpc"}:
        return "local"
    return profile


def _preferred_device(profile: str) -> str:
    if profile != "hpc":
        return "cpu"

    try:
        import torch
    except ImportError:
        # Keep the convention simple: HPC prefers GPU if available.
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_hf_token_from_env() -> str | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.getenv(key)
        if value and value.strip():
            os.environ.setdefault("HF_TOKEN", value.strip())
            return value.strip()
    return None


def _first_jsonl(raw: str) -> dict | None:

    for line in raw.splitlines():
        line = line.strip()
        if line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None


def _normalise_section_title(title: str) -> str:

    t = title.lower().strip()
    t = re.sub(r"^[\d\.\s]+", "", t)   # strip leading "1. " etc.
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _is_noise_title(title: str) -> bool:
    return bool(_NOISE_TITLE_RE.match(title.strip()))


def _imrad_match(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in _IMRAD_KEYWORDS)


def _extract_doi_from_bib(bib: dict) -> str | None:

    # Common field names
    for key in ("doi", "DOI", "ids"):
        v = bib.get(key)
        if isinstance(v, str):
            m = _DOI_RE.search(v)
            if m:
                return m.group(0)
        elif isinstance(v, dict):
            doi = v.get("doi") or v.get("DOI")
            if doi:
                return doi
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    m = _DOI_RE.search(item)
                    if m:
                        return m.group(0)
    # Fallback: scan raw_text
    raw = bib.get("raw_text") or bib.get("raw") or ""
    m = _DOI_RE.search(raw)
    return m.group(0) if m else None


def _extract_arxiv_from_bib(bib: dict) -> str | None:

    for key in ("arxiv_id", "arxivId", "eprint", "ids"):
        v = bib.get(key)
        if isinstance(v, str) and v.strip():
            m = _ARXIV_RE.search(v)
            if m:
                return m.group(0)
        elif isinstance(v, dict):
            aid = v.get("arxiv") or v.get("arxiv_id") or v.get("eprint")
            if aid:
                return aid
    raw = bib.get("raw_text") or bib.get("raw") or ""
    m = _ARXIV_RE.search(raw)
    return m.group(0) if m else None


def _extract_title_from_bib(bib: dict) -> str | None:
    for key in ("title", "bib_entry_raw"):
        v = bib.get(key)
        if isinstance(v, str) and len(v.strip()) > 5:
            return v.strip()
    return None


def analyse_paper(paper: dict, stats: Stats) -> None:
    stats.total += 1

    meta = paper.get("metadata") or {}
    title = meta.get("title") or paper.get("title") or ""
    if title.strip():
        stats.has_title += 1

    authors = meta.get("authors") or paper.get("authors") or []
    if authors:
        stats.has_authors += 1

    year_raw = (
        meta.get("year")
        or meta.get("date")
        or paper.get("year")
        or ""
    )
    if year_raw and _YEAR_RE.search(str(year_raw)):
        stats.has_year += 1

    doi_raw = meta.get("doi") or paper.get("doi") or ""
    if doi_raw and doi_raw.strip():
        stats.has_doi += 1

    arxiv_id = (
        meta.get("arxiv_id")
        or meta.get("id")
        or paper.get("paper_id")
        or paper.get("_id")
        or ""
    )
    if arxiv_id and re.match(r"[0-9]{4}\.", str(arxiv_id).strip()):
        stats.has_arxiv_id += 1

    cats = meta.get("categories") or paper.get("categories") or []
    if cats:
        stats.has_categories += 1
        if isinstance(cats, str):
            cats = cats.split()
        for cat in cats[:3]:
            prefix = str(cat).split(".")[0].lower()
            stats.category_counter[prefix] += 1

    abstract_text = (
        paper.get("abstract")
        or (paper.get("abstract_text") or "")
        or ""
    )
    if isinstance(abstract_text, dict):
        abstract_text = abstract_text.get("text") or abstract_text.get("value") or ""
    abstract_text = abstract_text.strip()

    if abstract_text:
        stats.has_abstract += 1
    else:
        stats.abstract_empty += 1

    body = paper.get("body_text") or []
    if body:
        stats.has_body += 1
    else:
        stats.body_empty += 1

    total_body_words = 0
    n_sections = 0
    all_titles_noise = True
    paper_cite_markers: set[str] = set()

    for sec in body:
        n_sections += 1
        sec_title = (sec.get("section") or sec.get("section_title") or "").strip()
        sec_text  = (sec.get("text") or "").strip()

        # Section title quality
        if not sec_title:
            stats.sections_with_empty_title += 1
        elif _is_noise_title(sec_title):
            stats.sections_with_noise_title += 1
        else:
            all_titles_noise = False
            norm = _normalise_section_title(sec_title)
            stats.section_title_counter[norm] += 1
            if _imrad_match(sec_title):
                stats.sections_imrad_matched += 1

        # Citations in this section
        cites_here = _CITE_RE.findall(sec_text)
        paper_cite_markers.update(cites_here)
        stats.total_cite_markers += len(cites_here)

        # Figures / tables
        stats.total_figure_markers += len(_FIGURE_RE.findall(sec_text))
        stats.total_table_markers  += len(_TABLE_RE.findall(sec_text))

        total_body_words += len(sec_text.split())

    stats.total_sections += n_sections
    if body and all_titles_noise:
        stats.papers_with_no_section_titles += 1
    if not paper_cite_markers:
        stats.papers_with_no_citations += 1

    if n_sections:
        stats.section_counts.append(n_sections)
    if total_body_words:
        stats.body_word_counts.append(total_body_words)

    bib_entries = paper.get("bib_entries") or {}
    if isinstance(bib_entries, list):
        # Some versions store as list
        bib_entries = {e.get("ref_id", str(i)): e for i, e in enumerate(bib_entries) if isinstance(e, dict)}

    for ref_id, bib in bib_entries.items():
        if not isinstance(bib, dict):
            continue
        stats.total_bib_entries += 1

        doi   = _extract_doi_from_bib(bib)
        arxiv = _extract_arxiv_from_bib(bib)
        title = _extract_title_from_bib(bib)
        year  = bib.get("year") or bib.get("Year") or ""

        if doi:
            stats.bib_has_doi += 1
        if arxiv:
            stats.bib_has_arxiv += 1
        if title:
            stats.bib_has_title += 1
        if year and _YEAR_RE.search(str(year)):
            stats.bib_has_year += 1
        if doi or arxiv:
            stats.bib_has_any_id += 1
        else:
            stats.bib_sha_only += 1

    stats.total_cite_ref_ids += len(paper_cite_markers)
    for ref_id in paper_cite_markers:
        bib = bib_entries.get(ref_id)
        if bib and isinstance(bib, dict):
            doi   = _extract_doi_from_bib(bib)
            arxiv = _extract_arxiv_from_bib(bib)
            if doi:
                stats.cite_ref_resolved_doi   += 1
            elif arxiv:
                stats.cite_ref_resolved_arxiv += 1
            else:
                stats.cite_ref_sha_only       += 1
        else:
            # marker points to a ref_id not in bib_entries at all
            stats.cite_ref_sha_only += 1


def _percentile(data: list[int | float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = max(0, int(len(s) * p / 100) - 1)
    return s[idx]


def print_report(s: Stats, n_requested: int, output_file: str | None) -> None:
    N = s.total

    def pct(numerator: int, denominator: int = N) -> str:
        if denominator == 0:
            return "N/A"
        return f"{100 * numerator / denominator:.1f}%"

    def avg(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    lines: list[str] = []
    A = lines.append

    A("=" * 72)
    A("  unarXive 2024 — Data Quality Report")
    A("=" * 72)
    A(f"  Papers analysed : {N:,}  (requested {n_requested:,})")
    A("")

    A("── 1. PAPER METADATA ────────────────────────────────────────────────")
    A(f"  Title present          : {s.has_title:>7,}  ({pct(s.has_title)})")
    A(f"  Authors present        : {s.has_authors:>7,}  ({pct(s.has_authors)})")
    A(f"  Year present           : {s.has_year:>7,}  ({pct(s.has_year)})")
    A(f"  Paper DOI present      : {s.has_doi:>7,}  ({pct(s.has_doi)})")
    A(f"  Paper arXiv ID present : {s.has_arxiv_id:>7,}  ({pct(s.has_arxiv_id)})")
    A(f"  Categories present     : {s.has_categories:>7,}  ({pct(s.has_categories)})")
    A("")

    A("── 2. ABSTRACT ──────────────────────────────────────────────────────")
    A(f"  Has non-empty abstract : {s.has_abstract:>7,}  ({pct(s.has_abstract)})")
    A(f"  Empty / missing        : {s.abstract_empty:>7,}  ({pct(s.abstract_empty)})")
    A("")

    A("── 3. BODY TEXT ─────────────────────────────────────────────────────")
    A(f"  Has body text          : {s.has_body:>7,}  ({pct(s.has_body)})")
    A(f"  Empty / missing body   : {s.body_empty:>7,}  ({pct(s.body_empty)})")
    if s.body_word_counts:
        A(f"  Body word count (mean) : {avg(s.body_word_counts):>9,.0f}")
        A(f"  Body word count (p10)  : {_percentile(s.body_word_counts, 10):>9,.0f}")
        A(f"  Body word count (p50)  : {_percentile(s.body_word_counts, 50):>9,.0f}")
        A(f"  Body word count (p90)  : {_percentile(s.body_word_counts, 90):>9,.0f}")
    A("")

    A("── 4. SECTION STRUCTURE ─────────────────────────────────────────────")
    A(f"  Total sections found   : {s.total_sections:>7,}")
    A(f"  Avg sections / paper   : {avg(s.section_counts):>9.1f}")
    if s.section_counts:
        A(f"  Sections p10/p50/p90   : {_percentile(s.section_counts,10):.0f} / "
          f"{_percentile(s.section_counts,50):.0f} / "
          f"{_percentile(s.section_counts,90):.0f}")
    A(f"  Sections w/ empty title: {s.sections_with_empty_title:>7,}  "
      f"({pct(s.sections_with_empty_title, s.total_sections)}  of all sections)")
    A(f"  Sections w/ noise title: {s.sections_with_noise_title:>7,}  "
      f"({pct(s.sections_with_noise_title, s.total_sections)}  of all sections)")
    A(f"  Sections w/ IMRAD match: {s.sections_imrad_matched:>7,}  "
      f"({pct(s.sections_imrad_matched, s.total_sections)}  of all sections)")
    A(f"  Papers w/ ALL titles   ")
    A(f"    missing/noise (body≥1): {s.papers_with_no_section_titles:>6,}  "
      f"({pct(s.papers_with_no_section_titles, s.has_body)}  of papers with body)")
    A("")
    A("  Top 30 normalised section titles:")
    for title, cnt in s.section_title_counter.most_common(30):
        bar = "█" * min(40, int(40 * cnt / max(1, s.section_title_counter.most_common(1)[0][1])))
        A(f"    {cnt:>6,}  {bar:<40}  {title}")
    A("")

    A("── 5. IN-TEXT CITATION MARKERS ──────────────────────────────────────")
    A(f"  Total {{{{cite:xxx}}}} markers : {s.total_cite_markers:>7,}")
    if N:
        A(f"  Avg per paper            : {s.total_cite_markers/N:>9.1f}")
    A(f"  Papers with NO citations : {s.papers_with_no_citations:>7,}  ({pct(s.papers_with_no_citations)})")
    A(f"  Total figure markers     : {s.total_figure_markers:>7,}")
    A(f"  Total table  markers     : {s.total_table_markers:>7,}")
    A("")

    A("── 6. BIB ENTRY QUALITY ─────────────────────────────────────────────")
    B = s.total_bib_entries
    A(f"  Total bib entries        : {B:>7,}")
    if N:
        A(f"  Avg bib entries / paper  : {B/N:>9.1f}")
    A(f"  Has DOI                  : {s.bib_has_doi:>7,}  ({pct(s.bib_has_doi, B)}  of entries)")
    A(f"  Has arXiv ID             : {s.bib_has_arxiv:>7,}  ({pct(s.bib_has_arxiv, B)}  of entries)")
    A(f"  Has DOI OR arXiv         : {s.bib_has_any_id:>7,}  ({pct(s.bib_has_any_id, B)}  of entries)")
    A(f"  SHA-only (no ext ID)     : {s.bib_sha_only:>7,}  ({pct(s.bib_sha_only, B)}  of entries)")
    A(f"  Has title string         : {s.bib_has_title:>7,}  ({pct(s.bib_has_title, B)}  of entries)")
    A(f"  Has year                 : {s.bib_has_year:>7,}  ({pct(s.bib_has_year, B)}  of entries)")
    A("")

    A("── 7. CITE-MARKER → BIB RESOLUTION ─────────────────────────────────")
    R = s.total_cite_ref_ids
    A(f"  Unique ref_ids in markers: {R:>7,}")
    A(f"  → resolved to DOI        : {s.cite_ref_resolved_doi:>7,}  ({pct(s.cite_ref_resolved_doi, R)})")
    A(f"  → resolved to arXiv ID   : {s.cite_ref_resolved_arxiv:>7,}  ({pct(s.cite_ref_resolved_arxiv, R)})")
    A(f"  → SHA-only fallback      : {s.cite_ref_sha_only:>7,}  ({pct(s.cite_ref_sha_only, R)})")
    A("")
    resolved = s.cite_ref_resolved_doi + s.cite_ref_resolved_arxiv
    A(f"  ► Resolvable to graph node: {pct(resolved, R)}  of all in-text citations")
    A(f"  ► SHA-only (graph edge gap): {pct(s.cite_ref_sha_only, R)}  — these need Crossref / OpenAlex fallback")
    A("")

    A("── 8. DISCIPLINE BREAKDOWN (arXiv category prefix) ──────────────────")
    for cat, cnt in s.category_counter.most_common(15):
        A(f"    {cnt:>6,}  {cat}")
    A("")

    A("── 9. PIPELINE COMPATIBILITY SUMMARY ────────────────────────────────")
    issues: list[str] = []

    missing_abstract_pct = 100 * s.abstract_empty / N if N else 0
    if missing_abstract_pct > 5:
        issues.append(f"WARN  {missing_abstract_pct:.1f}% papers have no abstract → abstract chunks missing")

    missing_body_pct = 100 * s.body_empty / N if N else 0
    if missing_body_pct > 5:
        issues.append(f"WARN  {missing_body_pct:.1f}% papers have no body text → retrieval gap")

    no_title_pct = 100 * s.papers_with_no_section_titles / max(1, s.has_body)
    if no_title_pct > 20:
        issues.append(f"WARN  {no_title_pct:.1f}% of papers with body have no usable section titles → chunk_section will be empty string")

    empty_title_pct = 100 * s.sections_with_empty_title / max(1, s.total_sections)
    if empty_title_pct > 15:
        issues.append(f"INFO  {empty_title_pct:.1f}% of sections have empty title field (expected for some datasets)")

    sha_bib_pct = 100 * s.bib_sha_only / max(1, B)
    if sha_bib_pct > 30:
        issues.append(f"WARN  {sha_bib_pct:.1f}% of bib entries have NO external ID → graph coverage limited without API resolution")

    sha_cite_pct = 100 * s.cite_ref_sha_only / max(1, R)
    if sha_cite_pct > 30:
        issues.append(f"WARN  {sha_cite_pct:.1f}% of in-text citations cannot be resolved to graph node without API calls")

    no_year_pct = 100 * (N - s.has_year) / N if N else 0
    if no_year_pct > 10:
        issues.append(f"INFO  {no_year_pct:.1f}% of papers missing year metadata")

    no_citations_pct = 100 * s.papers_with_no_citations / N if N else 0
    if no_citations_pct > 20:
        issues.append(f"INFO  {no_citations_pct:.1f}% of papers have zero in-text citation markers")

    if not issues:
        A("  ✓ No major issues detected at this sample size.")
    for issue in issues:
        A(f"  {issue}")
    A("")
    A("=" * 72)

    report_text = "\n".join(lines)
    print(report_text)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        print(f"\nReport saved → {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="unarXive data quality analysis")
    parser.add_argument("--n",        type=int,   default=3000,                    help="Number of papers to analyse")
    parser.add_argument("--dataset",  type=str,   default="ines-besrour/unarxive_2024")
    parser.add_argument("--split",    type=str,   default="train")
    parser.add_argument("--out",      type=str,   default=None,                    help="Optional path to save the report as a .txt file")
    parser.add_argument("--verbose",  action="store_true",                          help="Print progress every 500 papers")
    args = parser.parse_args()
    profile = _runtime_profile()
    device = _preferred_device(profile)
    hf_token = _load_hf_token_from_env()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed.  pip install datasets", file=sys.stderr)
        sys.exit(1)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nunarXive Data Quality Analysis — {now_str}\n", flush=True)
    print(f"Runtime profile: {profile} (preferred device: {device})", flush=True)
    print(f"HF token loaded from env: {'yes' if hf_token else 'no'}", flush=True)
    print(f"Streaming {args.n:,} papers from {args.dataset} …", flush=True)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    print(f"Finished at time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", flush=True)
    print("Processing papers …\n", flush=True)

    stats = Stats()
    skipped = 0

    for row in ds:
        if stats.total >= args.n:
            break

        raw_jsonl = (row.get("jsonl") or "").strip()
        if not raw_jsonl:
            skipped += 1
            continue

        paper = _first_jsonl(raw_jsonl)
        if paper is None:
            skipped += 1
            continue

        analyse_paper(paper, stats)

        if args.verbose and stats.total % 500 == 0:
            print(f"  … {stats.total:,} / {args.n:,}", flush=True)

    if skipped:
        print(f"  (skipped {skipped} empty/unparseable rows)", flush=True)

    print(f"Done — analysed {stats.total:,} papers.\n", flush=True)
    print_report(stats, args.n, args.out)


if __name__ == "__main__":
    main()
