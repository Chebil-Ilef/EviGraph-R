from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()

_CITE_RE = re.compile(r"\{\{cite:([0-9a-z\-]+)\}\}", re.I)
_FIGURE_RE = re.compile(r"\{\{figure:[0-9a-z\-]+\}\}", re.I)
_TABLE_RE = re.compile(r"\{\{table:[0-9a-z\-]+\}\}", re.I)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_ARXIV_RE = re.compile(
    r"(?:arxiv[^\d]*)?([0-9]{4}\.[0-9]{4,5}|(?:hep-ph|hep-th|cs|math|eess|stat|physics)/[0-9]{7})",
    re.I,
)
_DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.I)

_IMRAD_GROUPS = {
    "abstract": ("abstract",),
    "introduction": ("introduction", "intro"),
    "background": ("background", "related work", "related", "preliminar"),
    "methods": ("method", "approach", "model", "architecture", "setup", "algorithm", "material"),
    "results": ("experiment", "evaluation", "result", "analysis", "findings", "ablation"),
    "discussion": ("discussion",),
    "conclusion": ("conclusion", "limitations", "future work"),
    "references": ("reference", "bibliograph"),
    "appendix": ("appendix", "supplement"),
}

_NOISE_TITLE_RE = re.compile(
    r"^(\d+\.?\s*)*$"
    r"|^\s*$"
    r"|^[ivxIVX]+\.?\s*$"
    r"|^\W+$"
    r"|^(nan|none|null)$",
    re.I,
)

_PLACEHOLDER_REF_RE = re.compile(r"\b(?:section|figure|table|appendix)?\s*REF\b", re.I)

_SECTION_CONTAINER_KEYS = {"sections", "subsections", "children"}
_SECTION_TEXT_KEYS = {"text", "content", "value"}
_SECTION_META_KEYS = {
    "text",
    "content",
    "value",
    "section",
    "section_title",
    "title",
    "cite_spans",
    "ref_spans",
}
_GENERIC_CONTAINER_KEYS = {"metadata", "bib_entries", "ref_entries", "abstract", "versions", "authors_parsed"}
_ID_FIELDS = ("openalex", "sem_openalex", "pubmed", "pmc", "doi", "arxiv")


@dataclass
class SectionRecord:
    title: str
    text: str
    level: int
    path: tuple[str, ...] = ()


@dataclass
class Stats:
    total: int = 0
    has_abstract: int = 0
    abstract_empty: int = 0
    has_body: int = 0
    body_empty: int = 0

    has_year: int = 0
    has_authors: int = 0
    has_doi: int = 0
    has_arxiv_id: int = 0
    has_categories: int = 0
    has_title: int = 0

    total_sections: int = 0
    total_subsections: int = 0
    sections_with_empty_title: int = 0
    sections_with_noise_title: int = 0
    sections_with_empty_text: int = 0
    sections_imrad_matched: int = 0
    papers_with_no_section_titles: int = 0
    papers_with_any_imrad_title: int = 0
    papers_with_duplicate_titles: int = 0
    section_title_counter: Counter = field(default_factory=Counter)
    imrad_counter: Counter = field(default_factory=Counter)

    total_cite_markers: int = 0
    papers_with_no_citations: int = 0
    total_figure_markers: int = 0
    total_table_markers: int = 0

    total_bib_entries: int = 0
    bib_has_title: int = 0
    bib_has_year: int = 0
    bib_has_any_id: int = 0
    bib_no_unique_id: int = 0
    bib_id_counter: Counter = field(default_factory=Counter)

    total_cite_ref_ids: int = 0
    cited_ref_has_any_id: int = 0
    cited_ref_no_unique_id: int = 0
    cited_ref_missing_bib: int = 0
    cited_ref_id_counter: Counter = field(default_factory=Counter)

    body_word_counts: list[int] = field(default_factory=list)
    section_counts: list[int] = field(default_factory=list)
    category_counter: Counter = field(default_factory=Counter)

    papers_with_placeholder_refs: int = 0
    papers_with_empty_sections: int = 0
    total_placeholder_refs: int = 0
    papers_with_missing_cited_bib: int = 0
    papers_with_unresolved_cited_refs: int = 0
    papers_with_sections_but_no_text: int = 0
    ref_entries_no_caption: int = 0
    ref_entries_total: int = 0


def _runtime_profile() -> str:
    profile = os.getenv("INDEXING_PROFILE", "local").strip().lower()
    return profile if profile in {"local", "hpc"} else "local"


def _preferred_device(profile: str) -> str:
    if profile != "hpc":
        return "cpu"
    try:
        import torch
    except ImportError:
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
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
    return None


def _normalise_section_title(title: str) -> str:
    t = title.lower().strip()
    t = re.sub(r"^[\d\W_]+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _is_noise_title(title: str) -> bool:
    return bool(_NOISE_TITLE_RE.match(title.strip()))


def _imrad_bucket(title: str) -> str | None:
    t = _normalise_section_title(title)
    for bucket, keywords in _IMRAD_GROUPS.items():
        if any(kw in t for kw in keywords):
            return bucket
    return None


def _extract_year(paper: dict, meta: dict) -> str | None:
    candidates: list[object] = [
        meta.get("year"),
        meta.get("date"),
        meta.get("update_date"),
        paper.get("year"),
        paper.get("date"),
    ]
    versions = meta.get("versions") or []
    if isinstance(versions, list):
        for version in versions:
            if isinstance(version, dict):
                candidates.append(version.get("created"))
    for candidate in candidates:
        if not candidate:
            continue
        match = _YEAR_RE.search(str(candidate))
        if match:
            return match.group(0)
    return None


def _extract_abstract_text(paper: dict) -> str:
    abstract = paper.get("abstract") or paper.get("abstract_text") or ""
    if isinstance(abstract, dict):
        for key in ("text", "value", "abstract"):
            value = abstract.get(key)
            if isinstance(value, str):
                return value.strip()
        return ""
    if isinstance(abstract, list):
        return " ".join(str(x).strip() for x in abstract if str(x).strip()).strip()
    return str(abstract).strip()


def _looks_like_section_leaf(node: dict) -> bool:
    if any(key in node for key in _SECTION_TEXT_KEYS | {"cite_spans", "ref_spans"}):
        return True
    if any(key in node for key in ("section", "section_title", "title")):
        return True
    return False


def _yield_section_records_from_body(body: object) -> Iterable[SectionRecord]:
    if not isinstance(body, list):
        return
    for item in body:
        if not isinstance(item, dict):
            continue
        title = str(item.get("section") or item.get("section_title") or item.get("title") or "").strip()
        text = str(item.get("text") or item.get("content") or "").strip()
        if title or text:
            yield SectionRecord(title=title, text=text, level=1, path=(title,) if title else ())


def _yield_section_records_from_sections(node: object, level: int = 1, parent_path: tuple[str, ...] = ()) -> Iterable[SectionRecord]:
    if isinstance(node, list):
        for item in node:
            yield from _yield_section_records_from_sections(item, level=level, parent_path=parent_path)
        return

    if not isinstance(node, dict):
        return

    for key, value in node.items():
        if key in _GENERIC_CONTAINER_KEYS:
            continue

        if isinstance(value, str):
            title = str(key).strip()
            text = value.strip()
            if title or text:
                path = parent_path + ((title,) if title else ())
                yield SectionRecord(title=title, text=text, level=level, path=path)
            continue

        if isinstance(value, dict):
            title = str(value.get("section") or value.get("section_title") or value.get("title") or key).strip()
            text = ""
            for text_key in _SECTION_TEXT_KEYS:
                maybe_text = value.get(text_key)
                if isinstance(maybe_text, str) and maybe_text.strip():
                    text = maybe_text.strip()
                    break
            path = parent_path + ((title,) if title else ())
            if title or text or _looks_like_section_leaf(value):
                yield SectionRecord(title=title, text=text, level=level, path=path)

            for subkey, subvalue in value.items():
                if subkey in _SECTION_META_KEYS:
                    continue
                if subkey in _SECTION_CONTAINER_KEYS or isinstance(subvalue, (dict, list)):
                    child_path = path if title else parent_path
                    yield from _yield_section_records_from_sections(
                        {subkey: subvalue} if subkey not in _SECTION_CONTAINER_KEYS else subvalue,
                        level=level + 1,
                        parent_path=child_path,
                    )
            continue

        if isinstance(value, list):
            title = str(key).strip()
            path = parent_path + ((title,) if title else ())
            yield from _yield_section_records_from_sections(value, level=level + 1, parent_path=path)


def _extract_section_records(paper: dict) -> list[SectionRecord]:
    records: list[SectionRecord] = []
    seen: set[tuple[str, str, int, tuple[str, ...]]] = set()

    def add(record: SectionRecord) -> None:
        key = (record.title, record.text, record.level, record.path)
        if key not in seen:
            seen.add(key)
            records.append(record)

    body_records = list(_yield_section_records_from_body(paper.get("body_text")))
    for record in body_records:
        add(record)

    # Only fall back to recursive sections parser if body_text yielded nothing.
    # unarXive uses body_text exclusively; the sections field can be arbitrarily
    # nested and causes exponential blowup if traversed unconditionally.
    if not body_records:
        sections = paper.get("sections")
        if sections:
            for record in _yield_section_records_from_sections(sections):
                add(record)

    return records

def _extract_id_fields_from_bib(bib: dict) -> dict[str, str]:
    ids = bib.get("ids") if isinstance(bib.get("ids"), dict) else {}
    raw = str(bib.get("bib_entry_raw") or bib.get("raw_text") or bib.get("raw") or "")

    openalex = ids.get("open_alex_id") or ids.get("openalex_id") or ids.get("openalex") or ""
    sem_openalex = ids.get("sem_open_alex_id") or ids.get("semantic_scholar_id") or ids.get("sem_openalex") or ""
    pubmed = ids.get("pubmed_id") or ids.get("pubmed") or ""
    pmc = ids.get("pmc_id") or ids.get("pmcid") or ids.get("pmc") or ""

    doi = ""
    for candidate in (bib.get("doi"), ids.get("doi"), raw):
        if isinstance(candidate, str):
            match = _DOI_RE.search(candidate)
            if match:
                doi = match.group(0)
                break

    arxiv = ""
    for candidate in (bib.get("arxiv_id"), bib.get("eprint"), ids.get("arxiv_id"), raw):
        if isinstance(candidate, str):
            match = _ARXIV_RE.search(candidate)
            if match:
                arxiv = match.group(1)
                break

    return {
        "openalex": str(openalex).strip(),
        "sem_openalex": str(sem_openalex).strip(),
        "pubmed": str(pubmed).strip(),
        "pmc": str(pmc).strip(),
        "doi": doi.strip(),
        "arxiv": arxiv.strip(),
    }


def _has_any_external_id(id_map: dict[str, str]) -> bool:
    return any(id_map.get(field) for field in _ID_FIELDS)


def _extract_title_from_bib(bib: dict) -> str | None:
    for key in ("title", "bib_entry_raw"):
        value = bib.get(key)
        if isinstance(value, str) and len(value.strip()) > 5:
            return value.strip()
    return None


def _extract_bib_year(bib: dict) -> str | None:
    for candidate in (bib.get("year"), bib.get("Year"), bib.get("bib_entry_raw"), bib.get("raw_text"), bib.get("raw")):
        if not candidate:
            continue
        match = _YEAR_RE.search(str(candidate))
        if match:
            return match.group(0)
    return None


def _iter_bib_entries(paper: dict) -> dict[str, dict]:
    bib_entries = paper.get("bib_entries") or {}
    if isinstance(bib_entries, list):
        return {
            entry.get("ref_id", str(i)): entry
            for i, entry in enumerate(bib_entries)
            if isinstance(entry, dict)
        }
    if isinstance(bib_entries, dict):
        return {str(key): value for key, value in bib_entries.items() if isinstance(value, dict)}
    return {}


def _count_ref_entry_captions(paper: dict, stats: Stats) -> None:
    ref_entries = paper.get("ref_entries") or {}
    if not isinstance(ref_entries, dict):
        return
    for value in ref_entries.values():
        if not isinstance(value, dict):
            continue
        stats.ref_entries_total += 1
        caption = str(value.get("caption") or "").strip()
        if not caption or caption == "NO_CAPTION":
            stats.ref_entries_no_caption += 1


def analyse_paper(paper: dict, stats: Stats) -> None:
    stats.total += 1
    meta = paper.get("metadata") or {}

    title = str(meta.get("title") or paper.get("title") or "").strip()
    if title:
        stats.has_title += 1

    authors = meta.get("authors") or paper.get("authors") or []
    if authors:
        stats.has_authors += 1

    if _extract_year(paper, meta):
        stats.has_year += 1

    doi_raw = str(meta.get("doi") or paper.get("doi") or "").strip()
    if doi_raw:
        stats.has_doi += 1

    arxiv_id = str(meta.get("arxiv_id") or meta.get("id") or paper.get("paper_id") or paper.get("_id") or "").strip()
    if arxiv_id and _ARXIV_RE.search(arxiv_id):
        stats.has_arxiv_id += 1

    cats = meta.get("categories") or paper.get("categories") or []
    if cats:
        stats.has_categories += 1
        if isinstance(cats, str):
            cats = cats.split()
        for cat in cats[:3]:
            prefix = str(cat).split(".")[0].lower()
            stats.category_counter[prefix] += 1

    abstract_text = _extract_abstract_text(paper)
    if abstract_text:
        stats.has_abstract += 1
    else:
        stats.abstract_empty += 1

    sections = _extract_section_records(paper)
    sections_with_text = [section for section in sections if section.text.strip()]

    if sections_with_text:
        stats.has_body += 1
    else:
        stats.body_empty += 1

    if sections and not sections_with_text:
        stats.papers_with_sections_but_no_text += 1
    if not sections:
        stats.papers_with_empty_sections += 1

    total_body_words = 0
    paper_cite_markers: set[str] = set()
    all_titles_noise = True
    has_imrad_title = False
    placeholder_count = 0
    per_paper_titles: Counter = Counter()

    for section in sections:
        stats.total_sections += 1
        if section.level > 1:
            stats.total_subsections += 1

        sec_title = section.title.strip()
        sec_text = section.text.strip()

        if not sec_text:
            stats.sections_with_empty_text += 1
        else:
            total_body_words += len(sec_text.split())

        if not sec_title:
            stats.sections_with_empty_title += 1
        elif _is_noise_title(sec_title):
            stats.sections_with_noise_title += 1
        else:
            all_titles_noise = False
            norm = _normalise_section_title(sec_title)
            per_paper_titles[norm] += 1
            stats.section_title_counter[norm] += 1
            bucket = _imrad_bucket(sec_title)
            if bucket:
                has_imrad_title = True
                stats.sections_imrad_matched += 1
                stats.imrad_counter[bucket] += 1

        paper_cite_markers.update(_CITE_RE.findall(sec_text))
        stats.total_cite_markers += len(_CITE_RE.findall(sec_text))
        stats.total_figure_markers += len(_FIGURE_RE.findall(sec_text))
        stats.total_table_markers += len(_TABLE_RE.findall(sec_text))

        refs_here = len(_PLACEHOLDER_REF_RE.findall(sec_text))
        placeholder_count += refs_here

    if sections:
        stats.section_counts.append(len(sections))
    if total_body_words:
        stats.body_word_counts.append(total_body_words)
    if sections and all_titles_noise:
        stats.papers_with_no_section_titles += 1
    if has_imrad_title:
        stats.papers_with_any_imrad_title += 1
    if any(count > 1 for count in per_paper_titles.values()):
        stats.papers_with_duplicate_titles += 1
    if not paper_cite_markers:
        stats.papers_with_no_citations += 1
    if placeholder_count:
        stats.papers_with_placeholder_refs += 1
        stats.total_placeholder_refs += placeholder_count

    bib_entries = _iter_bib_entries(paper)
    _count_ref_entry_captions(paper, stats)

    for bib in bib_entries.values():
        stats.total_bib_entries += 1
        id_map = _extract_id_fields_from_bib(bib)

        if _extract_title_from_bib(bib):
            stats.bib_has_title += 1
        if _extract_bib_year(bib):
            stats.bib_has_year += 1
        if _has_any_external_id(id_map):
            stats.bib_has_any_id += 1
        else:
            stats.bib_no_unique_id += 1

        for field in _ID_FIELDS:
            if id_map.get(field):
                stats.bib_id_counter[field] += 1

    stats.total_cite_ref_ids += len(paper_cite_markers)
    paper_has_missing_cited_bib = False
    paper_has_unresolved_cited_refs = False

    for ref_id in paper_cite_markers:
        bib = bib_entries.get(ref_id)
        if not bib:
            stats.cited_ref_missing_bib += 1
            paper_has_missing_cited_bib = True
            continue

        id_map = _extract_id_fields_from_bib(bib)
        if _has_any_external_id(id_map):
            stats.cited_ref_has_any_id += 1
        else:
            stats.cited_ref_no_unique_id += 1
            paper_has_unresolved_cited_refs = True

        for field in _ID_FIELDS:
            if id_map.get(field):
                stats.cited_ref_id_counter[field] += 1

    if paper_has_missing_cited_bib:
        stats.papers_with_missing_cited_bib += 1
    if paper_has_unresolved_cited_refs:
        stats.papers_with_unresolved_cited_refs += 1


def _percentile(data: list[int | float], p: float) -> float:
    if not data:
        return 0.0
    ordered = sorted(data)
    idx = min(len(ordered) - 1, max(0, int(len(ordered) * p / 100) - 1))
    return ordered[idx]


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{100 * numerator / denominator:.1f}%"


def _status(ok: bool) -> str:
    return "OK" if ok else "WARN"


def print_report(s: Stats, n_requested: int, output_file: str | None) -> None:
    N = s.total
    B = s.total_bib_entries
    R = s.total_cite_ref_ids

    def avg(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    lines: list[str] = []
    add = lines.append

    add("=" * 88)
    add("  unarXive 2024 — Data Quality Report")
    add("=" * 88)
    add(f"  Papers analysed : {N:,}  (requested {n_requested:,})")
    add("")

    add("── 1. PAPER METADATA ─────────────────────────────────────────────────────────────")
    add(f"  Title present                : {s.has_title:>7,}  ({_pct(s.has_title, N)})")
    add(f"  Authors present              : {s.has_authors:>7,}  ({_pct(s.has_authors, N)})")
    add(f"  Year present                 : {s.has_year:>7,}  ({_pct(s.has_year, N)})")
    add(f"  Paper DOI present            : {s.has_doi:>7,}  ({_pct(s.has_doi, N)})")
    add(f"  Paper arXiv ID present       : {s.has_arxiv_id:>7,}  ({_pct(s.has_arxiv_id, N)})")
    add(f"  Categories present           : {s.has_categories:>7,}  ({_pct(s.has_categories, N)})")
    add("")

    add("── 2. ABSTRACT ───────────────────────────────────────────────────────────────────")
    add(f"  Has non-empty abstract       : {s.has_abstract:>7,}  ({_pct(s.has_abstract, N)})")
    add(f"  Empty / missing              : {s.abstract_empty:>7,}  ({_pct(s.abstract_empty, N)})")
    add("")

    add("── 3. NARRATIVE CONTENT ──────────────────────────────────────────────────────────")
    add(f"  Has body / sections text     : {s.has_body:>7,}  ({_pct(s.has_body, N)})")
    add(f"  Empty / missing text         : {s.body_empty:>7,}  ({_pct(s.body_empty, N)})")
    add(f"  Papers with sections object")
    add(f"    but no non-empty text      : {s.papers_with_sections_but_no_text:>7,}  ({_pct(s.papers_with_sections_but_no_text, N)})")
    add(f"  Papers with no section nodes : {s.papers_with_empty_sections:>7,}  ({_pct(s.papers_with_empty_sections, N)})")
    if s.body_word_counts:
        add(f"  Body word count mean         : {avg(s.body_word_counts):>9,.0f}")
        add(f"  Body word count p10/p50/p90  : {_percentile(s.body_word_counts,10):.0f} / {_percentile(s.body_word_counts,50):.0f} / {_percentile(s.body_word_counts,90):.0f}")
    add("")

    add("── 4. SECTION + SUBSECTION STRUCTURE ────────────────────────────────────────────")
    add(f"  Total section nodes found    : {s.total_sections:>7,}")
    add(f"  Total subsection nodes       : {s.total_subsections:>7,}")
    add(f"  Avg section nodes / paper    : {avg(s.section_counts):>9.1f}")
    if s.section_counts:
        add(f"  Section nodes p10/p50/p90    : {_percentile(s.section_counts,10):.0f} / {_percentile(s.section_counts,50):.0f} / {_percentile(s.section_counts,90):.0f}")
    add(f"  Sections w/ empty title      : {s.sections_with_empty_title:>7,}  ({_pct(s.sections_with_empty_title, s.total_sections)} of all section nodes)")
    add(f"  Sections w/ noise title      : {s.sections_with_noise_title:>7,}  ({_pct(s.sections_with_noise_title, s.total_sections)} of all section nodes)")
    add(f"  Sections w/ empty text       : {s.sections_with_empty_text:>7,}  ({_pct(s.sections_with_empty_text, s.total_sections)} of all section nodes)")
    add(f"  Sections w/ IMRAD match      : {s.sections_imrad_matched:>7,}  ({_pct(s.sections_imrad_matched, s.total_sections)} of all section nodes)")
    add(f"  Papers w/ >=1 IMRAD title    : {s.papers_with_any_imrad_title:>7,}  ({_pct(s.papers_with_any_imrad_title, N)})")
    add(f"  Papers w/ all titles bad     : {s.papers_with_no_section_titles:>7,}  ({_pct(s.papers_with_no_section_titles, s.has_body)} of papers with body)")
    add(f"  Papers w/ duplicate titles   : {s.papers_with_duplicate_titles:>7,}  ({_pct(s.papers_with_duplicate_titles, N)})")
    add("")
    add("  IMRAD bucket counts:")
    for bucket in ("abstract", "introduction", "background", "methods", "results", "discussion", "conclusion", "references", "appendix"):
        add(f"    {s.imrad_counter.get(bucket, 0):>7,}  {bucket}")
    add("")
    add("  Top 30 normalised section titles:")
    if s.section_title_counter:
        max_count = s.section_title_counter.most_common(1)[0][1]
        for title, count in s.section_title_counter.most_common(30):
            bar = "█" * min(40, int(40 * count / max(1, max_count)))
            add(f"    {count:>6,}  {bar:<40}  {title}")
    else:
        add("    none")
    add("")

    add("── 5. IN-TEXT MARKERS ────────────────────────────────────────────────────────────")
    add(f"  Total {{cite:...}} markers    : {s.total_cite_markers:>7,}")
    add(f"  Avg cite markers / paper      : {s.total_cite_markers / N:>9.1f}" if N else "  Avg cite markers / paper      : N/A")
    add(f"  Papers with NO citations      : {s.papers_with_no_citations:>7,}  ({_pct(s.papers_with_no_citations, N)})")
    add(f"  Total figure markers          : {s.total_figure_markers:>7,}")
    add(f"  Total table markers           : {s.total_table_markers:>7,}")
    add("")

    add("── 6. BIBLIOGRAPHY ID COVERAGE ───────────────────────────────────────────────────")
    add(f"  Total bib entries             : {B:>7,}")
    add(f"  Avg bib entries / paper       : {B / N:>9.1f}" if N else "  Avg bib entries / paper       : N/A")
    add(f"  Has title string              : {s.bib_has_title:>7,}  ({_pct(s.bib_has_title, B)} of entries)")
    add(f"  Has year                      : {s.bib_has_year:>7,}  ({_pct(s.bib_has_year, B)} of entries)")
    add(f"  Has any external ID           : {s.bib_has_any_id:>7,}  ({_pct(s.bib_has_any_id, B)} of entries)")
    add(f"  No unique external ID         : {s.bib_no_unique_id:>7,}  ({_pct(s.bib_no_unique_id, B)} of entries)")
    add("")
    add("  ID type availability across all bib entries:")
    for field in _ID_FIELDS:
        add(f"    {s.bib_id_counter.get(field, 0):>7,}  {field}")
    add("")

    add("── 7. CITED REF_ID -> UNIQUE ID RESOLUTION ──────────────────────────────────────")
    add(f"  Unique cited ref_ids          : {R:>7,}")
    add(f"  Cited ref_ids w/ any ID       : {s.cited_ref_has_any_id:>7,}  ({_pct(s.cited_ref_has_any_id, R)} of cited ref_ids)")
    add(f"  Cited ref_ids w/ no unique ID : {s.cited_ref_no_unique_id:>7,}  ({_pct(s.cited_ref_no_unique_id, R)} of cited ref_ids)")
    add(f"  Cited ref_ids missing in bib  : {s.cited_ref_missing_bib:>7,}  ({_pct(s.cited_ref_missing_bib, R)} of cited ref_ids)")
    add("")
    add("  Cited ref_ids mapped by identifier type:")
    for field in _ID_FIELDS:
        add(f"    {s.cited_ref_id_counter.get(field, 0):>7,}  {field}")
    add("")

    add("── 8. GENERAL ANOMALIES ──────────────────────────────────────────────────────────")
    add(f"  Papers with placeholder REF   : {s.papers_with_placeholder_refs:>7,}  ({_pct(s.papers_with_placeholder_refs, N)})")
    add(f"  Total placeholder REF tokens  : {s.total_placeholder_refs:>7,}")
    add(f"  Papers with missing cited bib : {s.papers_with_missing_cited_bib:>7,}  ({_pct(s.papers_with_missing_cited_bib, N)})")
    add(f"  Papers with unresolved cites  : {s.papers_with_unresolved_cited_refs:>7,}  ({_pct(s.papers_with_unresolved_cited_refs, N)})")
    add(f"  Ref entries with NO_CAPTION   : {s.ref_entries_no_caption:>7,}  ({_pct(s.ref_entries_no_caption, s.ref_entries_total)} of ref entries)")
    add("")

    add("── 9. DISCIPLINE BREAKDOWN (arXiv prefix) ────────────────────────────────────────")
    for cat, count in s.category_counter.most_common(15):
        add(f"    {count:>7,}  {cat}")
    add("")

    add("── 10. EVIGRAPH-R COMPATIBILITY (based on index.html architecture) ──────────────")
    retrieval_ready = s.has_body
    section_ready = s.has_body - s.papers_with_no_section_titles
    citation_ready = s.cited_ref_has_any_id
    add(f"  {_status(s.has_arxiv_id == N)}  Paper identifiers for `paper_id_arxiv`")
    add(f"      available on               : {s.has_arxiv_id:>7,}  ({_pct(s.has_arxiv_id, N)})")
    add(f"  {_status(s.has_abstract > 0)}  Abstract chunking")
    add(f"      non-empty abstracts        : {s.has_abstract:>7,}  ({_pct(s.has_abstract, N)})")
    add(f"  {_status(retrieval_ready == N)}  Retrieval chunking from body/sections")
    add(f"      papers with narrative text : {retrieval_ready:>7,}  ({_pct(retrieval_ready, N)})")
    add(f"  {_status(section_ready >= max(1, int(0.8 * max(1, s.has_body))))}  Section-aware routing / IMRAD filtering")
    add(f"      papers with usable titles  : {section_ready:>7,}  ({_pct(section_ready, s.has_body)} of papers with body)")
    add(f"  {_status(s.papers_with_any_imrad_title >= max(1, int(0.7 * max(1, s.has_body))))}  IMRAD-style title coverage")
    add(f"      papers with >=1 IMRAD hit  : {s.papers_with_any_imrad_title:>7,}  ({_pct(s.papers_with_any_imrad_title, s.has_body)} of papers with body)")
    add(f"  {_status(citation_ready >= max(1, int(0.7 * max(1, R))))}  Citation-hop expansion readiness")
    add(f"      cited refs with ext IDs    : {citation_ready:>7,}  ({_pct(citation_ready, R)} of cited ref_ids)")
    add("")

    add("── 11. PIPELINE SUMMARY ──────────────────────────────────────────────────────────")
    issues: list[str] = []

    if N and s.body_empty / N > 0.05:
        issues.append(f"WARN  {_pct(s.body_empty, N)} papers lack usable narrative text in `body_text`/`sections`")
    if s.has_body and s.papers_with_no_section_titles / s.has_body > 0.20:
        issues.append(f"WARN  {_pct(s.papers_with_no_section_titles, s.has_body)} of papers with text have no usable section titles")
    if s.total_sections and s.sections_with_empty_text / s.total_sections > 0.15:
        issues.append(f"INFO  {_pct(s.sections_with_empty_text, s.total_sections)} of section nodes have empty text")
    if B and s.bib_no_unique_id / B > 0.30:
        issues.append(f"WARN  {_pct(s.bib_no_unique_id, B)} of bib entries have no external identifier")
    if R and s.cited_ref_no_unique_id / R > 0.30:
        issues.append(f"WARN  {_pct(s.cited_ref_no_unique_id, R)} of cited ref_ids cannot map to a unique external work ID")
    if R and s.cited_ref_missing_bib / R > 0.01:
        issues.append(f"WARN  {_pct(s.cited_ref_missing_bib, R)} of cited ref_ids do not exist in `bib_entries`")
    if N and s.papers_with_placeholder_refs / N > 0.10:
        issues.append(f"INFO  {_pct(s.papers_with_placeholder_refs, N)} of papers contain unresolved placeholder references like `REF`")
    if N and (N - s.has_year) / N > 0.10:
        issues.append(f"INFO  {_pct(N - s.has_year, N)} of papers are missing year metadata")
    if N and s.papers_with_no_citations / N > 0.20:
        issues.append(f"INFO  {_pct(s.papers_with_no_citations, N)} of papers have zero in-text citation markers")

    if not issues:
        add("  OK    No major compatibility or data-quality risks detected at this sample size.")
    for issue in issues:
        add(f"  {issue}")

    add("")
    add("=" * 88)

    report_text = "\n".join(lines)
    print(report_text)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as handle:
            handle.write(report_text + "\n")
        print(f"\nReport saved -> {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="unarXive data quality analysis")
    parser.add_argument("--n", type=int, default=3000, help="Number of papers to analyse")
    parser.add_argument("--dataset", type=str, default="ines-besrour/unarxive_2024")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save the report")
    parser.add_argument("--verbose", action="store_true", help="Print progress every 500 analysed papers")
    parser.add_argument("--log-every", type=int, default=0, help="Print current paper info every N analysed papers (0 disables)")
    parser.add_argument("--scan-log-every", type=int, default=0, help="Print scan progress every N streamed rows, including skipped rows (0 disables)")
    args = parser.parse_args()

    profile = _runtime_profile()
    device = _preferred_device(profile)
    hf_token = _load_hf_token_from_env()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. pip install datasets", file=sys.stderr)
        sys.exit(1)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nunarXive Data Quality Analysis — {now_str}\n", flush=True)
    print(f"Runtime profile: {profile} (preferred device: {device})", flush=True)
    print(f"HF token loaded from env: {'yes' if hf_token else 'no'}", flush=True)
    print(f"Streaming {args.n:,} papers from {args.dataset} …", flush=True)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    print(f"Dataset stream initialised at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("Processing papers …\n", flush=True)

    stats = Stats()
    skipped = 0
    rows_seen = 0
    fetch_started_at = perf_counter()
    first_row_announced = False

    for row in ds:
        rows_seen += 1
        if not first_row_announced:
            wait_s = perf_counter() - fetch_started_at
            print(f"  First streamed row received after {wait_s:.1f}s", flush=True)
            first_row_announced = True

        if stats.total >= args.n:
            break

        raw_jsonl = str(row.get("jsonl") or "").strip()
        if not raw_jsonl:
            skipped += 1
            if args.scan_log_every > 0 and rows_seen % args.scan_log_every == 0:
                print(
                    f"  [scan {rows_seen:,}] analysed={stats.total:,} skipped={skipped:,} current_row=empty-jsonl",
                    flush=True,
                )
            continue

        paper = _first_jsonl(raw_jsonl)
        if paper is None:
            skipped += 1
            if args.scan_log_every > 0 and rows_seen % args.scan_log_every == 0:
                print(
                    f"  [scan {rows_seen:,}] analysed={stats.total:,} skipped={skipped:,} current_row=unparseable-jsonl",
                    flush=True,
                )
            continue

        analyse_paper(paper, stats)

        current_paper_id = str(
            (paper.get("metadata") or {}).get("id")
            or paper.get("paper_id")
            or paper.get("_id")
            or "unknown"
        ).strip()
        current_title = str(
            (paper.get("metadata") or {}).get("title")
            or paper.get("title")
            or ""
        ).strip()

        if args.verbose and stats.total % 500 == 0:
            print(f"  … {stats.total:,} / {args.n:,}", flush=True)
        if args.log_every > 0 and stats.total % args.log_every == 0:
            title_suffix = f" | {current_title[:100]}" if current_title else ""
            print(
                f"  [{stats.total:,}/{args.n:,}] paper_id={current_paper_id}{title_suffix}",
                flush=True,
            )
        if args.scan_log_every > 0 and rows_seen % args.scan_log_every == 0:
            print(
                f"  [scan {rows_seen:,}] analysed={stats.total:,} skipped={skipped:,} last_paper_id={current_paper_id}",
                flush=True,
            )

    if skipped:
        print(f"  (skipped {skipped} empty/unparseable rows)", flush=True)

    print(f"Done — analysed {stats.total:,} papers.\n", flush=True)
    print_report(stats, args.n, args.out)


if __name__ == "__main__":
    main()
