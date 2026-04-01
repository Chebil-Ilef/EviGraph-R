from __future__ import annotations

import re
from typing import Optional
from indexing.utils.models import NormalizedPaper, NormalizedSection

_CITE_RE = re.compile(r"\{\{cite:([0-9a-f]+)\}\}")
_REF_MARKER_RE = re.compile(r"\{\{(?:figure|table):([0-9a-f\-]+)\}\}")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MULTI_SPACE_RE = re.compile(r"  +")

_YEAR_MIN = 1900
_YEAR_MAX = 2100

_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
)


def normalize_doi(raw: str) -> str:
    raw = (raw or "").strip()
    for prefix in _DOI_PREFIXES:
        if raw.startswith(prefix):
            return raw[len(prefix):]
    return raw


def build_ref_caption_map(ref_entries: dict) -> dict[str, str]:
    caption_map: dict[str, str] = {}
    for uid, entry in (ref_entries or {}).items():
        if not isinstance(entry, dict):
            caption_map[uid] = ""
            continue
        caption_map[uid] = (entry.get("caption") or "").strip()
    return caption_map


def build_citation_lookup(bib_entries: dict) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for ref_id, bib in (bib_entries or {}).items():
        if not isinstance(bib, dict):
            lookup[ref_id] = {"source_ref_id": ref_id}
            continue

        ids = bib.get("ids") or {}
        entry = {"source_ref_id": ref_id}
        doi = normalize_doi(ids.get("doi") or "")
        if doi:
            entry["doi"] = doi
        if ids.get("open_alex_id"):
            entry["openalex_id"] = ids["open_alex_id"]
        if ids.get("arxiv_id"):
            entry["arxiv_id"] = ids["arxiv_id"]
        # Extract raw citation string for resolution fallback
        if bib.get("bib_entry_raw"):
            entry["raw"] = (bib.get("bib_entry_raw") or "").strip()
        lookup[ref_id] = entry
    return lookup


def clean_ref_markers(text: str, ref_caption_map: dict[str, str]) -> str:
    def _replace(match: re.Match[str]) -> str:
        full_match = match.group(0)
        uuid = match.group(1)
        label = "Figure" if "figure:" in full_match else "Table"
        caption = ref_caption_map.get(uuid, "")
        return f"[{label}: {caption}]" if caption else ""

    return _REF_MARKER_RE.sub(_replace, text)


def process_text(
    text: str,
    citation_lookup: dict[str, dict],
) -> tuple[str, list[dict]]:
    parts: list[str] = []
    cite_spans: list[dict] = []
    offset_shift = 0
    last_end = 0

    for match in _CITE_RE.finditer(text):
        ref_id = match.group(1)
        orig_start, orig_end = match.start(), match.end()
        parts.append(text[last_end:orig_start])

        pos = orig_start + offset_shift
        info = citation_lookup.get(ref_id) or {"source_ref_id": ref_id}
        cite_spans.append(
            {
                "start": pos,
                "end": pos,
                "source_ref_id": info.get("source_ref_id") or ref_id,
                "doi": info.get("doi") or "",
                "openalex_id": info.get("openalex_id") or "",
                "arxiv_id": info.get("arxiv_id") or "",
                "title": info.get("title") or "",
                "raw": info.get("raw") or "",
            }
        )

        offset_shift -= (orig_end - orig_start)
        last_end = orig_end

    parts.append(text[last_end:])
    return "".join(parts), cite_spans


def _extract_year(metadata: dict) -> Optional[int]:
    for version in metadata.get("versions") or []:
        match = _YEAR_RE.search(version.get("created") or "")
        if match:
            year = int(match.group())
            return year if _YEAR_MIN <= year <= _YEAR_MAX else None
    update_date = metadata.get("update_date") or ""
    match = _YEAR_RE.match(update_date)
    if match:
        year = int(match.group())
        return year if _YEAR_MIN <= year <= _YEAR_MAX else None
    return None


def _extract_authors(metadata: dict) -> list[str]:
    parsed = metadata.get("authors_parsed") or []
    if parsed:
        authors: list[str] = []
        for entry in parsed:
            last = (entry[0] if len(entry) > 0 else "").strip()
            first = (entry[1] if len(entry) > 1 else "").strip()
            name = f"{first} {last}".strip() if first else last
            if name:
                authors.append(name)
        return authors

    raw = metadata.get("authors") or ""
    return [author.strip() for author in re.split(r"[,;]", raw) if author.strip()]


def _extract_categories(metadata: dict) -> list[str]:
    categories = metadata.get("categories") or ""
    if isinstance(categories, list):
        return categories
    return categories.split()


def _safe_int(value) -> Optional[int]:

    if value is None:
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if -(2**31) <= v <= (2**31 - 1) else None


def build_paper_meta(paper: dict) -> dict:
    metadata = paper.get("metadata") or {}
    return {
        "doi": metadata.get("doi") or "",
        "title": metadata.get("title") or "",
        "authors": _extract_authors(metadata),
        "categories": _extract_categories(metadata),
        "year": _extract_year(metadata),
        "cited_by_count": _safe_int(metadata.get("cited_by_count")),
        "language": metadata.get("language"),
        "discipline": metadata.get("discipline"),
    }


def normalize_paper(paper: dict) -> NormalizedPaper:
    sections: list[NormalizedSection] = []
    for title, section in (paper.get("sections") or {}).items():
        if not isinstance(section, dict):
            continue
        raw_title = (title or "").strip()
        clean_title = "Body" if not raw_title or raw_title.lower() == "null" else raw_title
        sections.append(
            NormalizedSection(
                title=clean_title,
                text=(section.get("text") or "").strip(),
            )
        )

    meta = build_paper_meta(paper)
    return NormalizedPaper(
        paper_id=paper.get("paper_id") or "",
        paper_doi=meta["doi"],
        meta=meta,
        abstract_text=((paper.get("abstract") or {}).get("text") or "").strip(),
        sections=sections,
        citations_by_ref=build_citation_lookup(paper.get("bib_entries") or {}),
        ref_captions=build_ref_caption_map(paper.get("ref_entries") or {}),
    )


def make_embed_text(section_title: Optional[str], text: str) -> str:
    clean = _MULTI_SPACE_RE.sub(" ", text).strip()
    return f"{section_title}: {clean}" if section_title else clean


def make_uid(paper_id: str, section_label: str, text: str) -> str:
    import hashlib

    raw = f"{paper_id}\x00{section_label}\x00{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()
