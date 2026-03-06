
from __future__ import annotations
import json
import re
from typing import Optional


# Regex constants

_CITE_RE = re.compile(r"\{\{cite:([0-9a-f]+)\}\}")

# {{figure:uuid}} or {{table:uuid}}
_REF_MARKER_RE = re.compile(r"\{\{(?:figure|table):([0-9a-f\-]+)\}\}")

# DOI mining from raw bib strings
_DOI_LABEL_RE = re.compile(r"\bdoi:\s*(10\.\d{4,}/\S+)", re.IGNORECASE)
_DOI_URL_RE   = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,}/\S+)", re.IGNORECASE)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


# Figure / table reference-marker helpers

def build_ref_caption_map(ref_entries: dict) -> dict[str, str]:
    """Return {uuid → caption} from a paper's ref_entries dict."""
    
    caption_map: dict[str, str] = {}
    for uid, entry in (ref_entries or {}).items():
        if not entry or not isinstance(entry, dict):
            caption_map[uid] = ""
            continue
        caption_map[uid] = (entry.get("caption") or "").strip()
    return caption_map


def clean_ref_markers(text: str, ref_caption_map: dict[str, str]) -> str:
    """
    Replace ``{{figure:uuid}}`` / ``{{table:uuid}}`` markers in *text*.

    - uuid has a non-empty caption  → ``[Figure: <caption>]`` / ``[Table: <caption>]``
    - no caption                    → remove marker entirely
    """
    def _replace(m: re.Match) -> str:  # type: ignore[type-arg]
        full_match = m.group(0)
        uuid       = m.group(1)
        label      = "Figure" if "figure:" in full_match else "Table"
        caption    = ref_caption_map.get(uuid, "")
        return f"[{label}: {caption}]" if caption else ""

    return _REF_MARKER_RE.sub(_replace, text)


# DOI / arXiv-id resolution

def _parse_doi_from_raw(raw: str) -> str:
    """
    Last-resort DOI extraction from a raw bibliography string.

    Tries in order:
      1. ``doi: 10.xxx/yyy``  (explicit BibTeX label)
      2. ``https://doi.org/10.xxx/yyy``  (doi.org URL)

    Returns "" when nothing is found.
    """
    for pattern in (_DOI_LABEL_RE, _DOI_URL_RE):
        m = pattern.search(raw)
        if m:
            doi = m.group(1).rstrip(".,;:")
            if doi:
                return doi
    return ""


def build_doi_map(bib_entries: dict) -> dict[str, str]:
    """
    Build ``{ref_id → doi_string}`` from the paper's bib_entries dict.

    Resolution order per entry:
      1. ids.doi  (URL-prefixes normalised)
      2. _parse_doi_from_raw(bib_entry_raw)
      3. ""  if nothing found
    """
    doi_map: dict[str, str] = {}
    for ref_id, bib in bib_entries.items():
        if not bib or not isinstance(bib, dict):
            doi_map[ref_id] = ""
            continue

        ids     = bib.get("ids") or {}
        raw_doi = ids.get("doi") or ""

        for prefix in (
            "https://doi.org/",
            "http://doi.org/",
            "https://dx.doi.org/",
            "http://dx.doi.org/",
        ):
            if raw_doi.startswith(prefix):
                raw_doi = raw_doi[len(prefix):]
                break

        if raw_doi:
            doi_map[ref_id] = raw_doi
            continue

        mined = _parse_doi_from_raw(bib.get("bib_entry_raw") or "")
        doi_map[ref_id] = mined

    return doi_map


def build_arxiv_id_map(bib_entries: dict) -> dict[str, str]:

    arxiv_map: dict[str, str] = {}
    for ref_id, bib in (bib_entries or {}).items():
        if not bib or not isinstance(bib, dict):
            arxiv_map[ref_id] = ""
            continue
        ids = bib.get("ids") or {}
        arxiv_map[ref_id] = ids.get("arxiv_id") or ""
    return arxiv_map


# Citation-marker processing

def process_text(
    text: str,
    doi_map: dict[str, str],
    arxiv_id_map: dict[str, str] | None = None,
) -> tuple[str, list[dict]]:
    """
    Remove every ``{{cite:ref_id}}`` from *text* and record where each
    Give a start and end to the citation.
    """
    parts: list[str] = []
    cite_spans: list[dict] = []
    offset_shift = 0
    last_end = 0

    for m in _CITE_RE.finditer(text):
        ref_id               = m.group(1)
        orig_start, orig_end = m.start(), m.end()

        parts.append(text[last_end:orig_start])

        # Position in the output string where this citation sat
        pos = orig_start + offset_shift

        cite_spans.append({
            "start":    pos,
            "end":      pos,   # zero-length: marker removed, position preserved
            "doi":      doi_map.get(ref_id, ""),
            "arxiv_id": (arxiv_id_map or {}).get(ref_id, ""),
        })

        offset_shift -= (orig_end - orig_start)   # marker is gone
        last_end = orig_end

    parts.append(text[last_end:])
    return "".join(parts), cite_spans


# Paper metadata extraction

def _extract_year(metadata: dict) -> Optional[int]:
    for version in metadata.get("versions") or []:
        m = _YEAR_RE.search(version.get("created") or "")
        if m:
            return int(m.group())
    date_str = metadata.get("update_date") or ""
    m = _YEAR_RE.match(date_str)
    return int(m.group()) if m else None


def _extract_authors(metadata: dict) -> list[str]:
    parsed = metadata.get("authors_parsed") or []
    if parsed:
        names = []
        for entry in parsed:
            last  = (entry[0] if len(entry) > 0 else "").strip()
            first = (entry[1] if len(entry) > 1 else "").strip()
            name  = f"{first} {last}".strip() if first else last
            if name:
                names.append(name)
        return names
    raw: str = metadata.get("authors") or ""
    return [a.strip() for a in re.split(r"[,;]", raw) if a.strip()]


def _extract_categories(metadata: dict) -> list[str]:
    cats = metadata.get("categories") or ""
    if isinstance(cats, list):
        return cats
    return cats.split()


def build_paper_meta(paper: dict) -> dict:

    meta = paper.get("metadata") or {}
    return {
        "doi":            meta.get("doi") or "",
        "title":          meta.get("title") or "",
        "authors":        _extract_authors(meta),
        "categories":     _extract_categories(meta),
        "year":           _extract_year(meta),
        "cited_by_count": meta.get("cited_by_count"),
        "language":       meta.get("language"),
        "discipline":     meta.get("discipline"),
    }


# JSONL loader

def load_paper_from_batch_line(line: str) -> dict:

    outer = json.loads(line)
    return json.loads(outer["jsonl"])
