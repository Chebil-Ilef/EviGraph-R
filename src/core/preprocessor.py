
from __future__ import annotations
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from src.utils.resolve_title import resolve_title


# Regex constants

_CITE_RE = re.compile(r"\{\{cite:([0-9a-f]+)\}\}")

# {{figure:uuid}} or {{table:uuid}}
_REF_MARKER_RE = re.compile(r"\{\{(?:figure|table):([0-9a-f\-]+)\}\}")

# DOI mining from raw bib strings
_DOI_LABEL_RE = re.compile(r"\bdoi:\s*(10\.\d{4,}/\S+)", re.IGNORECASE)
_DOI_URL_RE   = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,}/\S+)", re.IGNORECASE)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

_TITLE_QUOTED_RE = re.compile(r'["\u201c]([^"\u201d]{15,})["\u201d]')

_CITE_DIGIT_RE = re.compile(
    r'\d+[(:!]\d'           # vol(issue) / colon-page
    r'|\d{2,}\s*,\s*\d'    # number, number  (volume, page)
    r'|\(\d{4}\)'           # (year) in parentheses
    r'|\w{2,}:\d'           # word:digit  (arXiv:1707, doi:10)
)

_AUTHOR_INITIAL_RE = re.compile(r',\s*[A-Z](?:\.|$)|\band\s+[A-Z](?:\.|$)')

_ARXIV_RAW_RE = re.compile(r"(?:arxiv[^\d]*)?([0-9]{4}\.[0-9]{4,5}|(?:hep-ph|hep-th|hep-lat|hep-ex|gr-qc|quant-ph|astro-ph|cond-mat|nucl-th|nucl-ex|math-ph|math|cs|q-bio|q-fin|stat|eess|econ|physics|nlin)/[0-9]{7})(?:v\d+)?", re.IGNORECASE)

_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
)


# DOI normalisation 

def normalize_doi(raw: str) -> str:

    raw = (raw or "").strip()
    for prefix in _DOI_PREFIXES:
        if raw.startswith(prefix):
            return raw[len(prefix):]
    return raw


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
 
    for pattern in (_DOI_LABEL_RE, _DOI_URL_RE):
        m = pattern.search(raw)
        if m:
            doi = m.group(1).rstrip(".,;:")
            if doi:
                return doi
    return ""


def _extract_title_from_raw(raw: str) -> str:

    if not raw:
        return ""

    # quoted title
    m = _TITLE_QUOTED_RE.search(raw)
    if m:
        return m.group(1).strip()

    # structural segment scan
    for seg in re.split(r'\.\s+', raw):
        seg = seg.strip()
        if not seg or not seg[0].isupper():
            continue
        if len(seg) < 20:
            continue
        if _CITE_DIGIT_RE.search(seg):
            continue
        if _AUTHOR_INITIAL_RE.search(seg):
            continue
        return seg

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
        raw_doi = normalize_doi(ids.get("doi") or "")

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


# Work-id resolution (KG node identity)

def _unresolved_work(ref_id: str, bib: dict | None) -> dict:
    return {
        "work_id":       f"unresolved:{ref_id}",
        "doi":           "",
        "openalex_id":   "",
        "arxiv_id":      "",
        "bib_entry_raw": (bib or {}).get("bib_entry_raw") or "",
    }


def build_work_id_map(bib_entries: dict, *, enrich: bool = False) -> dict[str, dict]:
    """
    Build ``{ref_id → work_info}`` for every bibliography entry.

    Fast offline pass (always):
      1. ``doi:<bare_doi>``      — ids.doi or mined from bib_entry_raw
      2. ``openalex:<W-id>``     — ids.open_alex_id
      3. ``arxiv:<arxiv_id>``    — ids.arxiv_id
      4. ```resolve_title(title, ref_id)``  — (OpenAlex → Crossref → arXiv)
      5. Fallback ``unresolved:<ref_id>`` with bib_entry_raw 

    Each value dict: work_id, doi, openalex_id, arxiv_id.
    Unresolved entries also carry bib_entry_raw for later enrichment.
    """
    result:    dict[str, dict] = {}
    needs_api: dict[str, dict] = {}   # entries that require an API call

    for ref_id, bib in (bib_entries or {}).items():
        if not bib or not isinstance(bib, dict):
            result[ref_id] = _unresolved_work(ref_id, bib)
            continue

        ids         = bib.get("ids") or {}
        doi         = normalize_doi(ids.get("doi") or "")
        if not doi:
            doi = _parse_doi_from_raw(bib.get("bib_entry_raw") or "")
        openalex_id = ids.get("open_alex_id") or ""
        arxiv_id    = ids.get("arxiv_id") or ""
        if not arxiv_id:
            raw_entry = bib.get("bib_entry_raw") or ""
            m = _ARXIV_RAW_RE.search(raw_entry)
            if m:
                arxiv_id = m.group(1)

        if doi:
            result[ref_id] = {
                "work_id":     f"doi:{doi}",
                "doi":         doi,
                "openalex_id": openalex_id,
                "arxiv_id":    arxiv_id,
            }
        elif openalex_id:
            oa_work = openalex_id.rstrip("/").split("/")[-1]
            result[ref_id] = {
                "work_id":     f"openalex:{oa_work}",
                "doi":         "",
                "openalex_id": openalex_id,
                "arxiv_id":    arxiv_id,
            }
        elif arxiv_id:
            result[ref_id] = {
                "work_id":     f"arxiv:{arxiv_id}",
                "doi":         "",
                "openalex_id": "",
                "arxiv_id":    arxiv_id,
            }
        else:
            needs_api[ref_id] = bib

    # parallel API resolution
    if needs_api:
        def _resolve_one(ref_id: str, bib: dict) -> tuple[str, dict]:
            raw_bib = bib.get("bib_entry_raw") or ""
            title   = _extract_title_from_raw(raw_bib)
            try:
                resolved = resolve_title(title, raw_bib, ref_id)
            except Exception:
                return ref_id, _unresolved_work(ref_id, bib)
            if resolved and not resolved.get("work_id", "").startswith("unresolved"):
                return ref_id, resolved
            return ref_id, _unresolved_work(ref_id, bib)

        max_workers = 2
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_resolve_one, rid, bib): rid
                for rid, bib in needs_api.items()
            }
            for fut in as_completed(futures):
                rid, info = fut.result()
                result[rid] = info

    # log all resolved IDs with their raw bibliography entries to manually verify on a small set
    try:
        import csv
        import os
        file_path = "_data/debug/resolved.csv"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_exists = os.path.exists(file_path)
        with open(file_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists or os.stat(file_path).st_size == 0:
                writer.writerow(["work_id", "bib_entry_raw"])
            for ref_id, info in result.items():
                w_id = info.get("work_id", "")
                if w_id and not w_id.startswith("unresolved:"):
                    bib = bib_entries.get(ref_id, {})
                    raw = (bib.get("bib_entry_raw") if isinstance(bib, dict) else "") or ""
                    if raw:
                        writer.writerow([w_id, raw])
    except Exception as e:
        print(f"Error writing to resolved.csv: {e}")

    return result


# Citation-marker processing

def process_text(
    text: str,
    work_id_map: dict[str, dict],
) -> tuple[str, list[dict]]:
    """
    Remove every ``{{cite:ref_id}}`` from *text* and record where each
    citation was, attaching the resolved work identity from *work_id_map*.
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

        info    = work_id_map.get(ref_id) or {}
        work_id = info.get("work_id") or f"unresolved:{ref_id}"
        span: dict = {
            "start":       pos,
            "end":         pos,   # zero-length: marker removed, position preserved
            "work_id":     work_id,
            "doi":         info.get("doi")         or "",
            "openalex_id": info.get("openalex_id") or "",
            "arxiv_id":    info.get("arxiv_id")    or "",
        }
        if work_id.startswith("unresolved:"):
            span["bib_entry_raw"] = info.get("bib_entry_raw") or ""
        cite_spans.append(span)

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
    return json.loads(line)
