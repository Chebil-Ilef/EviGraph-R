from __future__ import annotations
import re
from typing import Optional
import hashlib
from evigraph.indexing.utils.models import NormalizedPaper, NormalizedSection

_CITE_RE = re.compile(r"\{\{cite:([0-9a-f]+)\}\}")
_REF_MARKER_RE = re.compile(r"\{\{(?:figure|table):([0-9a-f\-]+)\}\}")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MULTI_SPACE_RE = re.compile(r"  +")
_SENTENCE_BREAK_RE = re.compile(r"[.!?]")
_ORPHAN_CITE_PUNCT_RE = re.compile(r"(\s*,)(\s*,)+")
_SENTENCE_START_RE = re.compile(r'[\s"\')\]]*([A-Z0-9(\[])')

# Patterns for sections to skip during indexing (acknowledgments, references, etc.)
_SKIP_SECTION_PATTERNS = [
    r"^acknowledg(e?ments?|ements?)$",
    r"^references$",
    r"^bibliography$",
    r"^index$",
    r"^table of contents$",
    r"^list of (figures|tables|algorithms)$",
]

_YEAR_MIN = 1900
_YEAR_MAX = 2100

_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
)

_COMMON_ABBREVIATIONS = {
    "al.",
    "approx.",
    "art.",
    "assoc.",
    "cf.",
    "chap.",
    "co.",
    "corp.",
    "dept.",
    "dr.",
    "e.g.",
    "eq.",
    "eqs.",
    "esp.",
    "et al.",
    "etc.",
    "fig.",
    "figs.",
    "i.e.",
    "inc.",
    "jr.",
    "mr.",
    "mrs.",
    "ms.",
    "no.",
    "nos.",
    "phys.",
    "pp.",
    "prof.",
    "ref.",
    "refs.",
    "rev.",
    "sec.",
    "secs.",
    "st.",
    "supp.",
    "vol.",
    "vs.",
}


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
    pending_citations: list[tuple[int, str]] = []
    offset_shift = 0
    last_end = 0

    for match in _CITE_RE.finditer(text):
        ref_id = match.group(1)
        orig_start, orig_end = match.start(), match.end()
        parts.append(text[last_end:orig_start])

        pos = orig_start + offset_shift
        pending_citations.append((pos, ref_id))

        offset_shift -= (orig_end - orig_start)
        last_end = orig_end

    parts.append(text[last_end:])
    cleaned_text = "".join(parts)
    sentence_spans = _split_sentence_spans(cleaned_text)
    cite_spans: list[dict] = []

    for pos, ref_id in pending_citations:
        sentence_start, sentence_end = _find_sentence_span(sentence_spans, pos, len(cleaned_text))
        info = citation_lookup.get(ref_id) or {"source_ref_id": ref_id}
        cite_spans.append(
            {
                "start": sentence_start,
                "end": sentence_end,
                "source_ref_id": info.get("source_ref_id") or ref_id,
                "doi": info.get("doi") or "",
                "openalex_id": info.get("openalex_id") or "",
                "arxiv_id": info.get("arxiv_id") or "",
                "raw": info.get("raw") or "",
            }
        )

    return cleaned_text, cite_spans


def _split_sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    idx = 0

    while idx < len(text):
        match = _SENTENCE_BREAK_RE.search(text, idx)
        if match is None:
            break

        punct_idx = match.start()
        if _looks_like_sentence_boundary(text, punct_idx):
            end = punct_idx + 1
            while end < len(text) and text[end] in '\'"”’)]}':
                end += 1

            trimmed_start = _skip_leading_whitespace(text, start)
            trimmed_end = _trim_trailing_whitespace(text, end)
            if trimmed_start < trimmed_end:
                spans.append((trimmed_start, trimmed_end))
            start = end
            idx = end
            continue

        idx = punct_idx + 1

    trimmed_start = _skip_leading_whitespace(text, start)
    trimmed_end = _trim_trailing_whitespace(text, len(text))
    if trimmed_start < trimmed_end:
        spans.append((trimmed_start, trimmed_end))

    return spans


def _find_sentence_span(
    spans: list[tuple[int, int]],
    anchor: int,
    text_length: int,
) -> tuple[int, int]:
    if not spans:
        return (0, text_length)

    clamped_anchor = max(0, min(anchor, text_length))
    for start, end in spans:
        if start <= clamped_anchor <= end:
            return (start, end)

    if clamped_anchor < spans[0][0]:
        return spans[0]
    return spans[-1]


def _looks_like_sentence_boundary(text: str, punct_idx: int) -> bool:
    char = text[punct_idx]
    if char == "." and punct_idx > 0 and punct_idx + 1 < len(text):
        if text[punct_idx - 1].isdigit() and text[punct_idx + 1].isdigit():
            return False

    token = _token_before_punctuation(text, punct_idx).lower()
    if token in _COMMON_ABBREVIATIONS:
        return False

    if re.fullmatch(r"(?:[a-z]\.){2,}", token):
        return False

    if re.fullmatch(r"[a-z]\.", token):
        return False

    tail = text[punct_idx + 1 :]
    if not tail:
        return True

    next_start = _SENTENCE_START_RE.match(tail)
    return next_start is not None


def _token_before_punctuation(text: str, punct_idx: int) -> str:
    start = punct_idx
    while start > 0 and text[start - 1] not in " \t\r\n([{":
        start -= 1
    return text[start : punct_idx + 1]


def _skip_leading_whitespace(text: str, start: int) -> int:
    while start < len(text) and text[start].isspace():
        start += 1
    return start


def _trim_trailing_whitespace(text: str, end: int) -> int:
    while end > 0 and text[end - 1].isspace():
        end -= 1
    return end


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


def _should_skip_section(title: str) -> bool:
    normalized = re.sub(r"\s+", " ", (title or "").strip().lower())
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in _SKIP_SECTION_PATTERNS)


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
        "language": metadata.get("language"),
        "discipline": metadata.get("discipline"),
    }


def normalize_paper(paper: dict) -> NormalizedPaper:
    sections: list[NormalizedSection] = []
    for title, section in (paper.get("sections") or {}).items():
        if not isinstance(section, dict):
            continue
        raw_title = (title or "").strip()
        
        # Skip sections matching skip patterns (acknowledgments, references, bibliography, etc.)
        if _should_skip_section(raw_title):
            continue
        
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


def make_embed_text(text: str) -> str:
    clean = _ORPHAN_CITE_PUNCT_RE.sub(",", text)
    clean = _MULTI_SPACE_RE.sub(" ", clean).strip()
    return clean


def make_uid(
    paper_id: str,
    section_label: str,
    text: str,
    *,
    chunk_index: int | None = None,
) -> str:

    index_part = "" if chunk_index is None else str(chunk_index)
    raw = f"{paper_id}\x00{section_label}\x00{index_part}\x00{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()
