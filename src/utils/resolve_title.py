
from __future__ import annotations
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional

logger = logging.getLogger(__name__)

# Configuration
MATCH_THRESHOLD: float = 0.85   # Jaccard score required to accept a candidate
LENGTH_RATIO_CAP = 1.5          # candidate title must be ≤ query length × this ratio to count for containment score
_OPENALEX_ROWS:  int   = 3      # candidates to fetch per API call
_CROSSREF_ROWS:  int   = 3
_ARXIV_ROWS:     int   = 3
_TIMEOUT:        int   = 10     # seconds per HTTP call
USER_AGENT = "scholarly-indexer/1.0 (research project)"
_ATOM_NS   = "http://www.w3.org/2005/Atom"


# Title normalisation & scoring
_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def _token_set(title: str) -> set[str]:

    t = _PUNCT_RE.sub(" ", title.lower())
    return set(_SPACE_RE.sub(" ", t).strip().split())


def title_score(query: str, candidate: str) -> float:
    """
    max( Jaccard, query-containment* ).

    - Jaccard      = |A ∩ B| / |A ∪ B|   — symmetric noise in both directions
    - Containment* = |A ∩ B| / |A|        — query words fully present in candidate
                     only counted when |B| ≤ |A| × LENGTH_RATIO_CAP (default 1.5)
                     → handles "BERT: … Transformers" matching the full title
                       "BERT: … Transformers for Language Understanding" (7 → 10 tokens)
                     → rejects unrelated longer papers that share a few keywords

    Returns 0.0 when either set is empty.
    """
    
    a, b = _token_set(query), _token_set(candidate)
    if not a or not b:
        return 0.0
    inter       = len(a & b)
    jaccard     = inter / len(a | b)
    containment = (inter / len(a)) if len(b) <= len(a) * LENGTH_RATIO_CAP else 0.0
    return max(jaccard, containment)


# HTTP helper

def _get_json(url: str, params: dict) -> Optional[dict]:

    full_url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        full_url,
        headers={"User-Agent": USER_AGENT},
    )
    logger.debug("GET %s", full_url[:140])
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        logger.warning("HTTP %s for %s", exc.code, full_url[:100])
        return None
    except Exception as exc:
        logger.warning("Request failed (%s): %s", type(exc).__name__, url)
        return None


def _get_xml(url: str, params: dict) -> Optional[ET.Element]:

    full_url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        full_url,
        headers={"User-Agent": USER_AGENT},
    )
    logger.debug("GET %s", full_url[:140])
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return ET.fromstring(resp.read())
    except urllib.error.HTTPError as exc:
        logger.warning("HTTP %s for %s", exc.code, full_url[:100])
        return None
    except Exception as exc:
        logger.warning("Request failed (%s): %s", type(exc).__name__, url)
        return None


# Per-API resolvers

def _try_openalex(title: str) -> Optional[dict]:

    logger.debug("OpenAlex search: %r", title[:80])
    data = _get_json(
        "https://api.openalex.org/works",
        {
            "search":   title,
            "select":   "id,title,doi,ids",
            "per-page": str(_OPENALEX_ROWS),
        },
    )
    if data is None:
        return None

    for item in (data.get("results") or []):
        candidate = item.get("title") or ""
        score = title_score(title, candidate)
        if score < MATCH_THRESHOLD:
            logger.debug("  OpenAlex no match (score=%.2f): %r", score, candidate[:60])
            continue
        logger.debug("  OpenAlex matched (score=%.2f): %r", score, candidate[:60])

        # --- extract identifiers ---
        oa_url   = item.get("id") or ""                        # https://openalex.org/W123
        oa_w_id  = oa_url.rstrip("/").split("/")[-1]           # W123

        raw_doi  = item.get("doi") or ""                       # https://doi.org/10.xxx/yyy
        doi      = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", raw_doi)

        ids      = item.get("ids") or {}
        arxiv_raw = ids.get("arxiv") or ""                     # https://arxiv.org/abs/1706.03762
        arxiv_id  = arxiv_raw.rstrip("/").split("/")[-1] if arxiv_raw else ""

        # --- choose primary id ---
        if doi:
            return {
                "work_id":     f"doi:{doi}",
                "id_source":   "doi",
                "doi":         doi,
                "openalex_id": oa_url,
                "arxiv_id":    arxiv_id,
            }
        if oa_w_id:
            return {
                "work_id":     f"openalex:{oa_w_id}",
                "id_source":   "openalex",
                "doi":         "",
                "openalex_id": oa_url,
                "arxiv_id":    arxiv_id,
            }

    return None   # no match above threshold


def _try_crossref(title: str, year: Optional[str] = None) -> Optional[dict]:

    logger.debug("Crossref title search: %r (year=%s)", title[:80], year)
    params: dict = {
        "query.title": title,
        "rows":        str(_CROSSREF_ROWS),
        "select":      "DOI,title",
    }
    if year:
        params["filter"] = f"from-pub-date:{year},until-pub-date:{year}"

    data = _get_json("https://api.crossref.org/works", params)
    if data is None:
        return None

    items = (data.get("message") or {}).get("items") or []
    for item in items:
        titles    = item.get("title") or []
        candidate = titles[0] if titles else ""
        score = title_score(title, candidate)
        if score < MATCH_THRESHOLD:
            logger.debug("  Crossref no match (score=%.2f): %r", score, candidate[:60])
            continue
        logger.debug("  Crossref matched (score=%.2f): %r", score, candidate[:60])

        doi = item.get("DOI") or ""
        if doi:
            return {
                "work_id":     f"doi:{doi}",
                "id_source":   "doi",
                "doi":         doi,
                "openalex_id": "",
                "arxiv_id":    "",
            }

    return None   # no match above threshold


def _try_crossref_bibliographic(raw: str, title_hint: str = "") -> Optional[dict]:
    """
    Submit the full raw bib string to Crossref ``query.bibliographic``.

    Crossref parses the free-text internally (authors, title, journal, year).
    This is the most powerful approach for journal-style references that have
    no structured IDs in the dataset.

    If *title_hint* is provided the returned title is verified against it;
    otherwise a word-overlap heuristic is used to reject wild mismatches.
    """
    if not raw or not raw.strip():
        return None

    logger.debug("Crossref bibliographic: %r", raw[:80])
    data = _get_json(
        "https://api.crossref.org/works",
        {
            "query.bibliographic": raw[:500],
            "rows":                "1",
            "select":              "DOI,title",
        },
    )
    if data is None:
        return None

    items = (data.get("message") or {}).get("items") or []
    for item in items:
        doi = item.get("DOI") or ""
        if not doi:
            continue
        titles    = item.get("title") or []
        candidate = titles[0] if titles else ""
        if not candidate:
            continue

        if title_hint:
            if title_score(title_hint, candidate) < MATCH_THRESHOLD:
                logger.debug("  Crossref biblio no match: %r", candidate[:60])
                continue
            logger.debug("  Crossref biblio matched: %r", candidate[:60])
        else:
            # No title extracted: verify ≥3 substantive words from raw appear in candidate
            raw_words  = {t for t in re.findall(r'[a-zA-Z]{4,}', raw.lower())}
            cand_words = {t for t in re.findall(r'[a-zA-Z]{4,}', candidate.lower())}
            overlap = len(raw_words & cand_words)
            if overlap < 3:
                logger.debug("  Crossref biblio weak overlap (%d words): %r", overlap, candidate[:60])
                continue
            logger.debug("  Crossref biblio heuristic match (overlap=%d): %r", overlap, candidate[:60])

        return {
            "work_id":     f"doi:{doi}",
            "id_source":   "doi",
            "doi":         doi,
            "openalex_id": "",
            "arxiv_id":    "",
        }

    return None


def _try_arxiv(title: str) -> Optional[dict]:

    logger.debug("arXiv search: %r", title[:80])
    words   = _PUNCT_RE.sub(" ", title).strip()
    query   = f'ti:"{words}"'

    root = _get_xml(
        "https://export.arxiv.org/api/query",
        {
            "search_query": query,
            "max_results":  str(_ARXIV_ROWS),
            "sortBy":       "relevance",
        },
    )
    if root is None:
        return None

    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        candidate_el = entry.find(f"{{{_ATOM_NS}}}title")
        candidate    = (candidate_el.text or "") if candidate_el is not None else ""
        score = title_score(title, candidate)
        if score < MATCH_THRESHOLD:
            logger.debug("  arXiv no match (score=%.2f): %r", score, candidate[:60])
            continue
        logger.debug("  arXiv matched (score=%.2f): %r", score, candidate[:60])

        # arxiv id: strip version suffix from the abs URL
        id_el    = entry.find(f"{{{_ATOM_NS}}}id")
        abs_url  = (id_el.text or "").strip() if id_el is not None else ""
        arxiv_id = abs_url.rstrip("/").split("/")[-1]   # 1810.04805v2
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)      # strip version → 1810.04805

        if arxiv_id:
            return {
                "work_id":     f"arxiv:{arxiv_id}",
                "id_source":   "arxiv",
                "doi":         "",
                "openalex_id": "",
                "arxiv_id":    arxiv_id,
            }

    return None


# Public API

def resolve_bib_entry(
    title: str,
    authors: Optional[list] = None,
    year: Optional[str] = None,
    raw: Optional[str] = None,
    ref_id: str = "",
) -> dict:
    """
    Resolve a bibliography entry to a canonical work identifier.

    Resolution chain (stops at first success):

    1. ``_try_crossref_bibliographic(raw)``   — full raw string, best for journals
    2. ``_try_openalex(title)``               — semantic title search
    3. ``_try_crossref(title, year=year)``    — title + optional year filter
    4. ``_try_arxiv(title)``                  — arXiv title search

    Parameters
    ----------
    title:   Extracted article title (may be empty string if extraction failed).
    authors: List of author last-names for future filtering (currently unused).
    year:    4-digit publication year string for Crossref date filter.
    raw:     Full raw bibliography string for the bibliographic endpoint.
    ref_id:  Original ref_id used in the fallback work_id.
    """
    # 1. Crossref bibliographic — most powerful for journal-style references
    if raw and raw.strip():
        result = _try_crossref_bibliographic(raw, title_hint=title)
        if result is not None:
            logger.info("  [biblio]    %s  ← %r", result['work_id'], (title or raw)[:70])
            return result

    if title.strip():
        # 2. OpenAlex title search
        result = _try_openalex(title)
        if result is not None:
            logger.info("  [openalex]  %s  ← %r", result['work_id'], title[:70])
            return result

        # 3. Crossref title search (+ year filter when available)
        result = _try_crossref(title, year=year)
        if result is not None:
            logger.info("  [crossref]  %s  ← %r", result['work_id'], title[:70])
            return result

        # 4. arXiv title search
        result = _try_arxiv(title)
        if result is not None:
            logger.info("  [arxiv]     %s  ← %r", result['work_id'], title[:70])
            return result

    logger.debug("  [unresolved] ref_id=%s  title=%r", ref_id, (title or raw or "")[:60])

    fallback = f"unresolved:{ref_id}" if ref_id else "unresolved"
    return {
        "work_id":     fallback,
        "id_source":   "unresolved",
        "doi":         "",
        "openalex_id": "",
        "arxiv_id":    "",
    }


def resolve_title(title: str, ref_id: str = "") -> dict:
    """Backward-compatible thin wrapper around ``resolve_bib_entry``."""
    return resolve_bib_entry(title=title, ref_id=ref_id)


# CLI smoke test

if __name__ == "__main__":

    test_cases = [
        {
            "title": "Attention Is All You Need",
            "ref_id": "aabbccdd",
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "ref_id": "deadbeef",
        },
        {
            # Journal-style: title only extracted, no raw
            "title": "Smooth bump functions and the geometry of Banach spaces: a brief survey",
            "year":  "2002",
            "ref_id": "aaa111",
        },
        {
            # Full raw string — bibliographic endpoint
            "raw":   "R. Fry and S. McManus. Smooth bump functions and the geometry of Banach spaces: a brief survey. Expo. Math., 20(2):143\u2013183, 2002.",
            "title": "Smooth bump functions and the geometry of Banach spaces: a brief survey",
            "year":  "2002",
            "ref_id": "aaa222",
        },
        {
            "title": "this title almost certainly does not exist anywhere 99999",
            "ref_id": "cafebabe",
        },
    ]

    for tc in test_cases:
        res = resolve_bib_entry(
            title   = tc.get("title", ""),
            year    = tc.get("year"),
            raw     = tc.get("raw"),
            ref_id  = tc.get("ref_id", ""),
        )
        print(f"\nTitle   : {tc.get('title', tc.get('raw',''))!r:.80}")
        print(f"work_id : {res['work_id']}")
        print(f"doi     : {res['doi']}")
        print(f"oa_id   : {res['openalex_id']}")
        print(f"arxiv   : {res['arxiv_id']}")
