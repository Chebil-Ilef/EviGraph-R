
from __future__ import annotations
import json
import logging
import re
import time
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
_RATE_LIMIT_COOLDOWN: float = 60.0  # seconds to skip a host after receiving HTTP 429
USER_AGENT = "scholarly-indexer/1.0 (mailto:ilef-chebil@example.com)"
_ATOM_NS   = "http://www.w3.org/2005/Atom"

# Per-host rate-limit state  {hostname → cooldown_expires_at}
_rate_limited_until: dict[str, float] = {}


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
    containment = 0.0
    if len(b) <= len(a) * LENGTH_RATIO_CAP and len(a) <= len(b) * LENGTH_RATIO_CAP:
        containment = max(inter / len(a), inter / len(b))
    return max(jaccard, containment)


# HTTP helper

def _get_json(url: str, params: dict) -> Optional[dict]:

    host = urllib.parse.urlparse(url).netloc
    if time.time() < _rate_limited_until.get(host, 0):
        logger.debug("[rate-limited] skipping %s (cooldown active)", host)
        return None

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
        if exc.code == 429:
            _rate_limited_until[host] = time.time() + 10
            logger.warning(
                "HTTP 429 from %s — pausing that API for %.0fs",
                host, _RATE_LIMIT_COOLDOWN,
            )
        else:
            logger.warning("HTTP %s for %s", exc.code, full_url[:100])
        return None
    except Exception as exc:
        logger.warning("Request failed (%s): %s", type(exc).__name__, url)
        return None


def _get_xml(url: str, params: dict) -> Optional[ET.Element]:

    host = urllib.parse.urlparse(url).netloc
    if time.time() < _rate_limited_until.get(host, 0):
        logger.debug("[rate-limited] skipping %s (cooldown active)", host)
        return None

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
        if exc.code == 429:
            _rate_limited_until[host] = time.time() + 10
            logger.warning(
                "HTTP 429 from %s — pausing that API for %.0fs",
                host, _RATE_LIMIT_COOLDOWN,
            )
        else:
            logger.warning("HTTP %s for %s", exc.code, full_url[:100])
        return None
    except Exception as exc:
        logger.warning("Request failed (%s): %s", type(exc).__name__, url)
        return None


# Per-API resolvers

def _try_openalex(query: str, title_hint: str = "") -> Optional[dict]:

    logger.debug("OpenAlex search: %r", query[:80])
    data = _get_json(
        "https://api.openalex.org/works",
        {
            "search":   query,
            "select":   "id,title,doi,ids,authorships",
            "per-page": str(_OPENALEX_ROWS),
        },
    )
    if data is None:
        return None

    for item in (data.get("results") or []):
        candidate = item.get("title") or ""
        matched = False
        if title_hint and candidate and title_score(title_hint, candidate) >= MATCH_THRESHOLD:
            matched = True
            
        if not matched:
            authorships = item.get("authorships") or []
            authors = [a.get("author", {}).get("display_name", "") for a in authorships]
            family_names = [name.split()[-1].lower() for name in authors if name]
            query_lower = query.lower()
            if family_names and any(f in query_lower for f in family_names if len(f) > 2):
                matched = True
        
        if not matched:
            continue

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



def _try_crossref_bibliographic(raw: str, title_hint: str = "") -> Optional[dict]:

    if not raw or not raw.strip():
        return None

    logger.debug("Crossref bibliographic: %r", raw[:80])
    data = _get_json(
        "https://api.crossref.org/works",
        {
            "query.bibliographic": raw[:500],
            "rows":                "1",
            "select":              "DOI,title,author",
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

        matched = False
        if title_hint and candidate and title_score(title_hint, candidate) >= MATCH_THRESHOLD:
            matched = True
        
        if not matched:
            author_names = [a.get("family", "").lower() for a in item.get("author", []) if a.get("family")]
            raw_lower = raw.lower()
            if author_names and any(auth in raw_lower for auth in author_names if len(auth) > 2):
                matched = True
                
        if not matched:
            logger.debug("  Crossref biblio rejected (no title nor author match): %r", candidate[:60])
            continue

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


# DOI / OpenAlex ID verification

def verify_doi(doi: str, title_hint: str) -> bool:

    if not doi or not title_hint.strip():
        return True

    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/:')}"
    host = urllib.parse.urlparse(url).netloc
    if time.time() < _rate_limited_until.get(host, 0):
        logger.debug("[rate-limited] skipping DOI verify for %s", doi)
        return True  # can't verify right now

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    logger.debug("Verify DOI %s", doi)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            _rate_limited_until[host] = time.time() + _RATE_LIMIT_COOLDOWN
        logger.debug("DOI verify HTTP %s for doi=%s", exc.code, doi)
        return True  # can't verify
    except Exception:
        return True  # can't verify

    item = (data.get("message") or {})
    titles = item.get("title") or []
    candidate = titles[0] if titles else ""
    if not candidate:
        return True  # no title to compare

    score = title_score(title_hint, candidate)
    if score < MATCH_THRESHOLD:
        logger.debug(
            "DOI mismatch (score=%.2f): doi=%s  expected=%r  got=%r",
            score, doi, title_hint[:70], candidate[:70],
        )
        return False
    return True


def verify_openalex_id(openalex_id: str, title_hint: str) -> bool:

    if not openalex_id or not title_hint.strip():
        return True

    # build canonical request URL: https://api.openalex.org/works/W...
    w_id = openalex_id.rstrip("/").split("/")[-1]
    url  = f"https://api.openalex.org/works/{w_id}"
    host = urllib.parse.urlparse(url).netloc
    if time.time() < _rate_limited_until.get(host, 0):
        logger.debug("[rate-limited] skipping OA verify for %s", w_id)
        return True

    req = urllib.request.Request(
        url + "?select=title",
        headers={"User-Agent": USER_AGENT},
    )
    logger.debug("Verify OpenAlex %s", w_id)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            _rate_limited_until[host] = time.time() + _RATE_LIMIT_COOLDOWN
        logger.debug("OA verify HTTP %s for %s", exc.code, w_id)
        return True
    except Exception:
        return True

    candidate = data.get("title") or ""
    if not candidate:
        return True

    score = title_score(title_hint, candidate)
    if score < MATCH_THRESHOLD:
        logger.debug(
            "OpenAlex mismatch (score=%.2f): id=%s  expected=%r  got=%r",
            score, w_id, title_hint[:70], candidate[:70],
        )
        return False
    return True


# Public API

_resolve_cache = {}

def resolve_bib_entry(
    title: str,
    raw: Optional[str] = None,
    ref_id: str = "",
) -> dict:
    """
    Resolve a bibliography entry to a canonical work identifier.

    Resolution chain (stops at first success):

    1. ``_try_crossref_bibliographic(raw)``   — full raw string, best for journals
    2. ``_try_arxiv(raw)``                    — arXiv title search
    3. ``_try_openalex(title)``               — semantic title search


    """
    # 1. Crossref bibliographic — most powerful for journal-style references
    cache_key = f"{title}:::{(raw or '').strip()}"
    if cache_key in _resolve_cache:
        return _resolve_cache[cache_key]

    if raw and raw.strip():
        result = _try_crossref_bibliographic(raw, title_hint=title)
        if result is not None:
            logger.info("  [biblio]    %s  ← %r", result['work_id'], (title or raw)[:70])
            _resolve_cache[cache_key] = result
            return result
        else:
            logger.debug("  [biblio]    no match for ref_id=%s  title=%r", ref_id, (title or raw)[:60])
            if title.strip():
                # 2. arXiv title search
                result = _try_arxiv(title)
                if result is not None:
                    logger.info("  [arxiv]     %s  ← %r", result['work_id'], title[:70])
                    _resolve_cache[cache_key] = result
            return result


    if title.strip():
        # 3. OpenAlex title search
        result = _try_openalex(title, title_hint=title)
        if result is not None:
            logger.info("  [openalex]  %s  ← %r", result['work_id'], title[:70])
            _resolve_cache[cache_key] = result
            return result
    elif raw and raw.strip():
        # OpenAlex raw search
        result = _try_openalex(raw[:500], title_hint="")
        if result is not None:
            logger.info("  [openalex raw] %s  ← %r", result['work_id'], raw[:70])
            _resolve_cache[cache_key] = result
            return result
                
        

    logger.debug("  [unresolved] ref_id=%s  title=%r", ref_id, (title or raw or "")[:60])

    fallback = f"unresolved:{ref_id}" if ref_id else "unresolved"
    res = {
        "work_id":     fallback,
        "id_source":   "unresolved",
        "doi":         "",
        "openalex_id": "",
        "arxiv_id":    "",
    }
    _resolve_cache[cache_key] = res
    return res


def resolve_title(title: str, raw: str , ref_id: str = "") -> dict:
    return resolve_bib_entry(title=title, raw=raw,  ref_id=ref_id)


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
