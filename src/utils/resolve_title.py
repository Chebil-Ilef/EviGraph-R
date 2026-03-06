
from __future__ import annotations
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional

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
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError:
        return None   
    except Exception:
        return None


def _get_xml(url: str, params: dict) -> Optional[ET.Element]:

    full_url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        full_url,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return ET.fromstring(resp.read())
    except urllib.error.HTTPError:
        return None
    except Exception:
        return None


# Per-API resolvers

def _try_openalex(title: str) -> Optional[dict]:

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
        if title_score(title, candidate) < MATCH_THRESHOLD:
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


def _try_crossref(title: str) -> Optional[dict]:

    data = _get_json(
        "https://api.crossref.org/works",
        {
            "query.title": title,
            "rows":        str(_CROSSREF_ROWS),
            "select":      "DOI,title",
        },
    )
    if data is None:
        return None

    items = (data.get("message") or {}).get("items") or []
    for item in items:
        titles    = item.get("title") or []
        candidate = titles[0] if titles else ""
        if title_score(title, candidate) < MATCH_THRESHOLD:
            continue

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


def _try_arxiv(title: str) -> Optional[dict]:

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
        if title_score(title, candidate) < MATCH_THRESHOLD:
            continue

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

def resolve_title(title: str, ref_id: str = "") -> dict:
   
    if title.strip():
        result = _try_openalex(title)
        if result is not None:
            return result

        result = _try_crossref(title)
        if result is not None:
            return result

        result = _try_arxiv(title)
        if result is not None:
            return result

    fallback = f"unresolved:{ref_id}" if ref_id else "unresolved"
    return {
        "work_id":     fallback,
        "id_source":   "unresolved",
        "doi":         "",
        "openalex_id": "",
        "arxiv_id":    "",
    }


# CLI smoke test

if __name__ == "__main__":

    test_titles = [
        ("Attention Is All You Need",                "aabbccdd"),
        ("BERT: Pre-training of Deep Bidirectional Transformers", "deadbeef"),
        ("this title almost certainly does not exist anywhere 99999", "cafebabe"),
    ]

    for title, rid in test_titles:
        res = resolve_title(title, rid)
        print(f"\nTitle   : {title!r}")
        print(f"work_id : {res['work_id']}")
        print(f"doi     : {res['doi']}")
        print(f"oa_id   : {res['openalex_id']}")
        print(f"arxiv   : {res['arxiv_id']}")
