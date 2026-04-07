
from __future__ import annotations
import asyncio
import json
import logging
import random
import re
import time
import urllib.parse
from dataclasses import dataclass
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Per-API call/error/429 counters — read by citation_ids.build_stats()
api_stats: dict[str, dict[str, int]] = {
    "crossref":  {"calls": 0, "errors": 0, "rate_limited": 0, "matched": 0},
    "openalex":  {"calls": 0, "errors": 0, "rate_limited": 0, "matched": 0},
}

# Configuration
MATCH_THRESHOLD: float = 0.85   # Jaccard score required to accept a candidate
LENGTH_RATIO_CAP = 1.5          # candidate title must be ≤ query length × this ratio to count for containment score
_OPENALEX_ROWS:  int   = 3      # candidates to fetch per API call
_CROSSREF_ROWS:  int   = 3
_TIMEOUT:        int   = 10     # seconds per HTTP call
_RATE_LIMIT_COOLDOWN: float = 60.0  # seconds to back off after HTTP 429
_RATE_LIMIT_MAX_WAIT: float = 120.0  # max seconds to wait for a rate-limited host to recover
_JITTER_MAX: float = 5.0  # max random jitter (seconds) on retry wakeups to desynchronize
_MAX_ATTEMPTS_PER_API: int = 2  # max retry attempts per (ref_id, host) pair before giving up
_RETRY_QUEUE_DELAY: float = 0.1  # delay between queued retries when API is rate-limited
USER_AGENT = "scholarly-indexer/1.0 (mailto:ilef-chebil@example.com)"

# Per-host concurrency limits — stay polite to public APIs
_HOST_CONCURRENCY: dict[str, int] = {
    "api.crossref.org":    8,
    "api.openalex.org":    1,
}
_DEFAULT_CONCURRENCY = 4

# Per-host request rate limits (requests per second).
# These are the true throughput caps — the token bucket enforces them
# proactively so we never need to wait for a 429 cooldown.
# Crossref polite pool: ~50 req/s documented, but HPC NAT may share quota —
# stay at 5 to be safe. OpenAlex free tier: ~10 req/s.
_HOST_RPS: dict[str, float] = {
    "api.crossref.org":    5.0,
    "api.openalex.org":    1.0,
}

# Per-host semaphores (created lazily inside the event loop)
_semaphores: dict[str, asyncio.Semaphore] = {}

# Per-host token bucket rate limiters (created lazily inside the event loop)
_rate_limiters: dict[str, "_TokenBucket"] = {}


class _TokenBucket:
    """
    Slot-reservation rate limiter.

    Each caller atomically reserves the next available slot (advancing
    _next_allowed by 1/rate seconds), releases the lock, then sleeps until
    its slot arrives.  This means waiters never refill the bucket by accident
    and the lock is held for only a few microseconds — no sleep inside the
    critical section.
    """

    def __init__(self, rate: float) -> None:
        self._interval = 1.0 / rate          # seconds between slots
        self._next_allowed = time.monotonic() # next slot available at this time
        self._lock: Optional[asyncio.Lock] = None  # created lazily in event loop

    async def acquire(self) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            now = time.monotonic()
            # If we're ahead of the queue, start from now (don't accumulate credit)
            if self._next_allowed < now:
                self._next_allowed = now
            slot = self._next_allowed
            self._next_allowed += self._interval
        # Sleep outside the lock so other callers can reserve their slots
        wait = slot - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)


def _rate_limiter(host: str) -> "_TokenBucket":
    if host not in _rate_limiters:
        rps = _HOST_RPS.get(host, float(_DEFAULT_CONCURRENCY))
        _rate_limiters[host] = _TokenBucket(rps)
    return _rate_limiters[host]

# Per-host rate-limit state  {hostname → cooldown_expires_at}
_rate_limited_until: dict[str, float] = {}

# Track attempt counts per (ref_id, host) to enforce max attempts and avoid infinite loops
# Key: (ref_id, hostname) → attempt count
_attempt_counts: dict[tuple[str, str], int] = {}


@dataclass(frozen=True)
class _ResolverSpec:
    name: str
    host: str


def _semaphore(host: str) -> asyncio.Semaphore:
    if host not in _semaphores:
        limit = _HOST_CONCURRENCY.get(host, _DEFAULT_CONCURRENCY)
        _semaphores[host] = asyncio.Semaphore(limit)
    return _semaphores[host]


def _cooldown_remaining(host: str) -> float:
    return max(0.0, _rate_limited_until.get(host, 0.0) - time.time())


def _is_rate_limited(host: str) -> bool:
    return _cooldown_remaining(host) > 0.0


def _can_attempt(ref_id: str, host: str) -> bool:

    key = (ref_id, host)
    attempts = _attempt_counts.get(key, 0)
    return attempts < _MAX_ATTEMPTS_PER_API


def _increment_attempt(ref_id: str, host: str) -> None:

    key = (ref_id, host)
    _attempt_counts[key] = _attempt_counts.get(key, 0) + 1


# Title normalisation & scoring
_PUNCT_RE   = re.compile(r"[^\w\s]")
_SPACE_RE   = re.compile(r"\s+")
_LATEX_CMD  = re.compile(r"\\[a-zA-Z]+\b")   # \displaystyle, \mathbb, etc.
_LATEX_MATH = re.compile(r"\$[^$]*\$")        # inline math $...$
_HTML_TAG   = re.compile(r"<[^>]+>")          # HTML tags from Crossref titles
# Common stop-words that don't help matching and skew Jaccard
_STOP_WORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "by", "with", "from", "is", "it", "as",
})


def _token_set(title: str) -> set[str]:
    t = _LATEX_MATH.sub(" ", title)     # strip $...$ math blocks first
    t = _LATEX_CMD.sub(" ", t)          # strip \cmd tokens
    t = _HTML_TAG.sub(" ", t)           # strip HTML tags (<sub>2</sub> etc.)
    t = t.replace("_", " ")            # gl_2 → gl 2
    t = _PUNCT_RE.sub(" ", t.lower())
    tokens = set(_SPACE_RE.sub(" ", t).strip().split())
    return tokens - _STOP_WORDS


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
    # Short candidate fully contained in query (e.g. book title inside full bib entry)
    # Only trust this when the candidate has enough tokens to be meaningful (≥ 3)
    candidate_coverage = inter / len(b) if len(b) >= 3 else 0.0
    return max(jaccard, containment, candidate_coverage)


# Async HTTP helpers

def _api_key(host: str) -> str:
    if "crossref" in host:
        return "crossref"
    if "openalex" in host:
        return "openalex"
    return host


async def _get_json(session: aiohttp.ClientSession, url: str, params: dict) -> Optional[dict]:
    host = urllib.parse.urlparse(url).netloc
    key  = _api_key(host)

    if _is_rate_limited(host):
        api_stats[key]["rate_limited"] += 1
        logger.debug("[rate-limited] %s still cooling down; skipping immediate request", host)
        return None
    full_url = url + "?" + urllib.parse.urlencode(params)
    logger.debug("GET %s", full_url[:140])
    api_stats[key]["calls"] += 1
    await _rate_limiter(host).acquire()
    async with _semaphore(host):
        try:
            async with session.get(full_url, timeout=aiohttp.ClientTimeout(total=_TIMEOUT)) as resp:
                if resp.status == 429:
                    _rate_limited_until[host] = time.time() + _RATE_LIMIT_COOLDOWN
                    api_stats[key]["rate_limited"] += 1
                    logger.warning("HTTP 429 from %s — pausing that API for %.0fs", host, _RATE_LIMIT_COOLDOWN)
                    return None
                if resp.status != 200:
                    api_stats[key]["errors"] += 1
                    logger.warning("HTTP %s for %s", resp.status, full_url[:100])
                    return None
                return await resp.json(content_type=None)
        except asyncio.TimeoutError:
            api_stats[key]["errors"] += 1
            logger.warning("Timeout for %s", url)
            return None
        except Exception as exc:
            api_stats[key]["errors"] += 1
            logger.warning("Request failed (%s): %s", type(exc).__name__, url)
            return None


# Per-API resolvers

async def _try_openalex(session: aiohttp.ClientSession, query: str, title_hint: str = "") -> Optional[dict]:
    logger.debug("OpenAlex search: %r", query[:80])
    data = await _get_json(
        session,
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
            if family_names and any(f in query_lower for f in family_names if len(f) >= 2):
                matched = True

        if not matched:
            continue

        oa_url   = item.get("id") or ""
        oa_w_id  = oa_url.rstrip("/").split("/")[-1]

        raw_doi  = item.get("doi") or ""
        doi      = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", raw_doi)

        ids      = item.get("ids") or {}
        arxiv_raw = ids.get("arxiv") or ""
        arxiv_id  = arxiv_raw.rstrip("/").split("/")[-1] if arxiv_raw else ""

        if doi:
            api_stats["openalex"]["matched"] += 1
            return {
                "work_id":     f"doi:{doi}",
                "id_source":   "doi",
                "doi":         doi,
                "openalex_id": oa_url,
                "arxiv_id":    arxiv_id,
            }
        if oa_w_id:
            api_stats["openalex"]["matched"] += 1
            return {
                "work_id":     f"openalex:{oa_w_id}",
                "id_source":   "openalex",
                "doi":         "",
                "openalex_id": oa_url,
                "arxiv_id":    arxiv_id,
            }

    return None


async def _try_crossref_bibliographic(session: aiohttp.ClientSession, raw: str, title_hint: str = "") -> Optional[dict]:
    if not raw or not raw.strip():
        return None

    logger.debug("Crossref bibliographic: %r", raw[:80])
    data = await _get_json(
        session,
        "https://api.crossref.org/works",
        {
            "query.bibliographic": raw[:500],
            "rows":                str(_CROSSREF_ROWS),
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
        score = title_score(title_hint, candidate) if (title_hint and candidate) else 0.0
        if score >= MATCH_THRESHOLD:
            matched = True

        if not matched:
            author_names = [
                a.get("family", "").lower() or a.get("name", "").lower().split()[-1]
                for a in item.get("author", [])
                if a.get("family") or a.get("name")
            ]
            raw_lower = raw.lower()
            if author_names and any(auth in raw_lower for auth in author_names if len(auth) >= 2):
                matched = True
            elif not author_names and candidate:
                # No author metadata (common for book-level DOIs) — accept if the
                # candidate title appears as a substring of the raw bib entry
                if candidate.lower() in raw.lower() or score >= 0.5:
                    matched = True

        if not matched:
            logger.debug("  Crossref biblio rejected (no title nor author match): %r", candidate[:60])
            continue

        api_stats["crossref"]["matched"] += 1
        return {
            "work_id":     f"doi:{doi}",
            "id_source":   "doi",
            "doi":         doi,
            "openalex_id": "",
            "arxiv_id":    "",
        }

    return None


# Public async API

_resolve_cache: dict[str, dict] = {}
# Deduplication: if two coroutines race on the same key, only one fires the HTTP calls.
_inflight: dict[str, asyncio.Future] = {}


async def resolve_bib_entry(
    session: aiohttp.ClientSession,
    raw: Optional[str],
    ref_id: str = "",
) -> Optional[dict]:
    """
    Async resolution chain (stops at first success):

    1. Crossref bibliographic  — full raw string, best for journals
    2. OpenAlex raw search     — semantic fallback
    """
    
    cache_key = (raw or "").strip()

    if cache_key in _resolve_cache:
        logger.debug("[cached] ref_id=%s  returning cached result", ref_id)
        return _resolve_cache[cache_key]

    # If another coroutine is already resolving this key, wait for it.
    if cache_key in _inflight:
        logger.debug("[dedup] ref_id=%s  awaiting in-flight resolution", ref_id)
        return await asyncio.shield(_inflight[cache_key])

    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()
    _inflight[cache_key] = future

    try:
        result = await _resolve_bib_entry_impl(session, raw, ref_id, cache_key)
        future.set_result(result)
        return result
    except Exception as exc:
        future.set_exception(exc)
        raise
    finally:
        _inflight.pop(cache_key, None)


async def _resolve_bib_entry_impl(
    session: aiohttp.ClientSession,
    raw: Optional[str],
    ref_id: str,
    cache_key: str,
) -> Optional[dict]:

    result: Optional[dict] = None

    if raw and raw.strip():
        resolvers = [
            _ResolverSpec("crossref", "api.crossref.org"),
            _ResolverSpec("openalex", "api.openalex.org"),
        ]
        pending = list(resolvers)

        while pending:
            available: list[_ResolverSpec] = []
            cooling: list[_ResolverSpec] = []
            exhausted: list[_ResolverSpec] = []  # APIs we've hit max attempts on

            for spec in pending:
                if not _can_attempt(ref_id, spec.host):
                    # Already hit max attempts; give up on this API
                    exhausted.append(spec)
                elif _is_rate_limited(spec.host):
                    cooling.append(spec)
                else:
                    available.append(spec)

            if available:
                retry_after_cooldown: list[_ResolverSpec] = []

                for spec in available:
                    _increment_attempt(ref_id, spec.host)
                    
                    if spec.name == "crossref":
                        logger.debug("[attempt] Trying Crossref with raw=%r", raw[:80])
                        result = await _try_crossref_bibliographic(session, raw, title_hint=raw)
                        if result is not None:
                            logger.info("  [biblio]    %s  ← %r", result['work_id'], raw[:70])
                            _resolve_cache[cache_key] = result
                            return result
                        logger.debug("  [biblio]    no match for ref_id=%s  raw=%r", ref_id, raw[:60])
                    elif spec.name == "openalex":
                        logger.debug("[attempt] Trying OpenAlex for ref_id=%s", ref_id)
                        result = await _try_openalex(session, raw[:500], title_hint=raw)
                        if result is not None:
                            logger.info("  [openalex]  %s  ← %r", result['work_id'], raw[:70])
                            _resolve_cache[cache_key] = result
                            return result
                        logger.debug("  [openalex]  no match for ref_id=%s", ref_id)

                    if _is_rate_limited(spec.host):
                        retry_after_cooldown.append(spec)

                pending = cooling + retry_after_cooldown
                continue

            # If all remaining APIs are exhausted (max attempts reached), give up
            if not cooling:
                logger.warning(
                    "[rate-limited] all remaining resolvers exhausted or max attempts reached for ref_id=%s",
                    ref_id,
                )
                break

            next_spec = min(cooling, key=lambda spec: _cooldown_remaining(spec.host))
            wait = _cooldown_remaining(next_spec.host)
            if wait <= 0:
                pending = cooling
                continue
            if wait > _RATE_LIMIT_MAX_WAIT:
                logger.warning(
                    "[rate-limited] remaining resolvers for ref_id=%s unavailable for too long (%.0fs); skipping wait",
                    ref_id,
                    wait,
                )
                break

            # Add jitter to desynchronize concurrent retries and avoid thundering herd
            jitter = random.uniform(0.0, _JITTER_MAX)
            actual_wait = wait + jitter
            logger.info(
                "[rate-limited] all remaining resolvers cooling down for ref_id=%s; waiting %.1fs (%.1fs base + %.1fs jitter) for %s",
                ref_id,
                actual_wait,
                wait,
                jitter,
                next_spec.host,
            )
            await asyncio.sleep(actual_wait)
            pending = cooling

    logger.warning("  [unresolved] ref_id=%s  raw=%r", ref_id, (raw or "")[:60])

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


# CLI smoke test

if __name__ == "__main__":
    import sys

    test_cases = [
        {"raw": "Vaswani et al. Attention Is All You Need. NeurIPS 2017.", "ref_id": "aabbccdd"},
        {"raw": "Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers. NAACL 2019.", "ref_id": "deadbeef"},
        {
            "raw":   "R. Fry and S. McManus. Smooth bump functions and the geometry of Banach spaces: a brief survey. Expo. Math., 20(2):143\u2013183, 2002.",
            "ref_id": "aaa222",
        },
        {"raw": "this title almost certainly does not exist anywhere 99999", "ref_id": "cafebabe"},
    ]

    async def _smoke():
        async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
            tasks = [
                resolve_bib_entry(session, tc.get("raw"), tc.get("ref_id", ""))
                for tc in test_cases
            ]
            results = await asyncio.gather(*tasks)
        for tc, res in zip(test_cases, results):
            print(f"\nTitle   : {tc.get('title', tc.get('raw',''))!r:.80}")
            if res is None:
                print("  [no result]")
                continue
            print(f"work_id : {res['work_id']}")
            print(f"doi     : {res['doi']}")
            print(f"oa_id   : {res['openalex_id']}")
            print(f"arxiv   : {res['arxiv_id']}")

    asyncio.run(_smoke())
