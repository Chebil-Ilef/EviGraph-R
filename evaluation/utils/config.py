from __future__ import annotations


DOMAIN_GROUPS: dict[str, list[str]] = {
    "cs":      ["cs."],
    "physics": ["hep-", "astro-", "quant-ph", "physics."],
    "math":    ["math.", "cond-mat."],
    "stats":   ["stat.", "econ.", "q-bio.", "q-fin.", "nlin."],
}

DOMAINS: list[str] = ["cs", "physics", "math", "stats"]

# Target share per domain per category (25% each)
DOMAIN_TARGET_SHARE: float = 0.25


def arxiv_category_to_domain(categories: list[str]) -> str:
    for cat in categories:
        cat_lower = cat.lower()
        if any(cat_lower.startswith(pfx) for pfx in DOMAIN_GROUPS["cs"]):
            return "cs"
        if any(cat_lower.startswith(pfx) for pfx in DOMAIN_GROUPS["physics"]):
            return "physics"
        if any(cat_lower.startswith(pfx) for pfx in DOMAIN_GROUPS["math"]):
            return "math"
    return "stats"



CATEGORY_TARGETS: dict[int, int] = {
    1: 200,  # single-paper synthesis
    2: 150,  # cross-section synthesis
    3: 200,  # citation expansion
    4: 200,  # thematic multi-paper
}

def domain_quota(category: int) -> int:
    return CATEGORY_TARGETS[category] // len(DOMAINS)

# Similarity windows (cosine)
# Category 3: citing chunk A ↔ cited chunk B
CAT3_SIM_MIN: float = 0.40
CAT3_SIM_MAX: float = 0.80

# Category 4: seed chunk ↔ thematic neighbours
CAT4_SIM_MIN: float = 0.50
CAT4_SIM_MAX: float = 0.80


VALID_IMRAD_PAIRS: list[tuple[str, str]] = [
    ("introduction", "results"),
    ("methods",      "results"),
    ("methods",      "discussion"),
    ("introduction", "conclusion"),
    ("methodology",  "results"),
    ("methodology",  "discussion"),
    ("approach",     "results"),
    ("approach",     "evaluation"),
    ("experiments",  "discussion"),
    ("experiments",  "conclusion"),
]

# Normalised set for O(1) membership checks
_PAIR_SET: set[frozenset[str]] = {
    frozenset(p) for p in VALID_IMRAD_PAIRS
}


def is_valid_imrad_pair(a: str, b: str) -> bool:
    return frozenset({a.lower(), b.lower()}) in _PAIR_SET


EXCLUDED_SECTION_LABELS: set[str] = {
    "", "abstract", "references", "bibliography", "acknowledgements",
    "acknowledgments", "appendix", "supplementary", "related work",
    "table of contents",
}


def section_label(payload: dict) -> str:
    raw = (
        payload.get("imrad_section_title")
        or payload.get("section_title")
        or ""
    )
    return raw.strip().lower()


SCROLL_LIMIT: int = 200          # points per scroll page
VECTOR_SEARCH_LIMIT: int = 50    # neighbours returned per vector search
MIN_EMBED_TEXT_LEN: int = 150    # chunks shorter than this are skipped
