from __future__ import annotations


DOMAIN_GROUPS: dict[str, list[str]] = {
    "cs":      ["cs."],
    "physics": ["hep-", "astro-", "quant-ph", "physics."],
    "math":    ["math.", "cond-mat."],
    "stats":   ["stat.", "econ.", "q-bio.", "q-fin.", "nlin."],
}

DOMAINS: list[str] = ["cs", "physics", "math", "stats"]

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


SCROLL_LIMIT: int = 200
VECTOR_SEARCH_LIMIT: int = 50
MIN_EMBED_TEXT_LEN: int = 150


EVALUATION_TABLE_ORDER: list[str] = [
    "full",
    "A1_1",
    "A1_2",
    "R1",
    "R2",
    "R3",
    "G1",
    "G2",
    "J1",
    "J2",
    "J3",
    "standard_rag",
]

EVALUATION_TABLE_LABELS: dict[str, str] = {
    "full":         "EviGraph-R (full)",
    "A1_1":         "A1.1 — No decomposition",
    "A1_2":         "A1.2 — Equal budget weights",
    "R1":           "R1 — Dense only",
    "R2":           "R2 — Sparse/BM25 only",
    "R3":           "R3 — No section boost",
    "G1":           "G1 — Flat chunks",
    "G2":           "G2 — No citation expansion",
    "J1":           "J1 — No judge",
    "J2":           "J2 — NLI only",
    "J3":           "J3 — LLM only",
    "standard_rag": "Standard RAG",
}
