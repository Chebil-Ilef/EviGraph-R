from __future__ import annotations
import re
from typing import Any

from config.settings import GRAPH_CONFIG


# NPM Non-Parametric Model
def npm_verify(claim_text: str, evidence_chunks: list[str]) -> dict[str, Any]:

    if not evidence_chunks:
        return {
            "verdict": "Not-Supported",
            "verifier_used": "npm",
            "evidence_trail": [],
            "error_stage": "no_evidence",
        }

    key_tokens = extract_key_tokens(claim_text)
    if not key_tokens:
        return {
            "verdict": "Inconclusive",
            "verifier_used": "npm",
            "evidence_trail": [],
            "error_stage": None,
        }

    evidence_combined = " ".join(evidence_chunks).lower()
    matched = sum(1 for t in key_tokens if t.lower() in evidence_combined)
    coverage = matched / len(key_tokens)

    trail = [{"text": chunk[:200]} for chunk in evidence_chunks]

    return {
        "verdict": "Supported" if coverage >= GRAPH_CONFIG.npm_threshold else "Not-Supported",
        "verifier_used": "npm",
        "evidence_trail": trail,
        "error_stage": None,
    }


def extract_key_tokens(text: str) -> list[str]:

    tokens: list[str] = []
    # Numeric values
    tokens += re.findall(r"\b\d+(?:\.\d+)?(?:%|x)?\b", text)
    # Acronyms / named entities (all-caps sequences ≥2 chars)
    tokens += re.findall(r"\b[A-Z][A-Z0-9\-]{1,}\b", text)
    # Quoted terms
    tokens += re.findall(r'"([^"]+)"', text)
    return list(dict.fromkeys(tokens))  # deduplicate, preserve order
