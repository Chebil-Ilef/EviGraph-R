from __future__ import annotations
import re
from typing import Any

from config.settings import GRAPH_CONFIG


# NPM Non-Parametric Model
#
# Role in the pipeline: LEXICAL PRE-FILTER only.
#   - "Not-Supported" is reliable: key tokens are absent → claim cannot be supported.
#   - "Supported" is NOT a final verdict: tokens present does not imply semantics are correct.
#     The caller (judge._route_and_verify) always passes "Supported" on to NLI for confirmation.
#   - "no_key_tokens" signals the claim has no extractable tokens; skip NPM entirely.
def npm_verify(claim_text: str, evidence_chunks: list[str]) -> dict[str, Any]:

    if not evidence_chunks:
        return {
            "verdict": "Not-Supported",
            "verifier_used": "npm",
            "evidence_trail": [],
            "error_stage": "no_evidence",
            "key_tokens_found": None,
        }

    key_tokens = extract_key_tokens(claim_text)
    if not key_tokens:
        # No extractable tokens → signal caller to skip NPM and go straight to NLI
        return {
            "verdict": "Inconclusive",
            "verifier_used": "npm",
            "evidence_trail": [],
            "error_stage": "no_key_tokens",
            "key_tokens_found": None,
        }

    evidence_combined = " ".join(evidence_chunks).lower()
    matched_tokens = [t for t in key_tokens if t.lower() in evidence_combined]
    missed_tokens  = [t for t in key_tokens if t.lower() not in evidence_combined]
    coverage = len(matched_tokens) / len(key_tokens)
    verdict  = "Supported" if coverage >= GRAPH_CONFIG.npm_threshold else "Not-Supported"

    if verdict == "Not-Supported":
        reason = (
            f"Key tokens missing from evidence ({len(matched_tokens)}/{len(key_tokens)} found). "
            + (f"Missing: {', '.join(missed_tokens)}." if missed_tokens else "")
        )
    else:
        # "Supported" here means tokens are present — caller will confirm via NLI
        reason = (
            f"Key tokens present in evidence ({len(matched_tokens)}/{len(key_tokens)}): "
            + ", ".join(matched_tokens) + ". Passing to NLI for semantic confirmation."
        )

    trail = [{"text": chunk[:200]} for chunk in evidence_chunks]

    return {
        "verdict": verdict,
        "verifier_used": "npm",
        "evidence_trail": trail,
        "error_stage": None,
        "key_tokens_found": matched_tokens,
        "reason": reason,
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
