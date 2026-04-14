from __future__ import annotations
import logging
from typing import Any
from config.settings import GRAPH_CONFIG

logger = logging.getLogger(__name__)


class NLIModel:
    
    _instance: "NLIModel | None" = None

    def __init__(self) -> None:
        from transformers import pipeline
        from config.settings import GRAPH_CONFIG
        
        self._pipe = pipeline(
            "zero-shot-classification",
            model=GRAPH_CONFIG.nli_model_id,
        )

    @classmethod
    def get(cls) -> "NLIModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def classify(self, claim: str, evidence: str) -> dict[str, float]:
 
        result = self._pipe(
            evidence,
            candidate_labels=["entails", "contradicts", "neutral"],
            hypothesis_template="{}",
        )
        scores: dict[str, float] = {}
        for label, score in zip(result["labels"], result["scores"]):
            scores[label] = score
        return scores


def nli_verify(claim_text: str, evidence_chunks: list[str]) -> dict[str, Any]:

    if not evidence_chunks:
        return {
            "verdict": "Not-Supported",
            "verifier_used": "nli",
            "evidence_trail": [],
            "error_stage": "no_evidence",
        }

    try:
        nli = NLIModel.get()
    except Exception as exc:
        logger.warning("[NLI] Model load failed: %s", exc)
        return {
            "verdict": "Neutral",  # Signal to escalate
            "verifier_used": "nli",
            "evidence_trail": [],
            "error_stage": "model_load_failed",
        }

    # Aggregate NLI scores across all evidence chunks
    agg: dict[str, float] = {"entails": 0.0, "contradicts": 0.0, "neutral": 0.0}
    trail: list[dict] = []

    for chunk in evidence_chunks:
        scores = nli.classify(claim_text, chunk)
        for k in agg:
            agg[k] = max(agg[k], scores.get(k, 0.0))
        trail.append({"text": chunk[:200], "scores": scores})

    if agg["entails"] >= GRAPH_CONFIG.nli_threshold:
        verdict = "Supported"
        reason = (
            f"NLI entailment score {agg['entails']:.2f} ≥ threshold {GRAPH_CONFIG.nli_threshold} "
            f"across {len(evidence_chunks)} chunk(s)."
        )
    elif agg["contradicts"] >= GRAPH_CONFIG.nli_threshold:
        verdict = "Contradicted"
        reason = (
            f"NLI contradiction score {agg['contradicts']:.2f} ≥ threshold {GRAPH_CONFIG.nli_threshold} "
            f"across {len(evidence_chunks)} chunk(s)."
        )
    else:
        # Neutral → signal escalation to LLM judge
        verdict = "Neutral"
        reason = (
            f"NLI scores ambiguous (entail={agg['entails']:.2f}, contradict={agg['contradicts']:.2f}) "
            f"— escalating to LLM."
        )

    return {
        "verdict": verdict,
        "verifier_used": "nli",
        "evidence_trail": trail,
        "error_stage": None,
        "reason": reason,
    }

