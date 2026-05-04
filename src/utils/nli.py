from __future__ import annotations
import logging
import threading
from typing import Any
from config.settings import GRAPH_CONFIG

logger = logging.getLogger(__name__)


class NLIModel:

    _instance: "NLIModel | None" = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        import torch
        from transformers import pipeline

        model_id = GRAPH_CONFIG.nli_model_id
        device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline(
            "text-classification",
            model=model_id,
            device=device,
            top_k=None,
        )
        self._infer_lock = threading.Lock()
        self._id2label = {
            int(idx): str(label)
            for idx, label in getattr(self._pipe.model.config, "id2label", {}).items()
        }

    @classmethod
    def get(cls) -> "NLIModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def prewarm(cls) -> None:
        cls.get()

    @staticmethod
    def _canonical_label(label: str) -> str | None:
        normalized = label.strip().lower().replace("_", " ").replace("-", " ")
        aliases = {
            "entailment": "entails",
            "entails": "entails",
            "support": "entails",
            "supported": "entails",
            "neutral": "neutral",
            "contradiction": "contradicts",
            "contradicts": "contradicts",
            "contradictory": "contradicts",
        }
        return aliases.get(normalized)

    def _normalize_raw_scores(self, raw: list[dict[str, Any]]) -> dict[str, float]:
        scores = {"entails": 0.0, "contradicts": 0.0, "neutral": 0.0}
        unresolved: list[str] = []

        for row in raw:
            raw_label = str(row.get("label", ""))
            canonical = self._canonical_label(raw_label)
            if canonical is None and raw_label.lower().startswith("label_"):
                try:
                    idx = int(raw_label.split("_", 1)[1])
                except ValueError:
                    idx = -1
                mapped = self._id2label.get(idx, "")
                canonical = self._canonical_label(mapped)
            if canonical is None:
                unresolved.append(raw_label)
                continue
            scores[canonical] = float(row.get("score", 0.0))

        if unresolved:
            logger.warning(
                "[NLI] Unrecognized labels from model %s: %s",
                GRAPH_CONFIG.nli_model_id,
                ", ".join(unresolved),
            )
        return scores

    def classify(self, claim: str, evidence: str) -> dict[str, float]:
        # Serialize inference to avoid CPU contention when called from multiple threads
        with self._infer_lock:
            raw = self._pipe(
                {"text": evidence, "text_pair": claim},
                truncation=True,
                max_length=512,
            )

        # HF pipelines can return either a flat list of labels or a nested list.
        if raw and isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]

        return self._normalize_raw_scores(raw)


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
    elif agg["contradicts"] >= GRAPH_CONFIG.nli_contradiction_threshold:
        verdict = "Contradicted"
        reason = (
            f"NLI contradiction score {agg['contradicts']:.2f} ≥ threshold {GRAPH_CONFIG.nli_contradiction_threshold} "
            f"across {len(evidence_chunks)} chunk(s)."
        )
    else:
        # Neutral is not a final public verdict here; it signals escalation.
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
