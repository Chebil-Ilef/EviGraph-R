from __future__ import annotations
import logging
import threading
from typing import Any
from evigraph.config.settings import GRAPH_CONFIG

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

    @staticmethod
    def _as_score_rows(raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        if raw and isinstance(raw[0], list):
            nested = raw[0]
            return nested if isinstance(nested, list) else []
        return raw

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

        return self._normalize_raw_scores(self._as_score_rows(raw))

    def classify_batch(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:

        inputs = [{"text": evidence, "text_pair": claim} for claim, evidence in pairs]
        with self._infer_lock:
            raw_batch = self._pipe(inputs, truncation=True, max_length=512, batch_size=len(inputs))

        results: list[dict[str, float]] = []
        for raw in raw_batch:
            results.append(self._normalize_raw_scores(self._as_score_rows(raw)))
        return results


def _aggregate_chunk_scores(chunk_scores: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate NLI scores across chunks: 60% weight on max, 40% on mean.

    Pure max lets a single loosely-matching chunk carry the verdict.
    Blending with the mean requires broader evidence consensus.
    """
    keys = ("entails", "contradicts", "neutral")
    agg: dict[str, float] = {}
    for k in keys:
        vals = [s.get(k, 0.0) for s in chunk_scores]
        agg[k] = 0.6 * max(vals) + 0.4 * (sum(vals) / len(vals))
    return agg


def nli_verify_batch(
    claims_and_chunks: list[tuple[str, list[str]]],
) -> list[dict[str, Any]]:

    if not claims_and_chunks:
        return []

    try:
        nli = NLIModel.get()
    except Exception as exc:
        logger.warning("[NLI] Model load failed, falling back to sequential: %s", exc)
        return [nli_verify(ct, ec) for ct, ec in claims_and_chunks]

    # Build flat list of (claim, evidence) pairs, tracking per-claim slice boundaries.
    flat_pairs: list[tuple[str, str]] = []
    slices: list[tuple[int, int]] = []  # (start, end) index into flat_pairs per claim
    for claim_text, evidence_chunks in claims_and_chunks:
        start = len(flat_pairs)
        for chunk in evidence_chunks:
            flat_pairs.append((claim_text, chunk))
        slices.append((start, len(flat_pairs)))

    if not flat_pairs:
        # All claims had empty evidence_chunks
        return [
            {
                "verdict": "Not-Supported",
                "verifier_used": "nli",
                "evidence_trail": [],
                "error_stage": "no_evidence",
            }
            for _ in claims_and_chunks
        ]

    try:
        all_scores = nli.classify_batch(flat_pairs)
    except Exception as exc:
        logger.warning("[NLI] Batch inference failed, falling back to sequential: %s", exc)
        return [nli_verify(ct, ec) for ct, ec in claims_and_chunks]

    if not isinstance(all_scores, list) or len(all_scores) != len(flat_pairs):
        logger.warning("[NLI] Batch inference returned unexpected shape, falling back to sequential.")
        return [nli_verify(ct, ec) for ct, ec in claims_and_chunks]

    results: list[dict[str, Any]] = []
    for (claim_text, evidence_chunks), (start, end) in zip(claims_and_chunks, slices):
        if start == end:
            results.append({
                "verdict": "Not-Supported",
                "verifier_used": "nli",
                "evidence_trail": [],
                "error_stage": "no_evidence",
            })
            continue

        chunk_scores = all_scores[start:end]
        agg = _aggregate_chunk_scores(chunk_scores)
        trail: list[dict] = []
        for chunk, scores in zip(evidence_chunks, chunk_scores):
            trail.append({"text": chunk[:200], "scores": scores})

        if agg["entails"] > GRAPH_CONFIG.nli_threshold:
            verdict = "Supported"
            reason = (
                f"NLI entailment score {agg['entails']:.2f} > threshold {GRAPH_CONFIG.nli_threshold} "
                f"across {len(evidence_chunks)} chunk(s)."
            )
        elif agg["contradicts"] > GRAPH_CONFIG.nli_contradiction_threshold:
            verdict = "Contradicted"
            reason = (
                f"NLI contradiction score {agg['contradicts']:.2f} > threshold {GRAPH_CONFIG.nli_contradiction_threshold} "
                f"across {len(evidence_chunks)} chunk(s)."
            )
        else:
            verdict = "Neutral"
            reason = (
                f"NLI scores ambiguous (entail={agg['entails']:.2f}, contradict={agg['contradicts']:.2f}) "
                f"— escalating to LLM."
            )

        results.append({
            "verdict": verdict,
            "verifier_used": "nli",
            "evidence_trail": trail,
            "error_stage": None,
            "reason": reason,
        })

    return results


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

    chunk_scores: list[dict[str, float]] = []
    trail: list[dict] = []
    for chunk in evidence_chunks:
        scores = nli.classify(claim_text, chunk)
        chunk_scores.append(scores)
        trail.append({"text": chunk[:200], "scores": scores})

    agg = _aggregate_chunk_scores(chunk_scores)

    if agg["entails"] > GRAPH_CONFIG.nli_threshold:
        verdict = "Supported"
        reason = (
            f"NLI entailment score {agg['entails']:.2f} > threshold {GRAPH_CONFIG.nli_threshold} "
            f"across {len(evidence_chunks)} chunk(s)."
        )
    elif agg["contradicts"] > GRAPH_CONFIG.nli_contradiction_threshold:
        verdict = "Contradicted"
        reason = (
            f"NLI contradiction score {agg['contradicts']:.2f} > threshold {GRAPH_CONFIG.nli_contradiction_threshold} "
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
