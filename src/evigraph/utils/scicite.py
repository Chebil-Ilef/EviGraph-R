from __future__ import annotations
import logging
from evigraph.schemas.objects import SciCiteLabel, EdgeRelation

logger = logging.getLogger(__name__)


_LABEL_MAP: dict[str, SciCiteLabel] = {
    "method":           EdgeRelation.METHOD,
    "background":       EdgeRelation.BACKGROUND,
    "result_comparison":EdgeRelation.RESULT_COMPARISON,
}
_FALLBACK: SciCiteLabel = EdgeRelation.BACKGROUND


class SciCiteModel:

    _instance: "SciCiteModel | None" = None

    def __init__(self) -> None:
        from transformers import pipeline
        from config.settings import GRAPH_CONFIG

        self._pipe = pipeline(
            "text-classification",
            model=GRAPH_CONFIG.scicite_model_id,
        )

    @classmethod
    def get(cls) -> "SciCiteModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def classify(self, citation_sentence: str) -> tuple[SciCiteLabel, float]:
        """Return (label, confidence) for a citation sentence."""
        result = self._pipe(citation_sentence[:512], truncation=True)[0]
        label = _LABEL_MAP.get((result["label"] or "").lower(), _FALLBACK)
        return label, float(result["score"])


def classify_citation(citation_sentence: str) -> tuple[SciCiteLabel, float]:

    if not citation_sentence.strip():
        return _FALLBACK, 0.0
    try:
        return SciCiteModel.get().classify(citation_sentence)
    except Exception as exc:
        logger.warning("[SCICITE] Classification failed, using fallback: %s", exc)
        return _FALLBACK, 0.0
