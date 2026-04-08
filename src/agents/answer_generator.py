from __future__ import annotations
import json
import logging
from typing import Any

from config.prompts import ANSWER_GENERATOR_SYSTEM_PROMPT, build_answer_generator_user_prompt
from config.settings import AGENT_MODELS
from schemas.objects import (
    AnnotatedSentence,
    Citation,
    EvidenceEdge,
    EvidenceGraph,
    FinalAnswer,
    NodeType,
    RetrievedDocument,
)
from utils.llm import LLMClient, get_llm_client
from utils.latex_sanitizer import safe_json_loads, desanitize_sentence_for_display

logger = logging.getLogger(__name__)

_CONFLICT_RELATIONS = {"contradicts", "CONTRADICTS"}


class AnswerGeneratorAgent:

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.config = AGENT_MODELS["answer_generator"]

  
    def generate(
        self,
        query: str,
        sub_queries: list,
        evidence_graph: EvidenceGraph,
        documents: list[RetrievedDocument],
    ) -> FinalAnswer:

        claims = self._collect_claims(evidence_graph, documents)

        if not claims:
            logger.warning("[ANSWER GENERATOR] No verified claims found — returning fallback answer")
            return FinalAnswer(
                text="Insufficient verified evidence to answer.",
                sentences=[],
                reasoning_summary="No verified claims available.",
            )

        try:
            sentences = self._generate_sentences(query, claims)
        except Exception as exc:
            logger.warning("[ANSWER GENERATOR] Generation failed: %s", exc)
            return FinalAnswer(
                text="Insufficient verified evidence to answer.",
                sentences=[],
                reasoning_summary=f"Generation error: {exc}",
            )

        annotated_sentences, answer_text = self._assemble(sentences)

        return FinalAnswer(
            text=answer_text,
            sentences=annotated_sentences,
            reasoning_summary=None,
        )


    def _collect_claims(
        self,
        evidence_graph: EvidenceGraph,
        documents: list[RetrievedDocument],
    ) -> list[dict[str, Any]]:

        if not evidence_graph or not evidence_graph.nodes:
            return []

        # Build lookup: chunk_id → RetrievedDocument
        doc_by_chunk: dict[str, RetrievedDocument] = {d.chunk_id: d for d in documents}

        # Build edge lookup: node_id → outgoing edges (for relation / score)
        edges_from: dict[str, list[EvidenceEdge]] = {}
        for edge in evidence_graph.edges:
            edges_from.setdefault(edge.source, []).append(edge)

        # Collect contradiction flags: set of node_ids that have a CONTRADICTS edge
        conflict_nodes: set[str] = {
            e.source
            for e in evidence_graph.edges
            if e.relation in _CONFLICT_RELATIONS
        }

        claims: list[dict[str, Any]] = []

        for node in evidence_graph.nodes:
            if node.node_type != NodeType.CLAIM:
                continue
            if not node.text.strip():
                continue

            chunk_id = node.chunk_id or node.node_id
            doc = doc_by_chunk.get(chunk_id)

            # Pick best outgoing edge for scicite_label + rel_score
            scicite_label = "extracted_from"
            rel_score = 0.0
            for edge in edges_from.get(node.node_id, []):
                if edge.score >= rel_score:
                    rel_score = edge.score
                    scicite_label = edge.relation

            claims.append({
                "text": node.text.strip(),
                "chunk_id": chunk_id,
                "doc_id": doc.doc_id if doc else (node.doc_id or ""),
                "section_title": doc.section_title if doc else node.metadata.get("section_title"),
                "scicite_label": scicite_label,
                "rel_score": rel_score,
                "verdict": node.metadata.get("verdict", "Supported"),
                "conflict": node.node_id in conflict_nodes,
            })

        return claims

    def _generate_sentences(
        self,
        query: str,
        claims: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        raw = self.llm_client.chat_text(
            model=self.config.model,
            system_prompt=ANSWER_GENERATOR_SYSTEM_PROMPT,
            user_prompt=build_answer_generator_user_prompt(query, claims),
            temperature=self.config.temperature,
            timeout=self.config.timeout_seconds,
        )

        return self._parse_sentences_json(raw)


    @staticmethod
    def _assemble(
        sentences: list[dict[str, Any]],
    ) -> tuple[list[AnnotatedSentence], str]:

        texts: list[str] = []
        annotated: list[AnnotatedSentence] = []

        for s in sentences:
            text = s.get("text", "").strip()
            if not text:
                continue
            texts.append(text)

            # Build citations for this sentence
            chunk_id = s.get("chunk_id", "")
            doc_id = s.get("doc_id", "")
            
            citation = Citation(
                doc_id=doc_id,
                chunk_id=chunk_id or None,
                section_title=s.get("section_title"),
                scicite_label=s.get("scicite_label"),
                rel_score=s.get("rel_score"),
                verdict=s.get("verdict"),
                title=None,
            )
            
            annotated_sentence = AnnotatedSentence(
                text=text,
                citations=[citation] if doc_id else [],
                conflict_flag=s.get("conflict", False),
            )
            annotated.append(annotated_sentence)

        answer_text = " ".join(texts) if texts else "Insufficient verified evidence to answer."
        return annotated, answer_text

    @staticmethod
    def _parse_sentences_json(raw: str) -> list[dict[str, Any]]:

        try:
            payload = safe_json_loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model did not return valid JSON: {e}") from e

        sentences = payload.get("sentences", [])
        if not isinstance(sentences, list):
            raise ValueError("'sentences' key is not a list")

        # Desanitize LaTeX content in sentences
        return [desanitize_sentence_for_display(s) for s in sentences]
