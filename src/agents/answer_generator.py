from __future__ import annotations
import json
import logging
import re
import time
from collections import defaultdict
from typing import Any

from config.prompts import ANSWER_GENERATOR_SYSTEM_PROMPT, build_answer_generator_user_prompt
from config.settings import AGENT_MODELS
from schemas.objects import (
    AnnotatedSentence,
    Citation,
    CONFLICT_RELATIONS,
    EdgeRelation,
    EvidenceEdge,
    EvidenceGraph,
    FinalAnswer,
    JUDGE_EDGE_PREFIX,
    NodeType,
    RetrievedDocument,
)
from utils.llm import LLMClient, get_llm_client
from utils.latex_sanitizer import safe_json_loads, desanitize_sentence_for_display

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would could should "
    "may might shall can of in on at to for with by from and or not".split()
)


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _claim_score(claim: dict[str, Any]) -> float:
    sq_texts: list[str] = claim.get("sub_query_texts") or []
    claim_text: str = claim["text"]
    sq_relevance = max((_jaccard(sq, claim_text) for sq in sq_texts), default=0.0)
    inconclusive_penalty = 0.5 if claim.get("confidence") == "low" else 0.0
    return claim.get("doc_score", 0.0) + 0.3 * sq_relevance - inconclusive_penalty


def _select_claims(
    claims: list[dict[str, Any]],
    sub_queries: list,
    max_total: int,
    min_per_subquery: int,
) -> list[dict[str, Any]]:
    ranked = sorted(claims, key=_claim_score, reverse=True)
    n_sq = max(len(sub_queries), 1)
    floor = max(min_per_subquery, max_total // n_sq)

    selected: list[dict] = []
    seen: set[str] = set()
    sq_fill: dict[int, int] = defaultdict(int)

    # Pass 1: guarantee floor claims per sub-query
    for claim in ranked:
        if claim["text"] in seen:
            continue
        for sq_idx in (claim["sub_query_indices"] or [0]):
            if sq_fill[sq_idx] < floor:
                selected.append(claim)
                seen.add(claim["text"])
                sq_fill[sq_idx] += 1
                break

    # Pass 2: fill remaining budget globally by score
    for claim in ranked:
        if len(selected) >= max_total:
            break
        if claim["text"] not in seen:
            selected.append(claim)
            seen.add(claim["text"])

    return selected


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
        verdict_details: dict | None = None,
    ) -> FinalAnswer:

        t0 = time.perf_counter()
        claims = self._collect_claims(evidence_graph, documents, verdict_details or {}, sub_queries)
        t_collect = time.perf_counter()
        logger.info(
            "[ANSWER GENERATOR] Collected %d supported claims in %.3fs",
            len(claims),
            t_collect - t0,
        )

        if not claims:
            logger.warning("[ANSWER GENERATOR] No verified claims found — returning fallback answer")
            return FinalAnswer(
                text="Something went wrong while generating the answer or there was insufficient verified evidence to give a reply. Please try again or adjust the question.",
                sentences=[],
                reasoning_summary="No verified claims available.",
            )

        try:
            t_gen0 = time.perf_counter()
            sentences = self._generate_sentences(query, claims)
            t_gen1 = time.perf_counter()
            logger.info(
                "[ANSWER GENERATOR] LLM generation returned %d sentences in %.3fs",
                len(sentences),
                t_gen1 - t_gen0,
            )
        except Exception as exc:
            logger.warning("[ANSWER GENERATOR] Generation failed: %s", exc)
            return FinalAnswer(
                text="Something went wrong while generating the answer or there was insufficient verified evidence to give a reply. Please try again or adjust the question.",
                sentences=[],
                reasoning_summary=f"Generation error: {exc}",
            )

        annotated_sentences, answer_text = self._assemble(sentences, claims)
        logger.info(
            "[ANSWER GENERATOR] Total answer generation time: %.3fs",
            time.perf_counter() - t0,
        )

        return FinalAnswer(
            text=answer_text,
            sentences=annotated_sentences,
            reasoning_summary=None,
        )


    def _collect_claims(
        self,
        evidence_graph: EvidenceGraph,
        documents: list[RetrievedDocument],
        verdict_details: dict[str, Any] | None = None,
        sub_queries: list | None = None,
    ) -> list[dict[str, Any]]:

        if not evidence_graph or not evidence_graph.nodes:
            return []

        verdict_details = verdict_details or {}

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
            if e.relation in CONFLICT_RELATIONS
        }

        claims: list[dict[str, Any]] = []

        for node in evidence_graph.nodes:
            if node.node_type != NodeType.CLAIM:
                continue
            if not node.text.strip():
                continue

            verdict = None
            if node.node_id in verdict_details:
                verdict = verdict_details[node.node_id].get("verdict")
            if not verdict:
                verdict = node.metadata.get("verdict")
            if verdict not in ("Supported", "Inconclusive"):
                continue

            chunk_id = node.chunk_id or node.node_id
            doc = doc_by_chunk.get(chunk_id)

            scicite_label = None
            rel_score = 0.0
            _valid_scicite = {EdgeRelation.METHOD.value, EdgeRelation.BACKGROUND.value, EdgeRelation.RESULT_COMPARISON.value}
            for edge in edges_from.get(node.node_id, []):
                if edge.relation.startswith(JUDGE_EDGE_PREFIX):
                    continue
                if edge.relation in _valid_scicite:
                    scicite_label = edge.relation
                    rel_score = edge.score
                    break

            claim_text = node.text.strip()
            claims.append({
                "text": claim_text,
                "chunk_id": chunk_id,
                "doc_id": doc.doc_id if doc else (node.doc_id or ""),
                "section_title": doc.section_title if doc else node.metadata.get("section_title"),
                "paper_title": doc.paper_title if doc else None,
                "chunk_content": doc.content if doc else None,
                "scicite_label": scicite_label,
                "rel_score": rel_score,
                "doc_score": doc.score if doc else 0.0,
                "verdict": verdict,
                "confidence": "low" if verdict == "Inconclusive" else "high",
                "conflict": node.node_id in conflict_nodes,
                "sub_query_indices": node.metadata.get("sub_query_indices") or [],
                "sub_query_texts": node.metadata.get("sub_query_texts") or [],
            })

        return _select_claims(
            claims,
            sub_queries or [],
            self.config.answer_max_claims_total,
            self.config.answer_min_claims_per_subquery,
        )

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
            max_tokens=self.config.max_tokens,
        )

        return self._parse_sentences_json(raw)


    @staticmethod
    def _assemble(
        sentences: list[dict[str, Any]],
        claims: list[dict[str, Any]] | None = None,
    ) -> tuple[list[AnnotatedSentence], str]:

        texts: list[str] = []
        annotated: list[AnnotatedSentence] = []
        claims = claims or []

        for s in sentences:
            text = s.get("text", "").strip()
            if not text:
                continue
            texts.append(text)

            citations, conflict_flag = AnswerGeneratorAgent._citations_for_sentence(s, claims)
            annotated_sentence = AnnotatedSentence(
                text=text,
                citations=citations,
                conflict_flag=conflict_flag,
            )
            annotated.append(annotated_sentence)

        answer_text = " ".join(texts) if texts else "Insufficient verified evidence to answer."
        return annotated, answer_text

    @staticmethod
    def _citations_for_sentence(
        sentence: dict[str, Any],
        claims: list[dict[str, Any]],
    ) -> tuple[list[Citation], bool]:
        claim_refs = sentence.get("claim_refs", [])
        if isinstance(claim_refs, int):
            claim_refs = [claim_refs]
        if not isinstance(claim_refs, list):
            claim_refs = []

        citations: list[Citation] = []
        seen: set[tuple[str, str]] = set()
        conflict_flag = False

        for ref in claim_refs:
            if not isinstance(ref, int):
                continue
            idx = ref - 1
            if idx < 0 or idx >= len(claims):
                continue
            claim = claims[idx]
            key = (claim.get("doc_id", "") or "", claim.get("chunk_id", "") or "")
            if key in seen:
                continue
            seen.add(key)
            conflict_flag = conflict_flag or bool(claim.get("conflict", False))
            citations.append(
                Citation(
                    doc_id=claim.get("doc_id", ""),
                    chunk_id=claim.get("chunk_id") or None,
                    section_title=claim.get("section_title"),
                    scicite_label=claim.get("scicite_label"),
                    rel_score=claim.get("rel_score"),
                    verdict=claim.get("verdict"),
                    title=claim.get("paper_title"),
                    claim_text=claim.get("text"),
                    chunk_content=claim.get("chunk_content"),
                )
            )

        if citations:
            return citations, conflict_flag

        # Backward compatibility: if the model still returns full citation metadata,
        # preserve the old behavior instead of failing the whole answer.
        chunk_id = sentence.get("chunk_id", "")
        doc_id = sentence.get("doc_id", "")
        if doc_id:
            return (
                [
                    Citation(
                        doc_id=doc_id,
                        chunk_id=chunk_id or None,
                        section_title=sentence.get("section_title"),
                        scicite_label=sentence.get("scicite_label"),
                        rel_score=sentence.get("rel_score"),
                        verdict=sentence.get("verdict"),
                        title=sentence.get("paper_title"),
                        claim_text=sentence.get("text"),
                        chunk_content=sentence.get("chunk_content"),
                    )
                ],
                bool(sentence.get("conflict", False)),
            )

        return [], bool(sentence.get("conflict", False))

    @staticmethod
    def _parse_sentences_json(raw: str) -> list[dict[str, Any]]:

        try:
            payload = safe_json_loads(raw)
        except json.JSONDecodeError as e:
            # Log the problematic response for debugging
            logger.error(
                f"[ANSWER GENERATOR] JSON parse error at position {e.pos}. "
                f"Raw response: {raw[:500]}..."
            )
            raise ValueError(f"Model did not return valid JSON: {e}") from e

        if not isinstance(payload, dict):
            raise ValueError(f"Response is not a JSON object, got: {type(payload).__name__}")
        
        sentences = payload.get("sentences", [])
        if not isinstance(sentences, list):
            raise ValueError("'sentences' key is not a list")

        # Desanitize LaTeX content in sentences
        return [desanitize_sentence_for_display(s) for s in sentences]
