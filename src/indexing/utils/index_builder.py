from __future__ import annotations
import logging
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import DEFAULT_EMBEDDING_MODEL
from indexing.utils.chunker import chunk_abstract, chunk_section, get_tokenizer
from indexing.preprocessing.preprocessor import make_embed_text, make_uid, normalize_paper

logger = logging.getLogger(__name__)


def build_chunk(
    window,
    *,
    paper_id: str,
    paper_doi: str,
    paper_meta: dict,
    chunk_index: int,
    total_chunks: int,
) -> dict:
    uid = make_uid(
        paper_id,
        window.section_title or window.chunk_type,
        window.text,
        chunk_index=chunk_index,
    )
    return {
        "chunk_uid": uid,
        "chunk_type": window.chunk_type,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "section_title": window.section_title,
        "embed_text": make_embed_text(window.text),
        "spans": {"cite_spans": window.cite_spans},
        "paper_doi": paper_doi,
        "paper_id_arxiv": paper_id,
        "title": paper_meta["title"],
        "authors": paper_meta["authors"],
        "categories": paper_meta["categories"],
        "year": paper_meta["year"],
        "language": paper_meta["language"],
        "discipline": paper_meta["discipline"],
    }


def build_paper_chunks(
    paper: dict,
    model_key: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict]:
    tokenizer = get_tokenizer(model_key)
    normalized = normalize_paper(paper)

    windows = chunk_abstract(
        normalized,
        tokenizer,
        normalized.citations_by_ref,
    )
    for section in normalized.sections:
        windows.extend(
            chunk_section(
                section=section,
                tokenizer=tokenizer,
                citation_lookup=normalized.citations_by_ref,
                ref_caption_map=normalized.ref_captions,
            )
        )

    total_chunks = len(windows)
    chunks = [
        build_chunk(
            window,
            paper_id=normalized.paper_id,
            paper_doi=normalized.paper_doi,
            paper_meta=normalized.meta,
            chunk_index=index,
            total_chunks=total_chunks,
        )
        for index, window in enumerate(windows)
    ]
    logger.info("Paper %s — %d chunks produced", normalized.paper_id, len(chunks))
    return chunks
