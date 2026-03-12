from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

from config.settings import (
    DEFAULT_EMBEDDING_MODEL,
    PATHS
)
from indexing.preprocessor import (
    build_work_id_map,
    build_paper_meta,
    build_ref_caption_map,
    make_embed_text,
    make_uid,
)
from indexing.chunker import (
    chunk_abstract,
    chunk_section,
    get_tokenizer,
)


# Qdrant schema factory

def build_chunk(
    window: dict,
    *,
    paper_id: str,
    paper_doi: str,
    paper_meta: dict,
    chunk_index: int,
    total_chunks: int,
) -> dict:

    text          = window["text"]
    section_title = window["section_title"]
    chunk_type    = window["chunk_type"]
    cite_spans    = window["cite_spans"]

    uid = make_uid(paper_id, section_title or chunk_type, text)
    return {
        "chunk_uid":      uid,
        "chunk_type":     chunk_type,
        "chunk_index":    chunk_index,   # 0-based position within this paper (paper-level)
        "total_chunks":   total_chunks,  # total chunks this paper produced
        "section_title":  section_title,
        "embed_text":     make_embed_text(section_title, text),
        "spans":          {"cite_spans": cite_spans},
        "paper_doi":      paper_doi,
        "paper_id_arxiv": paper_id,
        "title":          paper_meta["title"],
        "authors":        paper_meta["authors"],
        "categories":     paper_meta["categories"],
        "year":           paper_meta["year"],
        "cited_by_count": paper_meta["cited_by_count"],
        "language":       paper_meta["language"],
        "discipline":     paper_meta["discipline"],
    }


# Pipeline orchestrator

def build_paper_chunks(
    paper: dict,
    model_key: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict]:
    """
    Full pipeline for a single paper:

      1. preprocessor  → resolves cites, DOIs, ref captions, metadata
      2. chunker       → splits text into token-bounded windows
      3. builder       → assembles each window into a Qdrant-ready chunk dict
    """

    tokenizer       = get_tokenizer(model_key)
    bib_entries     = paper.get("bib_entries") or {}
    paper_meta      = build_paper_meta(paper)
    paper_id: str   = paper.get("paper_id") or ""
    paper_doi: str  = paper_meta["doi"]

    logger.info("Paper %s — resolving %d bib entries ...", paper_id, len(bib_entries))
    work_id_map     = build_work_id_map(bib_entries)
    ref_caption_map = build_ref_caption_map(paper.get("ref_entries") or {})

    windows: list[dict] = []

    windows.extend(
        chunk_abstract(paper, tokenizer, work_id_map, ref_caption_map)
    )
    for title, section in (paper.get("sections") or {}).items():
        if not section or not isinstance(section, dict):
            continue

        stripped = title.strip()
        normalized_title = 'Body' if (not stripped or stripped.lower() == "null") else stripped
        windows.extend(chunk_section(
            title           = normalized_title,
            section         = section,
            tokenizer       = tokenizer,
            work_id_map     = work_id_map,
            ref_caption_map = ref_caption_map,
        ))

    total_chunks = len(windows)
    chunks = [
        build_chunk(
            w,
            paper_id=paper_id,
            paper_doi=paper_doi,
            paper_meta=paper_meta,
            chunk_index=i,
            total_chunks=total_chunks,
        )
        for i, w in enumerate(windows)
    ]
    logger.info("Paper %s — %d chunks produced", paper_id, len(chunks))
    return chunks


# I/O helpers

def save_chunks(chunks: list[dict], batch_stem: str) -> None:
    out_path = PATHS.chunks / f"{batch_stem}_chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def load_chunks(batch_stem: str) -> list[dict]:
    in_path = PATHS.chunks / f"{batch_stem}_chunks.jsonl"
    if not in_path.exists():
        return []
    with open(in_path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def chunks_file_exists(batch_stem: str) -> bool:
    return (PATHS.chunks / f"{batch_stem}_chunks.jsonl").exists()