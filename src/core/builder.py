
from __future__ import annotations
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

from src.core.config import (
    CHUNKING,
    DEFAULT_EMBEDDING_MODEL,
    PATHS,
)
from src.core.preprocessor import (
    build_work_id_map,
    build_paper_meta,
    build_ref_caption_map,
    load_paper_from_batch_line,
)
from src.core.chunker import (
    chunk_abstract,
    chunk_section,
    get_tokenizer,
)


# embed_text helpers

_MULTI_SPACE_RE = re.compile(r"  +")


def make_embed_text(section_title: Optional[str], text: str) -> str:
    """
    Build the string passed to the embedding model.

      1. Collapse any double-spaces left where citation markers were removed.
      2. Prepend ``"{section_title}: "`` for positional context.
    """
    clean = _MULTI_SPACE_RE.sub(" ", text).strip()
    return f"{section_title}: {clean}" if section_title else clean


def make_uid(paper_id: str, section_label: str, text: str) -> str:

    raw = f"{paper_id}\x00{section_label}\x00{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# Qdrant schema factory

def build_chunk(
    window: dict,
    *,
    paper_id: str,
    paper_doi: str,
    paper_meta: dict,
) -> dict:

    text          = window["text"]
    section_title = window["section_title"]
    chunk_type    = window["chunk_type"]
    cite_spans    = window["cite_spans"]

    uid = make_uid(paper_id, section_title or chunk_type, text)
    return {
        "chunk_uid":      uid,
        "chunk_type":     chunk_type,
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

    chunks = [
        build_chunk(w, paper_id=paper_id, paper_doi=paper_doi, paper_meta=paper_meta)
        for w in windows
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


# CLI smoke test  (python -m src.core.builder)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence very chatty libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)

    batch_file = PATHS.batches / "batch_01.jsonl"
    if not batch_file.exists():
        print(f"ERROR: {batch_file} not found", file=sys.stderr)
        sys.exit(1)

    model_key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EMBEDDING_MODEL
    print(f"Model     : {model_key}")
    print(f"Tokenizer : {CHUNKING.tokeniser_model_id}")
    print("Loading tokenizer ...")
    tok = get_tokenizer(model_key)
    print(f"  OK {tok.__class__.__name__}")

    n_papers = 0
    n_chunks = 0
    type_counts: dict[str, int] = {}
    all_chunks: list[dict]      = []

    with open(batch_file) as fh:
        for i, raw_line in enumerate(fh):
            if i >= 5:
                break
            paper  = load_paper_from_batch_line(raw_line)
            chunks = build_paper_chunks(paper, model_key=model_key)
            n_papers += 1
            n_chunks += len(chunks)
            all_chunks.extend(chunks)
            for c in chunks:
                type_counts[c["chunk_type"]] = type_counts.get(c["chunk_type"], 0) + 1

            if i == 0 and chunks:
                c = chunks[0]
                print(f"\n=== Paper {paper['paper_id']} --- first chunk ===")
                print(f"  uid:              {c['chunk_uid']}")
                print(f"  chunk_type:       {c['chunk_type']}")
                print(f"  section_title:    {c['section_title']}")
                print(f"  embed_text[:120]: {c['embed_text'][:120]!r}")
                print(f"  cite_spans:       {c['spans']['cite_spans'][:2]}")
                print(f"  title:            {c['title']}")
                print(f"  year:             {c['year']}")
                print(f"  authors:          {c['authors'][:3]}")

    smoke_stem = f"{batch_file.stem}_smoke"
    save_chunks(all_chunks, smoke_stem)
    print(f"\nSaved {len(all_chunks)} chunks -> {PATHS.chunks / f'{smoke_stem}_chunks.jsonl'}")
    print(f"\n{'--'*20}")
    print(f"Papers processed : {n_papers}")
    print(f"Total chunks     : {n_chunks}")
    print(f"Avg chunks/paper : {n_chunks / n_papers:.1f}")
    print(f"Chunk types      : {type_counts}")
