
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

from src.core.config import (
    CHUNKING,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    PATHS
)
from src.core.preprocessor import (
    build_work_id_map,
    build_paper_meta,
    build_ref_caption_map,
    load_paper_from_batch_line,
    make_embed_text,
    make_uid,
)
from src.core.chunker import (
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


# ── Pipeline orchestration ────────────────────────────────────────────────────

def chunk_batches(
    batch_paths: list[str] | list[Path],
    model_key: str = DEFAULT_EMBEDDING_MODEL,
) -> list[str]:
    """
    Chunk a list of batch files.

    Parameters
    ----------
    batch_paths:
        List of batch file paths or stems.
        - Paths: ``["/path/to/batch_01.jsonl"]`` or ``["_data/unarxive_batches/batch_01.jsonl"]``
        - Stems: ``["batch_01", "batch_02"]`` (looks in PATHS.batches)
        - Special: ``["all"]`` to process all batch_*.jsonl in PATHS.batches
    model_key:
        Embedding model key (used for tokenizer selection).

    Returns
    -------
    list[str]
        List of batch stems that were chunked.
    """
    # Resolve paths to actual files
    batch_files: list[Path] = []

    if batch_paths == ["all"]:
        batch_files = sorted(PATHS.batches.glob("batch_*.jsonl"))
    else:
        for p in batch_paths:
            path = Path(p)
            if path.is_file():
                batch_files.append(path)
            elif path.suffix == "":
                # Stem: look in PATHS.batches
                stem_path = PATHS.batches / f"{path.name}.jsonl"
                if stem_path.is_file():
                    batch_files.append(stem_path)
                else:
                    logger.warning("Batch not found: %s", p)
            else:
                logger.warning("Not a valid batch file: %s", p)

    if not batch_files:
        logger.error("No batch files found to chunk")
        return []

    logger.info("Found %d batch files to chunk", len(batch_files))
    logger.info("Loading tokenizer for %s ...", model_key)
    tok = get_tokenizer(model_key)
    logger.info("Tokenizer ready: %s", tok.__class__.__name__)

    created_stems = []

    for batch_file in batch_files:
        stem = batch_file.stem
        chunk_file = PATHS.chunks / f"{stem}_chunks.jsonl"

        # Skip if already chunked
        if chunk_file.exists():
            logger.info(
                "Chunks for %s already exist — skipping (use --force to overwrite).",
                stem,
            )
            created_stems.append(stem)
            continue

        logger.info("Chunking %s (%s) ...", stem, batch_file.name)

        all_chunks = []
        n_papers = 0
        n_chunks = 0

        try:
            with open(batch_file, encoding="utf-8") as fh:
                for line_num, raw_line in enumerate(fh, start=1):
                    if not raw_line.strip():
                        continue

                    try:
                        paper = load_paper_from_batch_line(raw_line)
                        chunks = build_paper_chunks(paper, model_key=model_key)
                        n_papers += 1
                        n_chunks += len(chunks)
                        all_chunks.extend(chunks)
                    except Exception as exc:
                        logger.warning(
                            "Error processing paper at line %d in %s: %s",
                            line_num, stem, exc,
                        )
                        continue

            if all_chunks:
                save_chunks(all_chunks, stem)
                logger.info(
                    "%s — %d papers → %d chunks → saved to %s",
                    stem, n_papers, n_chunks, chunk_file.name,
                )
                created_stems.append(stem)
            else:
                logger.warning("%s — no chunks produced", stem)

        except Exception as exc:
            logger.error("Failed to process batch %s: %s", stem, exc)
            continue

    logger.info("\nChunking complete: %d batches processed", len(created_stems))
    return created_stems


def run_pipeline(
    batch_paths: list[str] | list[Path],
    model_key: str = DEFAULT_EMBEDDING_MODEL,
    *,
    skip_chunk: bool = False,
    skip_index: bool = False,
    recreate_collection: bool = False,
) -> int:
    """
    Full end-to-end pipeline: chunk → embed → index.

    Parameters
    ----------
    batch_paths:
        Batch file paths, stems, or ``["all"]``.
    model_key:
        Embedding model key.
    skip_chunk:
        If True, skip chunking phase (assume chunks exist).
    skip_index:
        If True, skip indexing phase (chunking only).
    recreate_collection:
        If True, drop and recreate Qdrant collection before indexing.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    from src.core.indexer import index_batches

    logger.info("=" * 60)
    logger.info("PIPELINE: Chunk → Embed → Index")
    logger.info("=" * 60)
    logger.info("Model: %s", model_key)
    logger.info("Skip chunk: %s", skip_chunk)
    logger.info("Skip index: %s", skip_index)
    logger.info("")

    # Phase 1: Chunking
    chunk_stems = []
    if not skip_chunk:
        logger.info("PHASE 1: Chunking batches …\n")
        chunk_stems = chunk_batches(batch_paths, model_key)
        if not chunk_stems:
            logger.error("No chunks created. Exiting.")
            return 1
    else:
        # Discover existing chunks
        chunk_stems = sorted(
            p.stem.replace("_chunks", "") for p in PATHS.chunks.glob("*_chunks.jsonl")
        )
        logger.info("Skipping chunking. Found %d existing chunk files.", len(chunk_stems))

    # Phase 2: Embedding + Indexing
    if not skip_index:
        logger.info("\nPHASE 2: Embedding and indexing into Qdrant …\n")
        try:
            index_batches(
                batch_stems=chunk_stems,
                model_key=model_key,
                recreate=recreate_collection,
            )
            logger.info("\nPipeline complete! ✓")
            return 0
        except RuntimeError as exc:
            logger.error("Indexing failed: %s", exc)
            return 1
    else:
        logger.info("\nSkipping indexing phase (chunks are ready in _data/chunks/).")
        return 0


# ── CLI ────────────────────────────────────────────────────────────────────────



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description=(
            "Modular chunking & pipeline orchestration. "
            "Chunk batches, then optionally embed & index into Qdrant."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        choices=list(EMBEDDING_MODELS),
        help="Embedding model key (used for tokenizer selection)",
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        required=True,
        metavar="PATH|STEM",
        help=(
            "Batch file paths, stems, or 'all'. "
            "Examples: batch_01 (stem in _data/unarxive_batches/) | "
            "_data/unarxive_batches/batch_01.jsonl (path) | "
            "all (all batch_*.jsonl files)"
        ),
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run full pipeline: chunk → embed → index (requires Qdrant)",
    )
    parser.add_argument(
        "--skip-chunk",
        action="store_true",
        help="Skip chunking phase (with --pipeline, assumes chunks exist)",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip indexing phase (chunking only)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate Qdrant collection before indexing (with --pipeline)",
    )
    args = parser.parse_args()

    if args.pipeline:
        # Full pipeline mode
        exit_code = run_pipeline(
            batch_paths=args.batches,
            model_key=args.model,
            skip_chunk=args.skip_chunk,
            skip_index=args.skip_index,
            recreate_collection=args.recreate,
        )
        sys.exit(exit_code)
    else:
        # Chunking-only mode (backward compatible with original CLI)
        if len(args.batches) == 1 and args.batches[0] not in ("all", "batch_01"):
            # Single file test mode (original behavior)
            batch_file = Path(args.batches[0])
            if not batch_file.is_file():
                batch_file = PATHS.batches / f"{args.batches[0]}.jsonl"
        else:
            batch_file = None

        if batch_file and batch_file.is_file():
            # Original smoke test on single file
            print(f"Model     : {args.model}")
            print(f"Tokenizer : {CHUNKING.tokeniser_model_id}")
            print("Loading tokenizer ...")
            tok = get_tokenizer(args.model)
            print(f"  OK {tok.__class__.__name__}")

            n_papers = 0
            n_chunks = 0
            type_counts: dict[str, int] = {}
            all_chunks: list[dict] = []

            with open(batch_file) as fh:
                for i, raw_line in enumerate(fh):
                    if i >= 5:
                        break
                    paper = load_paper_from_batch_line(raw_line)
                    chunks = build_paper_chunks(paper, model_key=args.model)
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
            print(
                f"\nSaved {len(all_chunks)} chunks -> {PATHS.chunks / f'{smoke_stem}_chunks.jsonl'}"
            )
            print(f"\n{'--' * 20}")
            print(f"Papers processed : {n_papers}")
            print(f"Total chunks     : {n_chunks}")
            print(f"Avg chunks/paper : {n_chunks / n_papers:.1f}")
            print(f"Chunk types      : {type_counts}")
        else:
            # Chunk multiple batches
            chunk_batches(args.batches, args.model)
