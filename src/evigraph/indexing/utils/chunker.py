from __future__ import annotations
import re
import sys
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from evigraph.config.settings import CHUNKING, DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS
from evigraph.indexing.utils.models import ChunkWindow, NormalizedPaper, NormalizedSection
from evigraph.indexing.preprocessing.preprocessor import clean_ref_markers, process_text

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_TOKENIZER_CACHE: dict[str, PreTrainedTokenizerBase] = {}
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\(\\])")


def get_tokenizer(model_key: str = DEFAULT_EMBEDDING_MODEL) -> PreTrainedTokenizerBase:
    if model_key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_key]

    cfg = EMBEDDING_MODELS[model_key]
    hf_id = cfg.hf_model_id if CHUNKING.tokeniser_model_id == "auto" else CHUNKING.tokeniser_model_id
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        cache_dir=str(cfg.local_cache_dir),
        trust_remote_code=True,
    )
    _TOKENIZER_CACHE[model_key] = tokenizer
    return tokenizer


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_sentences(text: str) -> list[tuple[int, int]]:
    boundaries = [0] + [match.end() for match in _SENTENCE_BOUNDARY.finditer(text)] + [len(text)]
    sentences: list[tuple[int, int]] = []
    for index in range(len(boundaries) - 1):
        start = boundaries[index]
        end = boundaries[index + 1]
        stripped = text[start:end].rstrip()
        if stripped:
            sentences.append((start, start + len(stripped)))
    return sentences


def sliding_window(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    window_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int]]:
    if count_tokens(text, tokenizer) <= max_tokens:
        return [(0, len(text))]

    sentences = split_sentences(text)
    if not sentences:
        return [(0, len(text))]

    windows: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(sentences):
        win_char_start = sentences[cursor][0]
        end_index = cursor
        current_tokens = 0

        while end_index < len(sentences):
            sent_start, sent_end = sentences[end_index]
            sent_tokens = count_tokens(text[sent_start:sent_end], tokenizer)
            if current_tokens + sent_tokens > window_tokens and end_index > cursor:
                break
            current_tokens += sent_tokens
            end_index += 1

        windows.append((win_char_start, sentences[end_index - 1][1]))
        if end_index >= len(sentences):
            break

        target_drop = window_tokens - overlap_tokens
        dropped = 0
        next_cursor = cursor
        while next_cursor < end_index and dropped < target_drop:
            sent_start, sent_end = sentences[next_cursor]
            dropped += count_tokens(text[sent_start:sent_end], tokenizer)
            next_cursor += 1
        cursor = max(next_cursor, cursor + 1)

    return windows


def _window_spans(all_spans: list[dict], char_start: int, char_end: int) -> list[dict]:
    results: list[dict] = []
    for span in all_spans:
        if span["start"] >= char_start and span["end"] <= char_end:
            shifted = {**span}
            shifted["start"] = span["start"] - char_start
            shifted["end"] = span["end"] - char_start
            results.append(shifted)
    return results


def chunk_abstract(
    paper: NormalizedPaper,
    tokenizer: PreTrainedTokenizerBase,
    citation_lookup: dict[str, dict],
) -> list[ChunkWindow]:
    if not paper.abstract_text.strip():
        return []

    cleaned = clean_ref_markers(paper.abstract_text, paper.ref_captions)
    processed_text, all_spans = process_text(cleaned, citation_lookup)
    windows = sliding_window(
        processed_text,
        tokenizer,
        max_tokens=CHUNKING.abstract_max_tokens,
        window_tokens=CHUNKING.abstract_max_tokens,
        overlap_tokens=CHUNKING.abstract_overlap,
    )
    return _build_windows(processed_text, all_spans, windows, "Abstract", "abstract")


def chunk_section(
    section: NormalizedSection,
    tokenizer: PreTrainedTokenizerBase,
    citation_lookup: dict[str, dict],
    ref_caption_map: dict[str, str],
) -> list[ChunkWindow]:
    if not section.text.strip():
        return []

    cleaned = clean_ref_markers(section.text, ref_caption_map)
    processed_text, all_spans = process_text(cleaned, citation_lookup)
    windows = sliding_window(
        processed_text,
        tokenizer,
        max_tokens=CHUNKING.section_max_tokens,
        window_tokens=CHUNKING.section_window_size,
        overlap_tokens=CHUNKING.section_overlap_tokens,
    )
    return _build_windows(processed_text, all_spans, windows, section.title, "subsection")


def _build_windows(
    text: str,
    all_spans: list[dict],
    windows: list[tuple[int, int]],
    section_title: str,
    chunk_type: str,
) -> list[ChunkWindow]:
    results: list[ChunkWindow] = []
    for char_start, char_end in windows:
        chunk_text = text[char_start:char_end].strip()
        if not chunk_text:
            continue
        results.append(
            ChunkWindow(
                text=chunk_text,
                cite_spans=_window_spans(all_spans, char_start, char_start + len(chunk_text)),
                section_title=section_title,
                chunk_type=chunk_type,
            )
        )
    return results
