from __future__ import annotations
import re
import sys
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from src.core.config import (
    CHUNKING,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
)
from src.core.preprocessor import (
    clean_ref_markers,
    process_text,
)
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# Tokenizer (lazy module-level cache)
_TOKENIZER_CACHE: dict[str, PreTrainedTokenizerBase] = {}


def get_tokenizer(model_key: str = DEFAULT_EMBEDDING_MODEL) -> PreTrainedTokenizerBase:
    if model_key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_key]

    cfg   = EMBEDDING_MODELS[model_key]
    hf_id = (
        cfg.hf_model_id
        if CHUNKING.tokeniser_model_id == "auto"
        else CHUNKING.tokeniser_model_id
    )

    tok = AutoTokenizer.from_pretrained(
        hf_id,
        cache_dir=str(cfg.local_cache_dir),
        trust_remote_code=True,
    )
    _TOKENIZER_CACHE[model_key] = tok
    return tok


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# Sentence splitting + sliding window

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\(\\])")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """
    Return ``[(char_start, char_end)]`` for each sentence in *text*.

    Splits at ``[.!?]`` followed by whitespace + ``[A-Z0-9(\\]``.
    Avoids splitting on "e.g., " "Fig. 2" "Eq. (3)" etc.
    """
    boundaries = [0] + [m.end() for m in _SENTENCE_BOUNDARY.finditer(text)] + [len(text)]
    sentences  = []
    for i in range(len(boundaries) - 1):
        s        = boundaries[i]
        e        = boundaries[i + 1]
        stripped = text[s:e].rstrip()
        if stripped:
            sentences.append((s, s + len(stripped)))
    return sentences


def sliding_window(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    window_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int]]:
    """
    Return ``[(char_start, char_end)]`` for each window over *text*.

    If *text* fits within *max_tokens*, returns a single full-text window.
    Otherwise uses sentence-aware sliding with *overlap_tokens* of overlap.
    """
    if count_tokens(text, tokenizer) <= max_tokens:
        return [(0, len(text))]

    sentences = split_sentences(text)
    if not sentences:
        return [(0, len(text))]

    windows: list[tuple[int, int]] = []
    i = 0

    while i < len(sentences):
        win_char_start = sentences[i][0]
        j              = i
        current_tokens = 0

        while j < len(sentences):
            sent_tok = count_tokens(text[sentences[j][0]:sentences[j][1]], tokenizer)
            if current_tokens + sent_tok > window_tokens and j > i:
                break
            current_tokens += sent_tok
            j += 1

        windows.append((win_char_start, sentences[j - 1][1]))

        if j >= len(sentences):
            break

        # Advance: drop sentences until we've moved at least
        # (window_tokens - overlap_tokens) tokens forward.
        target_drop = window_tokens - overlap_tokens
        dropped = 0
        k = i
        while k < j and dropped < target_drop:
            dropped += count_tokens(text[sentences[k][0]:sentences[k][1]], tokenizer)
            k += 1

        i = max(k, i + 1)

    return windows


def _window_spans(
    all_spans: list[dict],
    char_start: int,
    char_end: int,
) -> list[dict]:

    return [
        {
            "start":       sp["start"] - char_start,
            "end":         sp["end"]   - char_start,
            "work_id":     sp.get("work_id", ""),
            "doi":         sp.get("doi", ""),
            "openalex_id": sp.get("openalex_id", ""),
            "arxiv_id":    sp.get("arxiv_id", ""),
        }
        for sp in all_spans
        if sp["start"] >= char_start and sp["end"] <= char_end
    ]


# Per-section splitters : return raw windows

def chunk_abstract(
    paper: dict,
    tokenizer: PreTrainedTokenizerBase,
    work_id_map: dict[str, dict],
    ref_caption_map: dict[str, str] | None = None,
) -> list[dict]:

    abstract = paper.get("abstract") or {}
    raw_text: str = abstract.get("text") or ""
    if not raw_text.strip():
        return []

    if ref_caption_map:
        raw_text = clean_ref_markers(raw_text, ref_caption_map)

    proc_text, all_spans = process_text(raw_text, work_id_map)

    windows = sliding_window(
        proc_text,
        tokenizer,
        max_tokens    = CHUNKING.abstract_max_tokens,
        window_tokens = CHUNKING.abstract_max_tokens,
        overlap_tokens= CHUNKING.abstract_overlap,
    )

    results = []
    for char_start, char_end in windows:
        chunk_text = proc_text[char_start:char_end].strip()
        if not chunk_text:
            continue
        results.append({
            "text":          chunk_text,
            "cite_spans":    _window_spans(all_spans, char_start, char_start + len(chunk_text)),
            "section_title": "Abstract",
            "chunk_type":    "abstract",
        })
    return results


def chunk_section(
    title: str,
    section: dict,
    tokenizer: PreTrainedTokenizerBase,
    work_id_map: dict[str, dict],
    ref_caption_map: dict[str, str] | None = None,
) -> list[dict]:

    raw_text: str = section.get("text") or ""
    if not raw_text.strip():
        return []

    if ref_caption_map:
        raw_text = clean_ref_markers(raw_text, ref_caption_map)

    proc_text, all_spans = process_text(raw_text, work_id_map)

    windows = sliding_window(
        proc_text,
        tokenizer,
        max_tokens    = CHUNKING.section_max_tokens,
        window_tokens = CHUNKING.section_window_size,
        overlap_tokens= CHUNKING.section_overlap_tokens,
    )

    results = []
    for char_start, char_end in windows:
        chunk_text = proc_text[char_start:char_end].strip()
        if not chunk_text:
            continue
        results.append({
            "text":          chunk_text,
            "cite_spans":    _window_spans(all_spans, char_start, char_start + len(chunk_text)),
            "section_title": title,
            "chunk_type":    "subsection",
        })
    return results



