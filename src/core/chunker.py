from __future__ import annotations
import sys
from pathlib import Path
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import hashlib
import json
import re
from typing import Optional
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.core.config import (
    CHUNKING,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    PATHS,
)

# 1.  CONSTANTS

_CITE_RE = re.compile(r"\{\{cite:([0-9a-f]+)\}\}")
_CITE_REPLACEMENT = "[CITE]"

# Figure / table reference markers: {{figure:uuid}} or {{table:uuid}}
_REF_MARKER_RE = re.compile(r"\{\{(?:figure|table):([0-9a-f\-]+)\}\}")

# DOI patterns for mining bib_entry_raw when ids.doi is empty
_DOI_LABEL_RE  = re.compile(r"\bdoi:\s*(10\.\d{4,}/\S+)", re.IGNORECASE)
_DOI_URL_RE    = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,}/\S+)", re.IGNORECASE)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\(\\])")

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# 2.  REF-ENTRY HELPERS

def build_ref_caption_map(ref_entries: dict) -> dict[str, str]:
    caption_map: dict[str, str] = {}
    for uid, entry in (ref_entries or {}).items():
        if not entry or not isinstance(entry, dict):
            caption_map[uid] = ""
            continue
        caption_map[uid] = (entry.get("caption") or "").strip()
    return caption_map


def clean_ref_markers(text: str, ref_caption_map: dict[str, str]) -> str:
    """
    Replace ``{{figure:uuid}}`` / ``{{table:uuid}}`` markers in *text*.

    * If the uuid has a non-empty caption in *ref_caption_map*:
        → ``[Figure: <caption>]``  (or ``[Table: <caption>]``)
    * Otherwise the marker is removed entirely (no noise left in text).
    """
    def _replace(m: re.Match) -> str:  # type: ignore[type-arg]
        full_match = m.group(0)          # e.g. {{figure:abc-123}}
        uuid       = m.group(1)
        # Determine label from the original marker text
        label = "Figure" if "figure:" in full_match else "Table"
        caption = ref_caption_map.get(uuid, "")
        if caption:
            return f"[{label}: {caption}]"
        return ""  # strip invisible hash noise

    return _REF_MARKER_RE.sub(_replace, text)


# 3.  TOKENIZER (lazy, module-level cache)

_TOKENIZER_CACHE: dict[str, PreTrainedTokenizerBase] = {}


def get_tokenizer(model_key: str = DEFAULT_EMBEDDING_MODEL) -> PreTrainedTokenizerBase:

    if model_key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_key]

    cfg = EMBEDDING_MODELS[model_key]

    hf_id = (
        cfg.hf_model_id
        if CHUNKING.tokeniser_model_id == "auto"
        else CHUNKING.tokeniser_model_id
    )

    cache_dir = str(cfg.local_cache_dir)

    tok = AutoTokenizer.from_pretrained(
        hf_id,
        cache_dir=cache_dir,
        trust_remote_code=True,   
    )
    _TOKENIZER_CACHE[model_key] = tok
    return tok


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# 3.  CITE RESOLUTION + TEXT PROCESSING

def _parse_doi_from_raw(raw: str) -> str:
    """
    Last-resort DOI extraction from a raw bibliography string when ids.doi is empty.

    Tries, in order:
      1. ``doi: 10.xxx/yyy``  — explicit BibTeX / bibliography label
      2. ``https://doi.org/10.xxx/yyy``  — doi.org URL in the raw string

    Trailing punctuation (.,:;) stripped from the match.
    Returns "" when nothing is found.
    """
    for pattern in (_DOI_LABEL_RE, _DOI_URL_RE):
        m = pattern.search(raw)
        if m:
            doi = m.group(1).rstrip(".,;:")
            if doi:
                return doi
    return ""


def build_doi_map(bib_entries: dict) -> dict[str, str]:
    """
    Build {ref_id → doi_string} from the paper's bib_entries dict.

    Resolution order for each entry:
      1. ids.doi  (may look like "https://doi.org/10.xxx" — normalised below)
      2. _parse_doi_from_raw(bib_entry_raw)  — mine the raw BibTeX string
      3. ids.arxiv_id  (stored as "arxiv:{id}")
      4. ""  if nothing found

    Normalisation: strip leading https://doi.org/ or http://dx.doi.org/
    """
    doi_map: dict[str, str] = {}
    for ref_id, bib in bib_entries.items():
        if not bib or not isinstance(bib, dict):
            doi_map[ref_id] = ""
            continue

        ids = bib.get("ids") or {}
        raw_doi: str = ids.get("doi") or ""

        # Normalise doi URL prefixes
        for prefix in (
            "https://doi.org/",
            "http://doi.org/",
            "https://dx.doi.org/",
            "http://dx.doi.org/",
        ):
            if raw_doi.startswith(prefix):
                raw_doi = raw_doi[len(prefix):]
                break

        if raw_doi:
            doi_map[ref_id] = raw_doi
            continue

        # ids.doi was empty — try mining the raw bibliography string
        mined = _parse_doi_from_raw(bib.get("bib_entry_raw") or "")
        if mined:
            doi_map[ref_id] = mined
            continue

        # No doi found anywhere
        doi_map[ref_id] = ""

    return doi_map


def build_arxiv_id_map(bib_entries: dict) -> dict[str, str]:
    """
    Build {ref_id → arxiv_id_string} from bib_entries ids.arxiv_id.
    Returns "" for entries with no arxiv id.
    """
    arxiv_map: dict[str, str] = {}
    for ref_id, bib in (bib_entries or {}).items():
        if not bib or not isinstance(bib, dict):
            arxiv_map[ref_id] = ""
            continue
        ids = bib.get("ids") or {}
        arxiv_map[ref_id] = ids.get("arxiv_id") or ""
    return arxiv_map


def process_text(
    text: str,
    doi_map: dict[str, str],
    arxiv_id_map: dict[str, str] | None = None,
) -> tuple[str, list[dict]]:
    """
    Replace every ``{{cite:ref_id}}`` in *text* with ``[CITE]`` and collect
    resolved cite spans with character offsets relative to the *processed* text.

    Returns
    -------
    processed_text : str
        Original text with all ``{{cite:…}}`` replaced by ``[CITE]``.
    cite_spans : list[dict]
        [{start, end, doi, arxiv_id}] — offsets into *processed_text*.
    """
    parts: list[str] = []
    cite_spans: list[dict] = []
    offset_shift = 0   # cumulative delta from replacements so far
    last_end = 0

    for m in _CITE_RE.finditer(text):
        ref_id = m.group(1)
        orig_start, orig_end = m.start(), m.end()

        # Text before this marker
        parts.append(text[last_end:orig_start])

        # New start position in the output string
        new_start = orig_start + offset_shift
        parts.append(_CITE_REPLACEMENT)
        new_end = new_start + len(_CITE_REPLACEMENT)

        cite_spans.append({
            "start":    new_start,
            "end":      new_end,
            "doi":      doi_map.get(ref_id, ""),
            "arxiv_id": (arxiv_id_map or {}).get(ref_id, ""),
        })

        # Every replacement shifts subsequent offsets by this delta
        offset_shift += len(_CITE_REPLACEMENT) - (orig_end - orig_start)
        last_end = orig_end

    parts.append(text[last_end:])
    return "".join(parts), cite_spans


def _window_spans(
    all_spans: list[dict],
    char_start: int,
    char_end: int,
) -> list[dict]:

    result = []
    for sp in all_spans:
        if sp["start"] >= char_start and sp["end"] <= char_end:
            result.append({
                "start":    sp["start"] - char_start,
                "end":      sp["end"]   - char_start,
                "doi":      sp["doi"],
                "arxiv_id": sp.get("arxiv_id", ""),
            })
    return result


# 4.  SENTENCE SPLITTING + SLIDING WINDOW

def split_sentences(text: str) -> list[tuple[int, int]]:
    """
    Return a list of (char_start, char_end) for each sentence in *text*.

    Boundary heuristic: split at [.!?] followed by whitespace + [A-Z0-9(\\].
    This avoids splitting on "e.g., " "Fig. 2" "Eq. (3)" etc.
    Trailing whitespace is excluded from each sentence.
    """
    boundaries = [0] + [m.end() for m in _SENTENCE_BOUNDARY.finditer(text)] + [len(text)]
    sentences = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1]
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

    total = count_tokens(text, tokenizer)
    if total <= max_tokens:
        return [(0, len(text))]

    sentences = split_sentences(text)
    if not sentences:
        return [(0, len(text))]

    windows: list[tuple[int, int]] = []
    i = 0  # sentence index for window start

    while i < len(sentences):
        win_char_start = sentences[i][0]
        j = i
        current_tokens = 0

        # Grow window until we hit the token limit
        while j < len(sentences):
            sent = text[sentences[j][0]:sentences[j][1]]
            sent_tok = count_tokens(sent, tokenizer)
            if current_tokens + sent_tok > window_tokens and j > i:
                break
            current_tokens += sent_tok
            j += 1

        win_char_end = sentences[j - 1][1]
        windows.append((win_char_start, win_char_end))

        if j >= len(sentences):
            break   # consumed all sentences

        # Advance past overlap: drop sentences from the front until we've
        # moved forward by at least (window_tokens - overlap_tokens) tokens
        target_drop = window_tokens - overlap_tokens
        dropped = 0
        k = i
        while k < j and dropped < target_drop:
            sent = text[sentences[k][0]:sentences[k][1]]
            dropped += count_tokens(sent, tokenizer)
            k += 1

        i = max(k, i + 1)   # always advance by at least one sentence

    return windows


# 5.  METADATA HELPERS

def _extract_year(metadata: dict) -> Optional[int]:

    for version in metadata.get("versions") or []:
        m = _YEAR_RE.search(version.get("created") or "")
        if m:
            return int(m.group())
    # Fallback: update_date "2024-10-01"
    date_str = metadata.get("update_date") or ""
    m = _YEAR_RE.match(date_str)
    if m:
        return int(m.group())
    return None


def _extract_authors(metadata: dict) -> list[str]:

    parsed = metadata.get("authors_parsed") or []
    if parsed:
        names = []
        for entry in parsed:
            last  = (entry[0] if len(entry) > 0 else "").strip()
            first = (entry[1] if len(entry) > 1 else "").strip()
            name  = f"{first} {last}".strip() if first else last
            if name:
                names.append(name)
        return names
    raw: str = metadata.get("authors") or ""
    return [a.strip() for a in re.split(r"[,;]", raw) if a.strip()]


def _extract_categories(metadata: dict) -> list[str]:

    cats = metadata.get("categories") or ""
    if isinstance(cats, list):
        return cats
    return cats.split()


def build_paper_meta(paper: dict) -> dict:

    meta = paper.get("metadata") or {}
    return {
        "doi":            meta.get("doi") or "",
        "title":          meta.get("title") or "",
        "authors":        _extract_authors(meta),
        "categories":     _extract_categories(meta),
        "year":           _extract_year(meta),
        "cited_by_count": meta.get("cited_by_count"),
        "language":       meta.get("language"),
        "discipline":     meta.get("discipline"),
    }

_CITE_STRIP_RE = re.compile(r"\[CITE\]")
_MULTI_SPACE_RE = re.compile(r"  +")


def _make_embed_text(section_title: Optional[str], text: str) -> str:
    """
    Build the string that gets passed to the embedding model.

    Steps:
      1. Strip all ``[CITE]`` placeholders (zero semantic value for vectors).
      2. Collapse any double-spaces left behind.
      3. Prepend ``"{section_title}: "`` so the model sees positional context
         ("Methods: ...", "Abstract: ...", "Related Work: ...").
    """
    clean = _CITE_STRIP_RE.sub("", text)
    clean = _MULTI_SPACE_RE.sub(" ", clean).strip()
    if section_title:
        return f"{section_title}: {clean}"
    return clean


def make_uid(paper_id: str, section_label: str, text: str) -> str:
  
    raw = f"{paper_id}\x00{section_label}\x00{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# 7.  CHUNK BUILDERS

def _build_chunk(
    *,
    paper_id: str,
    paper_doi: str,
    chunk_type: str,
    section_title: Optional[str],
    text: str,
    cite_spans: list[dict],
    paper_meta: dict,
) -> dict:

    uid = make_uid(paper_id, section_title or chunk_type, text)
    return {
        "chunk_uid":       uid,
        "chunk_type":      chunk_type,
        "section_title":   section_title,
        "embed_text":      _make_embed_text(section_title, text),
        "spans": {
            "cite_spans": cite_spans,
        },
        "paper_doi":       paper_doi,
        "paper_id_arxiv":  paper_id,
        "title":           paper_meta["title"],
        "authors":         paper_meta["authors"],
        "categories":      paper_meta["categories"],
        "year":            paper_meta["year"],
        "cited_by_count":  paper_meta["cited_by_count"],
        "language":        paper_meta["language"],
        "discipline":      paper_meta["discipline"],
    }


def chunk_abstract(
    paper: dict,
    tokenizer: PreTrainedTokenizerBase,
    doi_map: dict[str, str],
    paper_meta: dict,
    ref_caption_map: dict[str, str] | None = None,
    arxiv_id_map: dict[str, str] | None = None,
) -> list[dict]:
    """
        ≤ abstract_max_tokens → single chunk
        > abstract_max_tokens → sliding_window(window_size, overlap)
    """
    abstract = paper.get("abstract") or {}
    raw_text: str = abstract.get("text") or ""
    if not raw_text.strip():
        return []

    paper_id:  str = paper.get("paper_id") or ""
    paper_doi: str = (paper.get("metadata") or {}).get("doi") or ""

    # Replace / strip figure & table markers before cite processing
    if ref_caption_map:
        raw_text = clean_ref_markers(raw_text, ref_caption_map)

    proc_text, all_spans = process_text(raw_text, doi_map, arxiv_id_map)

    windows = sliding_window(
        proc_text,
        tokenizer,
        max_tokens   = CHUNKING.abstract_max_tokens,
        window_tokens= CHUNKING.abstract_max_tokens,
        overlap_tokens= CHUNKING.abstract_overlap,
    )

    chunks = []
    for char_start, char_end in windows:
        chunk_text = proc_text[char_start:char_end].strip()
        if not chunk_text:
            continue
        span_list = _window_spans(all_spans, char_start, char_start + len(chunk_text))
        chunks.append(
            _build_chunk(
                paper_id      = paper_id,
                paper_doi     = paper_doi,
                chunk_type    = "abstract",
                section_title = "Abstract",   # always labelled, never None
                text          = chunk_text,
                cite_spans    = span_list,
                paper_meta    = paper_meta,
            )
        )
    return chunks


def chunk_section(
    title: str,
    section: dict,
    tokenizer: PreTrainedTokenizerBase,
    doi_map: dict[str, str],
    paper_meta: dict,
    paper_id: str,
    paper_doi: str,
    ref_caption_map: dict[str, str] | None = None,
    arxiv_id_map: dict[str, str] | None = None,
) -> list[dict]:
    """
        ≤ section_max_tokens  → single chunk
        > section_max_tokens  → sliding_window(window_tokens, overlap_tokens)
    """
    raw_text: str = section.get("text") or ""
    if not raw_text.strip():
        return []

    # Replace / strip figure & table markers before cite processing
    if ref_caption_map:
        raw_text = clean_ref_markers(raw_text, ref_caption_map)

    proc_text, all_spans = process_text(raw_text, doi_map, arxiv_id_map)

    windows = sliding_window(
        proc_text,
        tokenizer,
        max_tokens    = CHUNKING.section_max_tokens,
        window_tokens = CHUNKING.section_window_size,
        overlap_tokens= CHUNKING.section_overlap_tokens,
    )

    chunks = []
    for char_start, char_end in windows:
        chunk_text = proc_text[char_start:char_end].strip()
        if not chunk_text:
            continue
        span_list = _window_spans(all_spans, char_start, char_start + len(chunk_text))
        chunks.append(
            _build_chunk(
                paper_id      = paper_id,
                paper_doi     = paper_doi,
                chunk_type    = "subsection",
                section_title = title,
                text          = chunk_text,
                cite_spans    = span_list,
                paper_meta    = paper_meta,
            )
        )
    return chunks

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


# PUBLIC API

def load_paper_from_batch_line(line: str) -> dict:

    outer = json.loads(line)
    return json.loads(outer["jsonl"])



def chunk_paper(
    paper: dict,
    model_key: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict]:

    tokenizer       = get_tokenizer(model_key)
    bib_entries     = paper.get("bib_entries") or {}
    doi_map         = build_doi_map(bib_entries)
    arxiv_id_map    = build_arxiv_id_map(bib_entries)
    ref_caption_map = build_ref_caption_map(paper.get("ref_entries") or {})
    paper_meta      = build_paper_meta(paper)
    paper_id:  str  = paper.get("paper_id") or ""
    paper_doi: str  = paper_meta["doi"]

    chunks: list[dict] = []

    # Abstract
    chunks.extend(
        chunk_abstract(paper, tokenizer, doi_map, paper_meta, ref_caption_map, arxiv_id_map)
    )

    # Sections
    for title, section in (paper.get("sections") or {}).items():
        if not section or not isinstance(section, dict):
            continue
        chunks.extend(
            chunk_section(
                title           = title,
                section         = section,
                tokenizer       = tokenizer,
                doi_map         = doi_map,
                paper_meta      = paper_meta,
                paper_id        = paper_id,
                paper_doi       = paper_doi,
                ref_caption_map = ref_caption_map,
                arxiv_id_map    = arxiv_id_map,
            )
        )

    return chunks


# 9.  CLI SMOKE TEST  (python -m src.core.chunker)

if __name__ == "__main__":
    import sys
    from pathlib import Path

    batch_file = PATHS.batches / "batch_01.jsonl"
    if not batch_file.exists():
        print(f"ERROR: {batch_file} not found", file=sys.stderr)
        sys.exit(1)

    model_key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EMBEDDING_MODEL
    print(f"Model: {model_key}")
    print(f"Tokenizer: {CHUNKING.tokeniser_model_id}")
    print("Loading tokenizer …")
    tok = get_tokenizer(model_key)
    print(f"  ✓ {tok.__class__.__name__}")

    n_papers = 0
    n_chunks = 0
    type_counts: dict[str, int] = {}
    batch_stem = batch_file.stem          # "batch_01"
    all_chunks: list[dict] = []

    with open(batch_file) as fh:
        for i, raw_line in enumerate(fh):
            if i >= 5:          # smoke-test: first 5 papers
                break
            paper = load_paper_from_batch_line(raw_line)
            chunks = chunk_paper(paper, model_key=model_key)
            n_papers += 1
            n_chunks += len(chunks)
            all_chunks.extend(chunks)
            for c in chunks:
                type_counts[c["chunk_type"]] = type_counts.get(c["chunk_type"], 0) + 1

            # Print first chunk of first paper
            if i == 0 and chunks:
                c = chunks[0]
                print(f"\n=== Paper {paper['paper_id']} — first chunk ===")
                print(f"  uid:           {c['chunk_uid']}")
                print(f"  chunk_type:    {c['chunk_type']}")
                print(f"  section_title: {c['section_title']}")
                print(f"  embed_text[:120]: {c['embed_text'][:120]!r}")
                print(f"  cite_spans:    {c['spans']['cite_spans'][:2]}")
                print(f"  title:         {c['title']}")
                print(f"  year:          {c['year']}")
                print(f"  authors:       {c['authors'][:3]}")

    # Save to _data/chunks/batch_01_smoke_chunks.jsonl
    smoke_stem = f"{batch_stem}_smoke"
    save_chunks(all_chunks, smoke_stem)
    saved_path = PATHS.chunks / f"{smoke_stem}_chunks.jsonl"
    print(f"\nSaved {len(all_chunks)} chunks → {saved_path}")

    print(f"\n{'─'*40}")
    print(f"Papers processed : {n_papers}")
    print(f"Total chunks     : {n_chunks}")
    print(f"Avg chunks/paper : {n_chunks / n_papers:.1f}")
    print(f"Chunk types      : {type_counts}")
