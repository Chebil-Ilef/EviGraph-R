from __future__ import annotations
import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

from src.config.settings import CHUNKING, DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS

load_dotenv()


_CITE_RE = re.compile(r"\{\{cite:([0-9a-f]+)\}\}")
_FIGURE_RE = re.compile(r"\{\{figure:[0-9a-f\-]+\}\}")
_TABLE_RE = re.compile(r"\{\{table:[0-9a-f\-]+\}\}")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_ARXIV_RE = re.compile(
    r"(?:arxiv[^\d]*)?([0-9]{4}\.[0-9]{4,5}|(?:hep-ph|hep-th|cs|math|eess|stat|physics)/[0-9]{7})",
    re.I,
)
_DOI_RE = re.compile(r"10\.\d{4,}/\S+")
_ARXIV_NEW_RE = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}")

_IMRAD_KEYWORDS = [
    "introduction",
    "related work",
    "background",
    "method",
    "approach",
    "experiment",
    "result",
    "evaluation",
    "discussion",
    "conclusion",
    "abstract",
    "acknowledgement",
    "reference",
    "appendix",
]

_NOISE_TITLE_RE = re.compile(
    r"^(\d+\.?\s*)*$|^\s*$|^[ivxIVX]+\.?\s*$|^\W+$|^(nan|none|null)$",
    re.I,
)

_CHUNK_WINDOW = CHUNKING.section_window_size
_CHUNK_OVERLAP = CHUNKING.section_overlap_tokens
_CHUNK_NO_SPLIT = CHUNKING.section_max_tokens
_WORDS_TO_TOKENS = 1.30  # empirical for English scientific prose

_TARGET_PAPERS = 2_338_911
_VECTOR_DIM = EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL].dim
_BYTES_FLOAT32 = 4
_BYTES_INT8 = 1

_UNARXIVE_REPO_ID = "ines-besrour/unarxive_2024"
_UNARXIVE_TAR_PARTS = [
    "unarXive_2024.tar.gz.part_aa",
    "unarXive_2024.tar.gz.part_ab",
    "unarXive_2024.tar.gz.part_ac",
]
_UNARXIVE_TAR_NAME = "unarXive_2024.tar.gz"

# Reservoir sampling caps for memory safety on full-scale runs
_BODY_WORD_SAMPLE_LIMIT = 200_000
_EST_CHUNK_SAMPLE_LIMIT = 200_000
_SECTION_COUNT_SAMPLE_LIMIT = 200_000
_SECTION_WORD_SAMPLE_LIMIT = 300_000
_CITE_COUNT_SAMPLE_LIMIT = 200_000


@dataclass
class Reservoir:
    limit: int
    seen: int = 0
    values: list[int] = field(default_factory=list)

    def add(self, value: int) -> None:
        self.seen += 1
        if len(self.values) < self.limit:
            self.values.append(value)
            return
        j = random.randint(1, self.seen)
        if j <= self.limit:
            self.values[j - 1] = value

    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def percentile(self, pct: float) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        idx = max(0, min(len(s) - 1, int(len(s) * pct / 100)))
        return s[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "seen": self.seen,
            "values": self.values,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Reservoir":
        return cls(
            limit=int(data["limit"]),
            seen=int(data["seen"]),
            values=list(data["values"]),
        )


@dataclass
class Stats:
    total: int = 0
    has_abstract: int = 0
    abstract_empty: int = 0
    has_body: int = 0
    body_empty: int = 0
    has_year_meta: int = 0
    has_year_derived: int = 0
    has_authors: int = 0
    has_doi: int = 0
    has_arxiv_id: int = 0
    has_categories: int = 0
    has_title: int = 0

    total_sections: int = 0
    sections_with_empty_title: int = 0
    sections_with_noise_title: int = 0
    sections_imrad_matched: int = 0
    section_title_counter: Counter = field(default_factory=Counter)
    papers_with_no_section_titles: int = 0

    total_cite_markers: int = 0
    papers_with_no_citations: int = 0

    total_bib_entries: int = 0
    bib_has_doi: int = 0
    bib_has_arxiv: int = 0
    bib_has_openalex: int = 0
    bib_has_title: int = 0
    bib_has_year: int = 0
    bib_has_any_id: int = 0
    bib_sha_only: int = 0
    bib_field_counter: Counter = field(default_factory=Counter)

    total_cite_ref_ids: int = 0
    cite_ref_resolved_doi: int = 0
    cite_ref_resolved_arxiv: int = 0
    cite_ref_resolved_openalex: int = 0
    cite_ref_sha_only: int = 0

    total_figure_markers: int = 0
    total_table_markers: int = 0

    category_counter: Counter = field(default_factory=Counter)

    # Memory-safe sampled distributions
    body_word_counts: Reservoir = field(default_factory=lambda: Reservoir(_BODY_WORD_SAMPLE_LIMIT))
    est_chunk_counts: Reservoir = field(default_factory=lambda: Reservoir(_EST_CHUNK_SAMPLE_LIMIT))
    section_counts: Reservoir = field(default_factory=lambda: Reservoir(_SECTION_COUNT_SAMPLE_LIMIT))
    section_word_counts: Reservoir = field(default_factory=lambda: Reservoir(_SECTION_WORD_SAMPLE_LIMIT))
    cite_counts_per_paper: Reservoir = field(default_factory=lambda: Reservoir(_CITE_COUNT_SAMPLE_LIMIT))
    cite_marker_counts_per_paper: Reservoir = field(default_factory=lambda: Reservoir(_CITE_COUNT_SAMPLE_LIMIT))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "has_abstract": self.has_abstract,
            "abstract_empty": self.abstract_empty,
            "has_body": self.has_body,
            "body_empty": self.body_empty,
            "has_year_meta": self.has_year_meta,
            "has_year_derived": self.has_year_derived,
            "has_authors": self.has_authors,
            "has_doi": self.has_doi,
            "has_arxiv_id": self.has_arxiv_id,
            "has_categories": self.has_categories,
            "has_title": self.has_title,
            "total_sections": self.total_sections,
            "sections_with_empty_title": self.sections_with_empty_title,
            "sections_with_noise_title": self.sections_with_noise_title,
            "sections_imrad_matched": self.sections_imrad_matched,
            "section_title_counter": dict(self.section_title_counter),
            "papers_with_no_section_titles": self.papers_with_no_section_titles,
            "total_cite_markers": self.total_cite_markers,
            "papers_with_no_citations": self.papers_with_no_citations,
            "total_bib_entries": self.total_bib_entries,
            "bib_has_doi": self.bib_has_doi,
            "bib_has_arxiv": self.bib_has_arxiv,
            "bib_has_openalex": self.bib_has_openalex,
            "bib_has_title": self.bib_has_title,
            "bib_has_year": self.bib_has_year,
            "bib_has_any_id": self.bib_has_any_id,
            "bib_sha_only": self.bib_sha_only,
            "bib_field_counter": dict(self.bib_field_counter),
            "total_cite_ref_ids": self.total_cite_ref_ids,
            "cite_ref_resolved_doi": self.cite_ref_resolved_doi,
            "cite_ref_resolved_arxiv": self.cite_ref_resolved_arxiv,
            "cite_ref_resolved_openalex": self.cite_ref_resolved_openalex,
            "cite_ref_sha_only": self.cite_ref_sha_only,
            "total_figure_markers": self.total_figure_markers,
            "total_table_markers": self.total_table_markers,
            "category_counter": dict(self.category_counter),
            "body_word_counts": self.body_word_counts.to_dict(),
            "est_chunk_counts": self.est_chunk_counts.to_dict(),
            "section_counts": self.section_counts.to_dict(),
            "section_word_counts": self.section_word_counts.to_dict(),
            "cite_counts_per_paper": self.cite_counts_per_paper.to_dict(),
            "cite_marker_counts_per_paper": self.cite_marker_counts_per_paper.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Stats":
        obj = cls()
        for k, v in data.items():
            if k in {"section_title_counter", "bib_field_counter", "category_counter"}:
                setattr(obj, k, Counter(v))
            elif k in {
                "body_word_counts",
                "est_chunk_counts",
                "section_counts",
                "section_word_counts",
                "cite_counts_per_paper",
                "cite_marker_counts_per_paper",
            }:
                setattr(obj, k, Reservoir.from_dict(v))
            else:
                setattr(obj, k, v)
        return obj


def _load_hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.getenv(key)
        if v and v.strip():
            os.environ.setdefault("HF_TOKEN", v.strip())
            return v.strip()
    return None


def _first_jsonl(raw: str) -> dict | None:
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _normalise_title(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"^[\d\.\s]+", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _is_noise(t: str) -> bool:
    return bool(_NOISE_TITLE_RE.match(t.strip()))


def _imrad(t: str) -> bool:
    return any(k in t.lower() for k in _IMRAD_KEYWORDS)


def _derive_year(arxiv_id: str) -> int | None:
    s = arxiv_id.strip()
    m = _ARXIV_NEW_RE.match(s)
    if m:
        yy = int(m.group(1))
        return 2000 + yy if yy < 90 else 1900 + yy
    m2 = re.search(r"/(\d{2})(\d{2})\d{3}", s)
    if m2:
        yy = int(m2.group(1))
        return 2000 + yy if yy < 90 else 1900 + yy
    return None


def _doi_from_bib(bib: dict) -> str | None:
    for key in ("doi", "DOI", "ids"):
        v = bib.get(key)
        if isinstance(v, str):
            m = _DOI_RE.search(v)
            if m:
                return m.group(0)
        elif isinstance(v, dict):
            d = v.get("doi") or v.get("DOI")
            if d:
                return str(d)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    m = _DOI_RE.search(item)
                    if m:
                        return m.group(0)

    links = bib.get("contained_links") or []
    if isinstance(links, list):
        for link in links:
            if isinstance(link, str):
                m = _DOI_RE.search(link)
                if m:
                    return m.group(0)

    raw = bib.get("raw_text") or bib.get("raw") or ""
    m = _DOI_RE.search(raw)
    return m.group(0) if m else None


def _arxiv_from_bib(bib: dict) -> str | None:
    for key in ("arxiv_id", "arxivId", "eprint", "ids"):
        v = bib.get(key)
        if isinstance(v, str) and v.strip():
            m = _ARXIV_RE.search(v)
            if m:
                return m.group(0)
        elif isinstance(v, dict):
            a = v.get("arxiv") or v.get("arxiv_id") or v.get("eprint")
            if a:
                return str(a)

    contained = bib.get("contained_arXiv_ids") or []
    if isinstance(contained, list):
        for item in contained:
            if isinstance(item, str) and item.strip():
                m = _ARXIV_RE.search(item)
                if m:
                    return m.group(0)

    raw = bib.get("raw_text") or bib.get("raw") or ""
    m = _ARXIV_RE.search(raw)
    return m.group(0) if m else None


def _openalex_from_bib(bib: dict) -> str | None:
    for key in ("openalex_id", "open_alex_id", "openAlexId", "ids"):
        v = bib.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            oa = v.get("openalex_id") or v.get("open_alex_id") or v.get("openAlexId")
            if oa:
                return str(oa)
    return None


def _title_from_bib(bib: dict) -> str | None:
    for key in ("title", "bib_entry_raw"):
        v = bib.get(key)
        if isinstance(v, str) and len(v.strip()) > 5:
            return v.strip()
    return None


def _est_chunks(word_count: int) -> int:
    tokens = int(word_count * _WORDS_TO_TOKENS)
    if tokens <= _CHUNK_NO_SPLIT:
        return 1
    stride = _CHUNK_WINDOW - _CHUNK_OVERLAP
    return math.ceil((tokens - _CHUNK_WINDOW) / stride) + 1


def _pct(num: int, denom: int) -> str:
    return "N/A" if denom == 0 else f"{100 * num / denom:.1f}%"


def _pctf(num: int, denom: int) -> float:
    return 0.0 if denom == 0 else 100 * num / denom


def _cite_concentration(counts: list[int]) -> tuple[float, float]:
    if not counts or sum(counts) == 0:
        return 0.0, 0.0
    s = sorted(counts, reverse=True)
    total = sum(s)
    n10 = max(1, len(s) // 10)
    n25 = max(1, len(s) // 4)
    return 100 * sum(s[:n10]) / total, 100 * sum(s[:n25]) / total


def _storage_gb(n_vecs: int, bpd: int) -> float:
    return n_vecs * _VECTOR_DIM * bpd * 1.30 / 1e9


def _ensure_local_tar(local_dir: str, verbose: bool = False) -> Path:

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: pip install huggingface_hub to use --local-dir/UNARXIVE_LOCAL_DIR", file=sys.stderr)
        sys.exit(1)

    root = Path(local_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    tar_path = root / _UNARXIVE_TAR_NAME

    if tar_path.exists() and tar_path.stat().st_size > 0:
        if verbose:
            print(f"[info] Using existing local tar: {tar_path}", flush=True)
        return tar_path

    if verbose:
        print(f"[info] Local tar not found, downloading parts into {root} ...", flush=True)

    part_paths: list[Path] = []
    for fname in _UNARXIVE_TAR_PARTS:
        if verbose:
            print(f"  [download] {fname}", flush=True)
        part_path_str = hf_hub_download(
            repo_id=_UNARXIVE_REPO_ID,
            filename=fname,
            repo_type="dataset",
            cache_dir=str(root),
        )
        part_paths.append(Path(part_path_str))

    if verbose:
        print(f"[info] Concatenating {len(part_paths)} parts → {tar_path}", flush=True)

    with open(tar_path, "wb") as out_f:
        for part in part_paths:
            with open(part, "rb") as in_f:
                while True:
                    chunk = in_f.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)

    if verbose:
        print(f"[info] Local tar ready: {tar_path} (size={tar_path.stat().st_size / 1e9:.1f} GB)", flush=True)

    return tar_path


def _iter_tar_jsonl_rows(tar_path: Path, verbose: bool = False):

    import tarfile
    import gzip

    if verbose:
        print(f"[info] Opening tar for streaming: {tar_path}", flush=True)

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            if not (name.endswith(".jsonl") or name.endswith(".jsonl.gz")):
                continue

            if verbose:
                print(f"[info] Streaming from member: {name}", flush=True)

            file_obj = tf.extractfile(member)
            if file_obj is None:
                continue

            if name.endswith(".gz"):
                stream = gzip.open(file_obj, "rb")  
            else:
                stream = file_obj

            for line in stream:
                if not line:
                    continue
                try:
                    text = line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
                if not text:
                    continue
                # Match the existing HF streaming contract
                yield {"jsonl": text}


def _stream_rows(dataset_name: str, split: str, verbose: bool = False, local_dir: str | None = None):

    # Prefer explicit local_dir (or env-var) to avoid relying on HF preview-only rows.
    if local_dir:
        tar_path = _ensure_local_tar(local_dir, verbose=verbose)
        return _iter_tar_jsonl_rows(tar_path, verbose=verbose)

    from datasets import load_dataset
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Dataset loading timed out after 120 seconds")

    try:
        if verbose:
            print("[info] Loading dataset via streaming...", flush=True)

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)  # 120 second timeout

        ds = load_dataset(
            dataset_name,
            split=split,
            streaming=True,
            keep_in_memory=False,
        )

        signal.alarm(0)  # Cancel the alarm

        if verbose:
            print("[info] ✓ Streaming dataset loaded", flush=True)
        return iter(ds)
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm if still set
        print(f"[ERROR] Failed to load dataset: {e}", file=sys.stderr)
        raise


def _save_checkpoint(
    path: str | None,
    stats: Stats,
    skipped: int,
    stream_rows_seen: int,
    started_at: str,
) -> None:
    if not path:
        return
    payload = {
        "started_at": started_at,
        "saved_at": datetime.now().isoformat(),
        "skipped": skipped,
        "stream_rows_seen": stream_rows_seen,
        "stats": stats.to_dict(),
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def _load_checkpoint(path: str | None) -> tuple[Stats, int, int, str] | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    stats = Stats.from_dict(payload["stats"])
    skipped = int(payload["skipped"])
    stream_rows_seen = int(payload["stream_rows_seen"])
    started_at = str(payload.get("started_at") or datetime.now().isoformat())
    return stats, skipped, stream_rows_seen, started_at

def analyse_paper(paper: dict, s: Stats) -> None:
    s.total += 1

    meta = paper.get("metadata") or {}

    if (meta.get("title") or paper.get("title") or "").strip():
        s.has_title += 1

    if meta.get("authors") or paper.get("authors"):
        s.has_authors += 1

    year_raw = meta.get("year") or meta.get("date") or paper.get("year") or ""
    if year_raw and _YEAR_RE.search(str(year_raw)):
        s.has_year_meta += 1

    if (meta.get("doi") or paper.get("doi") or "").strip():
        s.has_doi += 1

    arxiv_id = str(
        meta.get("arxiv_id")
        or meta.get("id")
        or paper.get("paper_id")
        or paper.get("_id")
        or ""
    ).strip()

    if arxiv_id and _ARXIV_RE.search(arxiv_id):
        s.has_arxiv_id += 1
        if _derive_year(arxiv_id) is not None:
            s.has_year_derived += 1

    cats = meta.get("categories") or paper.get("categories") or []
    if cats:
        s.has_categories += 1
        if isinstance(cats, str):
            cats = cats.split()
        for cat in cats[:3]:
            s.category_counter[str(cat).split(".")[0].lower()] += 1

    # Abstract
    ab = paper.get("abstract") or {}
    ab_text = (ab.get("text") or ab.get("section") or "") if isinstance(ab, dict) else str(ab)
    if ab_text.strip():
        s.has_abstract += 1
    else:
        s.abstract_empty += 1

    # Sections
    secs_raw = paper.get("sections") or {}
    secs: list[dict] = []

    if isinstance(secs_raw, dict):
        for ttl, obj in secs_raw.items():
            if isinstance(obj, dict):
                secs.append({"section": ttl, "text": obj.get("text") or ""})
    elif isinstance(paper.get("body_text"), list):
        secs = paper["body_text"]

    total_words = 0
    n_secs = 0
    all_noise = True
    cite_set: set[str] = set()
    cite_marker_count = 0
    chunks_est = 1  # abstract chunk

    for sec in secs:
        n_secs += 1
        ttl = (sec.get("section") or sec.get("section_title") or "").strip()
        text = (sec.get("text") or "").strip()
        wc = len(text.split())

        s.section_word_counts.add(wc)
        chunks_est += _est_chunks(wc)

        if not ttl:
            s.sections_with_empty_title += 1
        elif _is_noise(ttl):
            s.sections_with_noise_title += 1
        else:
            all_noise = False
            norm = _normalise_title(ttl)
            s.section_title_counter[norm] += 1
            if _imrad(ttl):
                s.sections_imrad_matched += 1

        cites = _CITE_RE.findall(text)
        cite_set.update(cites)
        s.total_cite_markers += len(cites)
        cite_marker_count += len(cites)
        s.total_figure_markers += len(_FIGURE_RE.findall(text))
        s.total_table_markers += len(_TABLE_RE.findall(text))
        total_words += wc

    s.total_sections += n_secs

    if total_words:
        s.has_body += 1
    else:
        s.body_empty += 1

    if total_words and secs and all_noise:
        s.papers_with_no_section_titles += 1

    if not cite_set:
        s.papers_with_no_citations += 1

    s.cite_counts_per_paper.add(len(cite_set))
    s.cite_marker_counts_per_paper.add(cite_marker_count)

    if n_secs:
        s.section_counts.add(n_secs)

    if total_words:
        s.body_word_counts.add(total_words)

    s.est_chunk_counts.add(chunks_est)

    # Bibliography
    bibs = paper.get("bib_entries") or {}
    if isinstance(bibs, list):
        bibs = {e.get("ref_id", str(i)): e for i, e in enumerate(bibs) if isinstance(e, dict)}

    for _, bib in bibs.items():
        if not isinstance(bib, dict):
            continue

        s.total_bib_entries += 1

        for k, v in bib.items():
            if bool(v):
                s.bib_field_counter[k] += 1

        doi = _doi_from_bib(bib)
        arxiv = _arxiv_from_bib(bib)
        openalex = _openalex_from_bib(bib)
        title = _title_from_bib(bib)
        year = bib.get("year") or bib.get("Year") or ""

        if doi:
            s.bib_has_doi += 1
        if arxiv:
            s.bib_has_arxiv += 1
        if openalex:
            s.bib_has_openalex += 1
        if title:
            s.bib_has_title += 1
        if year and _YEAR_RE.search(str(year)):
            s.bib_has_year += 1
        if doi or arxiv or openalex:
            s.bib_has_any_id += 1
        else:
            s.bib_sha_only += 1

    s.total_cite_ref_ids += len(cite_set)

    for ref_id in cite_set:
        bib = bibs.get(ref_id)
        if bib and isinstance(bib, dict):
            doi = _doi_from_bib(bib)
            arxiv = _arxiv_from_bib(bib)
            openalex = _openalex_from_bib(bib)
            if doi:
                s.cite_ref_resolved_doi += 1
            elif arxiv:
                s.cite_ref_resolved_arxiv += 1
            elif openalex:
                s.cite_ref_resolved_openalex += 1
            else:
                s.cite_ref_sha_only += 1
        else:
            s.cite_ref_sha_only += 1


def build_report(s: Stats, n_requested: int) -> str:
    N = s.total
    B = s.total_bib_entries
    R = s.total_cite_ref_ids

    lines: list[str] = []
    L = lines.append

    L("# unarXive 2024 — Data Quality Report")
    L("")
    L(f"> **Analysed:** {N:,} papers (requested {n_requested:,})  ")
    L(f"> **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    L(f"> **Scale target:** {_TARGET_PAPERS:,} papers  ")
    L(
        f"> **Chunk config:** window={_CHUNK_WINDOW} tok · overlap={_CHUNK_OVERLAP} tok · no-split≤{_CHUNK_NO_SPLIT} tok  "
    )
    L(f"> **Embedding:** e5-base-v2 · dim={_VECTOR_DIM} · float32")
    L("")
    L("---")
    L("##  0 · Dataset Schema")
    L("")
    L("Each paper is a JSONL row. The relevant top-level keys are:")
    L("")
    L("| Key | Type | Notes |")
    L("|-----|------|-------|")
    L("| `paper_id` | string | arXiv ID, e.g. `2310.00826` — primary identifier |")
    L("| `metadata` | dict | Title, authors, categories, DOI — **no `year` field exists** |")
    L("| `abstract` | dict | `{text, cite_spans, ref_spans}` |")
    L("| `sections` | dict | `{title → {text, cite_spans, ref_spans}}` — **all body content lives here** |")
    L("| `bib_entries` | dict | `{sha_hex → {bib_entry_raw, ids{doi, arxiv_id, open_alex_id}}}` |")
    L("| `ref_entries` | dict | Figure/table captions, keyed by UUID |")
    L("")
    L("> **Body text:** There is no separate `body_text` field. All body word counts and chunk estimates")
    L("> in this report are computed by summing across all entries in the `sections` dict.")
    L(">")
    L("> **Year:** Not present in `metadata`. Fully derivable from the arXiv ID YYMM prefix:")
    L("> `2310.00826` → year **2023**, month **10**. The preprocessor must derive and store this.")
    L("")
    L("---")
    L("##  1 · Paper Metadata")
    L("")
    L("| Field | Count | % of papers |")
    L("|-------|------:|------------:|")
    L(f"| Title | **{s.has_title:,}** | {_pct(s.has_title, N)} |")
    L(f"| Authors | **{s.has_authors:,}** | {_pct(s.has_authors, N)} |")
    L(f"| Year (metadata field) | **{s.has_year_meta:,}** | {_pct(s.has_year_meta, N)} |")
    L(f"| Year (derived from arXiv ID YYMM) | **{s.has_year_derived:,}** | {_pct(s.has_year_derived, N)} |")
    L(f"| Paper DOI | **{s.has_doi:,}** | {_pct(s.has_doi, N)} |")
    L(f"| Paper arXiv ID | **{s.has_arxiv_id:,}** | {_pct(s.has_arxiv_id, N)} |")
    L(f"| Categories | **{s.has_categories:,}** | {_pct(s.has_categories, N)} |")
    L("")
    if s.has_year_meta == 0 and s.has_year_derived > 0:
        L(f"> ⚠️ **Year is 0% in metadata** but {_pct(s.has_year_derived, N)} derivable from arXiv ID.")
        L("> Add `_derive_year(paper_id)` to the preprocessor and store result in the chunk payload.")
        L("")
    L("---")
    L("##  2 · Abstract")
    L("")
    L("| | Count | % |")
    L("|--|------:|--:|")
    L(f"| Non-empty abstract | **{s.has_abstract:,}** | {_pct(s.has_abstract, N)} |")
    L(f"| Empty / missing | **{s.abstract_empty:,}** | {_pct(s.abstract_empty, N)} |")
    L("")
    L("---")
    L("##  3 · Body Text")
    L("")
    L("> **What is body word count?** Sum of word counts across all `sections` entries for a paper.")
    L("> There is no separate body field — this is computed at analysis time.")
    L(">")
    L("> **What are p10, p25, p50, p75, p90?** These are *percentiles* estimated from a memory-safe sample.")
    L("")
    L("| | Count | % |")
    L("|--|------:|--:|")
    L(f"| Has body text | **{s.has_body:,}** | {_pct(s.has_body, N)} |")
    L(f"| Empty / missing | **{s.body_empty:,}** | {_pct(s.body_empty, N)} |")
    L("")

    if s.body_word_counts.values:
        L("| Statistic | Words |")
        L("|-----------|------:|")
        L(f"| Mean | **{s.body_word_counts.mean():,.0f}** |")
        L(f"| p10 | {s.body_word_counts.percentile(10):,.0f} |")
        L(f"| p25 | {s.body_word_counts.percentile(25):,.0f} |")
        L(f"| p50 (median) | **{s.body_word_counts.percentile(50):,.0f}** |")
        L(f"| p75 | {s.body_word_counts.percentile(75):,.0f} |")
        L(f"| p90 | {s.body_word_counts.percentile(90):,.0f} |")
        L("")

    L("---")
    L("##  4 · Section Structure")
    L("")
    L("> **What are section titles?** Each key of the `sections` dict is a section title as it")
    L("> appeared in the paper — e.g. `\"Introduction\"`, `\"The Obstacle Problem\"`, `\"3.1 Setup\"`.")
    L("> The pipeline stores these in `chunk_section` for filtered retrieval.")
    L(">")
    L("> **IMRAD titles** match standard scientific paper structure keywords (Introduction,")
    L("> Method, Results, Discussion, etc.). These are the reliable section-level filters.")
    L(">")
    L("> **Noise title** = a section title that is purely numeric, roman numeral, or punctuation.")
    L("")
    L("| Metric | Value |")
    L("|--------|------:|")
    L(f"| Total sections | **{s.total_sections:,}** |")
    L(f"| Avg sections / paper | **{s.section_counts.mean():.1f}** |")
    if s.section_counts.values:
        L(
            f"| Sections per paper (p10 / p50 / p90) | "
            f"{s.section_counts.percentile(10):.0f} / **{s.section_counts.percentile(50):.0f}** / {s.section_counts.percentile(90):.0f} |"
        )
    L(f"| With empty title | {s.sections_with_empty_title:,} ({_pct(s.sections_with_empty_title, s.total_sections)}) |")
    L(f"| With noise title | {s.sections_with_noise_title:,} ({_pct(s.sections_with_noise_title, s.total_sections)}) |")
    L(f"| With IMRAD match | {s.sections_imrad_matched:,} ({_pct(s.sections_imrad_matched, s.total_sections)}) |")
    L(f"| Papers where ALL titles are missing/noise | {s.papers_with_no_section_titles:,} ({_pct(s.papers_with_no_section_titles, s.has_body)}) |")
    L("")

    if s.section_word_counts.values:
        p50_w = s.section_word_counts.percentile(50)
        p50_tok = int(p50_w * _WORDS_TO_TOKENS)
        fit = (
            f"fits in one chunk (≤{_CHUNK_NO_SPLIT} tok)"
            if p50_tok <= _CHUNK_NO_SPLIT
            else f"needs sliding window (>{_CHUNK_NO_SPLIT} tok)"
        )
        L("**Per-section word count** (estimated from a sample):")
        L("")
        L("| Statistic | Words |")
        L("|-----------|------:|")
        L(f"| Mean | **{s.section_word_counts.mean():,.0f}** |")
        L(f"| p50 (median) | **{p50_w:,.0f}** (~{p50_tok} tokens → {fit}) |")
        L(f"| p90 | {s.section_word_counts.percentile(90):,.0f} |")
        L("")

    L("### Top 30 Normalised Section Titles")
    L("")
    L("> Titles are lowercased and leading numbers stripped.")
    L("")
    L("| Rank | Title | Count | IMRAD |")
    L("|-----:|-------|------:|:-----:|")
    top_count = s.section_title_counter.most_common(1)[0][1] if s.section_title_counter else 1
    for i, (title, cnt) in enumerate(s.section_title_counter.most_common(30), 1):
        bar_len = int(12 * cnt / top_count)
        bar = "█" * bar_len + "░" * (12 - bar_len)
        imrad_mark = "✓" if _imrad(title) else ""
        L(f"| {i} | {title} `{bar}` | {cnt:,} | {imrad_mark} |")
    L("")

    L("---")
    L("##  5 · In-Text Citation Markers")
    L("")
    L("> `{{cite:sha_hex}}` markers in section text point to a key in `bib_entries`.")
    L("> We count **unique** ref_ids per paper.")
    L("")
    L("| Metric | Value |")
    L("|--------|------:|")
    L(f"| Total marker occurrences | **{s.total_cite_markers:,}** |")
    if N:
        L(f"| Avg occurrences / paper | **{s.total_cite_markers / N:.1f}** |")
    L(f"| Papers with NO citations | {s.papers_with_no_citations:,} ({_pct(s.papers_with_no_citations, N)}) |")
    if s.cite_counts_per_paper.values:
        L(
            f"| Unique citations per paper (p50 / p90) | "
            f"**{s.cite_counts_per_paper.percentile(50):.0f}** / {s.cite_counts_per_paper.percentile(90):.0f} |"
        )
    L(f"| Figure markers | {s.total_figure_markers:,} |")
    L(f"| Table markers | {s.total_table_markers:,} |")
    L("")

    if s.cite_marker_counts_per_paper.values and len(s.cite_marker_counts_per_paper.values) >= 10:
        top10, top25 = _cite_concentration(s.cite_marker_counts_per_paper.values)
        L("**Citation concentration**")
        L("")
        L("| Group | Share of total citations |")
        L("|-------|------------------------:|")
        L(f"| Top 10% of papers | **{top10:.1f}%** |")
        L(f"| Top 25% of papers | **{top25:.1f}%** |")
        L("")

    L("---")
    L("##  6 · Bibliography Entry Quality")
    L("")
    L("> **SHA-only** = no external ID exists — the entry is only reachable via its local SHA key.")
    L("")
    L("| Metric | Count | % of entries |")
    L("|--------|------:|-------------:|")
    L(f"| Total bib entries | **{B:,}** | — |")
    if N:
        L(f"| Avg per paper | **{B / N:.1f}** | — |")
    L(f"| Has DOI | {s.bib_has_doi:,} | **{_pct(s.bib_has_doi, B)}** |")
    L(f"| Has arXiv ID | {s.bib_has_arxiv:,} | **{_pct(s.bib_has_arxiv, B)}** |")
    L(f"| Has OpenAlex ID | {s.bib_has_openalex:,} | **{_pct(s.bib_has_openalex, B)}** |")
    L(f"| Has DOI, arXiv, or OpenAlex | {s.bib_has_any_id:,} | **{_pct(s.bib_has_any_id, B)}** |")
    L(f"| SHA-only (no external ID) | **{s.bib_sha_only:,}** | **{_pct(s.bib_sha_only, B)}** |")
    L(f"| Has title string | {s.bib_has_title:,} | {_pct(s.bib_has_title, B)} |")
    L(f"| Has year | {s.bib_has_year:,} | {_pct(s.bib_has_year, B)} |")
    L("")

    if s.bib_field_counter:
        L("**Field presence across all bib entries**")
        L("")
        L("| Field key | % present |")
        L("|-----------|----------:|")
        for k, cnt in s.bib_field_counter.most_common(20):
            L(f"| `{k}` | {_pct(cnt, B)} |")
        L("")

    L("---")
    L("##  7 · Cite-Marker → Bib Resolution")
    L("")
    L("> For each unique `{{cite:sha}}` we look up `bib_entries[sha]` and check for a DOI/arXiv ID.")
    L("")
    resolved = s.cite_ref_resolved_doi + s.cite_ref_resolved_arxiv + s.cite_ref_resolved_openalex
    L("| Path | Count | % |")
    L("|------|------:|--:|")
    L(f"| Unique ref_ids total | **{R:,}** | — |")
    L(f"| → resolved via DOI | {s.cite_ref_resolved_doi:,} | {_pct(s.cite_ref_resolved_doi, R)} |")
    L(f"| → resolved via arXiv ID | {s.cite_ref_resolved_arxiv:,} | {_pct(s.cite_ref_resolved_arxiv, R)} |")
    L(f"| → resolved via OpenAlex ID | {s.cite_ref_resolved_openalex:,} | {_pct(s.cite_ref_resolved_openalex, R)} |")
    L(f"| → SHA-only (API required) | **{s.cite_ref_sha_only:,}** | **{_pct(s.cite_ref_sha_only, R)}** |")
    L("")
    L("| Summary | % |")
    L("|---------|--:|")
    L(f"| Resolvable without API call | **{_pct(resolved, R)}** |")
    L(f"| Require Crossref / OpenAlex | **{_pct(s.cite_ref_sha_only, R)}** |")
    L("")

    L("---")
    L("##  8 · Discipline Breakdown")
    L("")
    L("| arXiv prefix | Count |")
    L("|-------------|------:|")
    for cat, cnt in s.category_counter.most_common(15):
        L(f"| `{cat}` | {cnt:,} |")
    L("")

    L("---")
    L("##  9 · Chunking Yield Estimate")
    L("")
    stride = _CHUNK_WINDOW - _CHUNK_OVERLAP
    L("> Estimates how many Qdrant vectors each paper will produce.")
    L("")
    L("```")
    L(f"tokens  = words × {_WORDS_TO_TOKENS}")
    L(f"chunks  = 1 if tokens <= {_CHUNK_NO_SPLIT}")
    L(f"        = ceil((tokens - {_CHUNK_WINDOW}) / {stride}) + 1 otherwise")
    L("paper   = sum(section_chunks) + 1")
    L("```")
    L("")

    if s.est_chunk_counts.values:
        L("| Statistic | Chunks / paper |")
        L("|-----------|---------------:|")
        L(f"| Mean | **{s.est_chunk_counts.mean():.1f}** |")
        L(f"| p10 | {s.est_chunk_counts.percentile(10):.0f} |")
        L(f"| p25 | {s.est_chunk_counts.percentile(25):.0f} |")
        L(f"| p50 (median) | **{s.est_chunk_counts.percentile(50):.0f}** |")
        L(f"| p75 | {s.est_chunk_counts.percentile(75):.0f} |")
        L(f"| p90 | {s.est_chunk_counts.percentile(90):.0f} |")
        L("")

    L("---")
    L(f"##  10 · Scale Projections → {_TARGET_PAPERS:,} papers")
    L("")
    L("> Linear extrapolation from the analysed sample.")
    L("")
    mean_chunks = s.est_chunk_counts.mean() if s.est_chunk_counts.values else 0.0
    p50_chunks = s.est_chunk_counts.percentile(50) if s.est_chunk_counts.values else 0.0
    est_vecs_mean = int(mean_chunks * _TARGET_PAPERS)
    est_vecs_p50 = int(p50_chunks * _TARGET_PAPERS)

    L("### Qdrant Vectors")
    L("")
    L("| Basis | Vectors |")
    L("|-------|--------:|")
    L(f"| Mean chunks/paper | **{est_vecs_mean:,}** ({est_vecs_mean / 1e6:.1f}M) |")
    L(f"| p50 chunks/paper | {est_vecs_p50:,} ({est_vecs_p50 / 1e6:.1f}M) |")
    L("")

    if mean_chunks > 0:
        sf32 = _storage_gb(est_vecs_mean, _BYTES_FLOAT32)
        si8 = _storage_gb(est_vecs_mean, _BYTES_INT8)
        pay = est_vecs_mean * 800 / 1e9
        L(f"### Storage (dim={_VECTOR_DIM}, +30% HNSW graph overhead)")
        L("")
        L("| Component | float32 | int8 (HPC config) |")
        L("|-----------|--------:|------------------:|")
        L(f"| Vectors | {sf32:.1f} GB | **{si8:.1f} GB** |")
        L(f"| Payload (~800 B/point) | {pay:.1f} GB | {pay:.1f} GB |")
        L(f"| **Total** | **{sf32 + pay:.1f} GB** | **{si8 + pay:.1f} GB** |")
        L("")

    bib_per_paper = B / max(1, N)
    sha_rate = s.bib_sha_only / max(1, B)
    cite_sha_rate = s.cite_ref_sha_only / max(1, R)
    avg_unique_cites = s.cite_counts_per_paper.mean() if s.cite_counts_per_paper.values else 0.0
    avg_cite_markers = s.cite_marker_counts_per_paper.mean() if s.cite_marker_counts_per_paper.values else 0.0

    est_bib_total = int(bib_per_paper * _TARGET_PAPERS)
    est_sha_bib = int(sha_rate * est_bib_total)
    est_api_calls = int(cite_sha_rate * avg_unique_cites * _TARGET_PAPERS)

    L("### Bibliography at Scale")
    L("")
    L("| | Count |")
    L("|--|------:|")
    L(f"| Total bib entries | {est_bib_total:,} ({est_bib_total / 1e6:.1f}M) |")
    L(f"| SHA-only needing resolution | **{est_sha_bib:,}** ({est_sha_bib / 1e6:.1f}M) |")
    L("")
    L("### API Resolution Budget")
    L("")
    L("| | |")
    L("|--|--|")
    L(f"| Total Crossref/OpenAlex calls needed | **{est_api_calls:,}** ({est_api_calls / 1e6:.1f}M) |")
    L(f"| At 50 req/s (OpenAlex polite pool) | **{est_api_calls / 50 / 3600:.1f} hours** |")
    L(f"| At 10 req/s (Crossref free tier) | {est_api_calls / 10 / 3600:.1f} hours |")
    L("")

    L("---")
    L("##  11 · Evidence Graph Projection")
    L("")
    arxiv_bib_rate = s.bib_has_arxiv / max(1, B)
    doi_bib_rate = s.bib_has_doi / max(1, B)
    L("| Metric | Count | % |")
    L("|--------|------:|--:|")
    L(f"| Total citation marker occurrences | {int(avg_cite_markers * _TARGET_PAPERS):,} | — |")
    L(f"| Unique cited ref_ids | {int(avg_unique_cites * _TARGET_PAPERS):,} | — |")
    L(f"| Unique cited ref_ids resolvable without API | {int((1 - cite_sha_rate) * avg_unique_cites * _TARGET_PAPERS):,} | {100 * (1 - cite_sha_rate):.1f}% |")
    L(f"| Unique cited ref_ids needing API resolution | **{int(cite_sha_rate * avg_unique_cites * _TARGET_PAPERS):,}** | **{100 * cite_sha_rate:.1f}%** |")
    L(f"| Bib entries with DOI | — | {100 * doi_bib_rate:.1f}% |")
    L(f"| Bib entries with arXiv ID | — | {100 * arxiv_bib_rate:.1f}% |")
    L("")

    L("---")
    L("##  12 · Pipeline Compatibility Summary")
    L("")
    issues: list[tuple[str, str]] = []

    if N and _pctf(s.abstract_empty, N) > 5:
        issues.append(("⚠️", f"{_pctf(s.abstract_empty, N):.1f}% papers missing abstract → abstract chunks absent"))
    if N and _pctf(s.body_empty, N) > 5:
        issues.append(("⚠️", f"{_pctf(s.body_empty, N):.1f}% papers have no body text → retrieval gap at scale"))
    if s.has_body and _pctf(s.papers_with_no_section_titles, s.has_body) > 20:
        issues.append(("⚠️", f"{_pctf(s.papers_with_no_section_titles, s.has_body):.1f}% of papers with body have no usable section titles → `chunk_section` empty"))
    if N and s.has_year_meta == 0:
        issues.append(("⚠️", "Year is 0% in metadata — must derive from arXiv ID YYMM prefix at ingest"))
    if B and _pctf(s.bib_sha_only, B) > 30:
        issues.append(("⚠️", f"{_pctf(s.bib_sha_only, B):.1f}% of bib entries are SHA-only"))
    if R and _pctf(s.cite_ref_sha_only, R) > 30:
        issues.append(("⚠️", f"{_pctf(s.cite_ref_sha_only, R):.1f}% of citations need API → ~{est_api_calls / 1e6:.1f}M calls at scale"))
    if B and arxiv_bib_rate == 0:
        issues.append(("⚠️", "0% of bib entries carry arXiv ID — cross-paper edges depend on DOI→OpenAlex"))
    if mean_chunks > 0:
        issues.append(("ℹ️", f"{mean_chunks:.0f} chunks/paper (mean) → ~{est_vecs_mean / 1e6:.0f}M vectors at scale"))
    if N and _pctf(s.papers_with_no_citations, N) > 20:
        issues.append(("ℹ️", f"{_pctf(s.papers_with_no_citations, N):.1f}% of papers have zero citation markers"))

    if not issues:
        L("✅ No major issues detected at this sample size.")
    else:
        L("| Level | Issue |")
        L("|:-----:|-------|")
        for level, msg in issues:
            L(f"| {level} | {msg} |")

    L("")
    L("---")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="unarXive data quality analysis")
    parser.add_argument("--n", type=int, default=_TARGET_PAPERS)
    parser.add_argument("--dataset", type=str, default="ines-besrour/unarxive_2024")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out", type=str, default=None, help="Output path — use .md extension")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help=(
            "Optional local directory for full unarXive_2024 tar parts. "
            "If set (or if UNARXIVE_LOCAL_DIR is defined), the script will "
            "download the big tar once and stream JSONL from it instead of "
            "using the limited HF streaming preview."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint JSON for long runs",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
        help="Save checkpoint every N analysed papers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --checkpoint if it exists",
    )

    args = parser.parse_args()

    _load_hf_token()

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"\nunarXive Data Quality Analysis — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("CPU-only — no embedding inference — no --gres required")
    print(f"Streaming {args.n:,} papers from {args.dataset} …\n", flush=True)

    t0 = time()
    started_at = datetime.now().isoformat()

    # Resume support
    if args.resume and args.checkpoint and os.path.exists(args.checkpoint):
        restored = _load_checkpoint(args.checkpoint)
        if restored is None:
            stats, skipped, stream_rows_seen = Stats(), 0, 0
        else:
            stats, skipped, stream_rows_seen, started_at = restored
            print(
                f"[resume] restored checkpoint: analysed={stats.total:,}, "
                f"stream_rows_seen={stream_rows_seen:,}, skipped={skipped:,}",
                flush=True,
            )
    else:
        stats, skipped, stream_rows_seen = Stats(), 0, 0

    # Determine local data directory, if any
    local_dir = args.local_dir or os.getenv("UNARXIVE_LOCAL_DIR")

    # Build stream
    ds_iter = _stream_rows(args.dataset, args.split, verbose=args.verbose, local_dir=local_dir)

    # Fast-forward if resuming
    if stream_rows_seen > 0:
        print(f"[resume] fast-forwarding stream by {stream_rows_seen:,} raw rows ...", flush=True)
        skipped_ff = 0
        while skipped_ff < stream_rows_seen:
            try:
                next(ds_iter)
                skipped_ff += 1
            except StopIteration:
                break
            except Exception as e:
                skipped_ff += 1
                if args.verbose and skipped_ff % 1000 == 0:
                    print(f"  [WARN] fast-forward skipped corrupt row: {e}", flush=True)

    last_checkpoint_at = stats.total

    while stats.total < args.n:
        try:
            row = next(ds_iter)
            stream_rows_seen += 1
        except StopIteration:
            break
        except Exception as e:
            skipped += 1
            if args.verbose:
                print(f"  [WARN] skipped corrupt streamed record at raw row {stream_rows_seen:,}: {e}", flush=True)
            continue

        raw = (row.get("jsonl") or "").strip()
        if not raw:
            skipped += 1
            continue

        try:
            paper = _first_jsonl(raw)
            if paper is None:
                skipped += 1
                continue

            analyse_paper(paper, stats)

        except Exception as e:
            skipped += 1
            if args.verbose:
                print(f"  [WARN] skipped parse/analyse failure at analysed paper {stats.total:,}: {e}", flush=True)
            continue

        if args.verbose and stats.total % 500 == 0:
            print(f"  … {stats.total:,} / {args.n:,}  ({time() - t0:.0f}s)", flush=True)

        if (
            args.checkpoint
            and args.checkpoint_every > 0
            and (stats.total - last_checkpoint_at) >= args.checkpoint_every
        ):
            _save_checkpoint(
                path=args.checkpoint,
                stats=stats,
                skipped=skipped,
                stream_rows_seen=stream_rows_seen,
                started_at=started_at,
            )
            last_checkpoint_at = stats.total
            if args.verbose:
                print(f"  [checkpoint] saved at analysed={stats.total:,}", flush=True)

    elapsed = time() - t0

    if args.checkpoint:
        _save_checkpoint(
            path=args.checkpoint,
            stats=stats,
            skipped=skipped,
            stream_rows_seen=stream_rows_seen,
            started_at=started_at,
        )

    if skipped:
        print(f"  (skipped {skipped:,} empty/unparseable/corrupt rows)")

    print(f"Done — {stats.total:,} papers in {elapsed:.1f}s\n")

    report = build_report(stats, args.n)
    print(report)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nReport saved → {args.out}")


if __name__ == "__main__":
    main()