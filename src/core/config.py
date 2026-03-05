
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # src/core/config.py → src/core → src → project root

# 1.  PATHS

@dataclass(frozen=True)
class _Paths:
    root:           Path = PROJECT_ROOT
    data:           Path = PROJECT_ROOT / "_data"
    batches:        Path = PROJECT_ROOT / "_data" / "unarxive_batches"
    schemas:        Path = PROJECT_ROOT / "schemas"
    src:            Path = PROJECT_ROOT / "src"
    benchmark_out:  Path = PROJECT_ROOT / "benchmark" / "results"
    rocksdb:        Path = PROJECT_ROOT / "_data" / "rocksdb_citations"
    qdrant_storage: Path = PROJECT_ROOT / "_data" / "qdrant_storage"     # local on-disk Qdrant
    model_cache:    Path = PROJECT_ROOT / "_data" / ".model_cache"                    # per-model HF weight cache
    hf_home:        Optional[Path] = field(
        default_factory=lambda: Path(os.environ["HF_HOME"]) if "HF_HOME" in os.environ else None
    )

PATHS = _Paths()

# Ensure writable directories exist at import time (safe for all environments)
for _p in (PATHS.benchmark_out, PATHS.rocksdb, PATHS.qdrant_storage, PATHS.model_cache):
    _p.mkdir(parents=True, exist_ok=True)


# 2.  CHUNKING

@dataclass(frozen=True)
class _Chunking:
    # Abstract 
    # Keep as one chunk; only split if it exceeds `abstract_max_tokens`.
    abstract_max_tokens:    int = 300           # threshold before splitting
    abstract_overlap:       int = 30            # overlap when split is forced

    # Sections / Subsections 
    section_max_tokens:     int = 400           # 1 chunk when ≤ this
    section_window_size:    int = 350           # sliding-window chunk size
    section_overlap_tokens: int = 50            # overlap between windows
    split_at_sentence:      bool = True         # always split on sentence boundaries

    # Tokeniser (used for token-length estimation)
    # "auto"  → chunker calls AutoTokenizer.from_pretrained(active_model.hf_model_id)
    #           giving exact token counts for whichever embedding model is active.
    tokeniser_model_id:     str  = "auto"

    # UID 
    uid_hash_algo:          str  = "sha1"       # chunk_uid = sha1(paper_id + text)

CHUNKING = _Chunking()


# 3.  EMBEDDING MODELS

@dataclass(frozen=True)
class EmbeddingModelConfig:

    key:              str                       # short identifier used in filenames / logs
    hf_model_id:      str                       # HuggingFace model ID
    dim:              int                       # output vector dimension
    
    e5_prefix_passage:   str  = ""              # prepended to document text at encode time
    e5_prefix_query:     str  = ""              # prepended to query text at encode time
    
    qwen_task_instruction: Optional[str] = None # for Qwen3-Embedding, set the task instruction (overrides prefix_query)
    
    bge_produces_sparse:  bool = False          # BGE-M3 true hybrid
    
    normalize:        bool = True               # L2-norm before upsert (cosine collections)
    max_seq_length:   int  = 600                # hard cap passed to tokeniser
    batch_size:       int  = 64                 # inference batch size (tune to VRAM)
    device:           str  = "cuda"             # "cuda" | "cpu" | "mps"
    dtype:            str  = "float32"          # "float32" | "float16" | "bfloat16"

    @property
    def local_cache_dir(self) -> Path:
        return PATHS.model_cache / self.key


# Model registry 

EMBEDDING_MODELS: dict[str, EmbeddingModelConfig] = {

    # Option 1 candidates  (dense-only; pair with BM25 for hybrid)

    "e5-base-v2": EmbeddingModelConfig(
        key             = "e5-base-v2",
        hf_model_id     = "intfloat/e5-base-v2",
        dim             = 768,
        e5_prefix_passage  = "passage: ",
        e5_prefix_query    = "query: ",
        normalize       = True,
        max_seq_length  = 512,
        batch_size      = 64,
    ),

    "qwen3-0.6b": EmbeddingModelConfig(
        key             = "qwen3-0.6b",
        hf_model_id     = "Qwen/Qwen3-Embedding-0.6B",
        dim             = 1024,
        e5_prefix_passage  = "",
        e5_prefix_query    = "",
        qwen_task_instruction = (
            "Given a scientific query, retrieve the most relevant document passages"
        ),
        normalize       = True,
        max_seq_length  = 512,
        batch_size      = 32,   # larger model – halve the batch
    ),

    "jina-v3-nano": EmbeddingModelConfig(
        key             = "jina-v3-nano",
        hf_model_id     = "jinaai/jina-embeddings-v3",   # nano variant via task param
        dim             = 512,
        e5_prefix_passage  = "",
        e5_prefix_query    = "",
        normalize       = True,
        max_seq_length  = 512,
        batch_size      = 64,
    ),


    # Option 2 candidate  (dense + sparse in a single forward pass)

    "bge-m3": EmbeddingModelConfig(
        key             = "bge-m3",
        hf_model_id     = "BAAI/bge-m3",
        dim             = 1024,
        e5_prefix_passage  = "",
        e5_prefix_query    = "",
        bge_produces_sparse = True,     # triggers sparse vector column in Qdrant
        normalize       = True,
        max_seq_length  = 8192,         # BGE-M3 supports long context
        batch_size      = 16,           # sparse compute is heavier
    ),
}

# Convenience: default model used when none is specified
DEFAULT_EMBEDDING_MODEL: str = "e5-base-v2"


# 4.  QDRANT

@dataclass(frozen=True)
class _QdrantConnection:
    host:       str  = "localhost"
    port:       int  = 6333
    grpc_port:  int  = 6334
    prefer_grpc: bool = True            # gRPC is faster for bulk upserts
    # Set api_key / https_url for cloud Qdrant; leave None for local Docker
    api_key:    Optional[str] = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY")
    )
    url:        Optional[str] = field(
        default_factory=lambda: os.getenv("QDRANT_URL")    # e.g. https://xyz.cloud.qdrant.io
    )


@dataclass(frozen=True)
class _HNSWConfig:
    m:              int = 16    # number of bi-directional links per node
    ef_construct:   int = 64    # size of dynamic candidate list during index build
    # query-time ef is set per-search call; this is the default
    ef:             int = 64    # default search-time ef (recall/latency knob)
    full_scan_threshold: int = 10_000   # switch to exact search below this many vectors


@dataclass(frozen=True)
class _QdrantProfile:
    profile:            Literal["local", "hpc"]

    collection_name:    str  = "unarxive_chunks"

    # Distance 
    distance:           str  = "Cosine"         # must match normalize=True

    # HNSW 
    hnsw:               _HNSWConfig = field(default_factory=_HNSWConfig)

    # Storage 
    vectors_on_disk:    bool = False            # True → mmap on HPC
    payload_on_disk:    bool = False

    # Scalar quantisation
    # Disabled on laptop (prototype accuracy); enabled on HPC (~4× RAM reduction)
    quantize:           bool = False
    quantize_always_ram: bool = True            # keep quantised vectors in RAM even if on_disk

    # Sparse vector column (Option 2 / BGE-M3 only)
    enable_sparse:      bool = False

    # Full-text / BM25 index 
    fulltext_field:     str  = "text"           # payload field indexed for BM25

    # Payload indexes for filtered search 
    payload_indexes: tuple[str, ...] = (
        "paper_id_arxiv",
        "chunk_type",
        "section_title",
        "paper.year",
        "paper.categories",
    )

    # Upsert throughput
    upsert_batch_size:  int  = 256              # points per upsert call


# Local  — prototype, everything in RAM, no quantisation
QDRANT_LAPTOP: _QdrantProfile = _QdrantProfile(
    profile         = "local",
    hnsw            = _HNSWConfig(m=16, ef_construct=64, ef=64),
    vectors_on_disk = False,
    payload_on_disk = False,
    quantize        = False,
    enable_sparse   = False,
    upsert_batch_size = 128,
)

# HPC — full 2.8 M paper scale, mmap vectors, scalar quantisation
QDRANT_HPC: _QdrantProfile = _QdrantProfile(
    profile         = "hpc",
    hnsw            = _HNSWConfig(m=32, ef_construct=128, ef=64),
    vectors_on_disk = True,
    payload_on_disk = True,
    quantize        = True,
    quantize_always_ram = True,
    enable_sparse   = False,            # flip to True when using BGE-M3 (Option 2)
    upsert_batch_size = 512,
)

QDRANT_CONNECTION: _QdrantConnection = _QdrantConnection()

# Active profile
_ENV_PROFILE = os.getenv("INDEXING_PROFILE", "local").lower()
QDRANT_ACTIVE: _QdrantProfile = QDRANT_HPC if _ENV_PROFILE == "hpc" else QDRANT_LAPTOP


# 5.  ROCKSDB  (citation store)

@dataclass(frozen=True)
class _RocksDB:
    path:               Path = PATHS.rocksdb
    # Key   = DOI string  (bytes)
    # Value = msgpack-encoded dict  {doi, title, authors, year}
    encoding:           str  = "msgpack"        # "msgpack" | "json" :  msgpack is ~30 % smaller + ~2× faster, use on HPC
    create_if_missing:  bool = True
    max_open_files:     int  = 512

ROCKSDB = _RocksDB()


# 6.  BENCHMARK

@dataclass(frozen=True)
class _Benchmark:
    # Dataset princeton-nlp/LitSearch  — scientific literature retrieval QA benchmark
    # split "test"  (queries + candidate pool)
    hf_dataset_id:      str  = "princeton-nlp/LitSearch"
    hf_split:           str  = "test"

    # Retrieval depth 
    # k values at which recall / precision / ndcg are evaluated
    k_values:           tuple[int, ...] = (1, 5, 10, 20, 100)

    # Metrics
    # MRR@k, NDCG@k, Recall@k, Precision@k, MAP@k
    metrics: tuple[str, ...] = (
        "mrr",          # Mean Reciprocal Rank
        "ndcg",         # Normalised Discounted Cumulative Gain
        "recall",       # Recall@k
        "precision",    # Precision@k
        "map",          # Mean Average Precision
    )

    #  Hybrid retrieval knobs
    # Option 1 – dense + BM25, fused with RRF
    rrf_k:              int   = 60              # RRF constant (standard: 60)
    bm25_top_k:         int   = 100             # candidates from BM25 leg
    dense_top_k:        int   = 100             # candidates from dense leg

    # Option 2 – dense + sparse (BGE-M3), fused with RRF
    sparse_top_k:       int   = 100

    # Output 
    results_dir:        Path  = PATHS.benchmark_out
    # Each run saves a JSON: results_dir / {run_tag}.json
    # run_tag is built by the benchmark scripts as  "{model_key}_{retrieval_mode}"


BENCHMARK = _Benchmark()


# 7.  QUICK SANITY CHECK  (python -m src.config)

if __name__ == "__main__":
    import dataclasses
    import json

    def _default(obj):

        if isinstance(obj, Path):
            return str(obj)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, tuple):
            return list(obj)
        raise TypeError(repr(obj))

    import dataclasses as _dc

    sections = {
        "PATHS":             _dc.asdict(PATHS),
        "CHUNKING":          _dc.asdict(CHUNKING),
        "QDRANT_ACTIVE":     _dc.asdict(QDRANT_ACTIVE),
        "QDRANT_CONNECTION": _dc.asdict(QDRANT_CONNECTION),
        "ROCKSDB":           _dc.asdict(ROCKSDB),
        "BENCHMARK":         _dc.asdict(BENCHMARK),
        "EMBEDDING_MODELS":  {k: _dc.asdict(v) for k, v in EMBEDDING_MODELS.items()},
    }

    print(json.dumps(sections, indent=2, default=_default))
