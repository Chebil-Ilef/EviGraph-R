from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  
_ENV_PROFILE = os.getenv("INDEXING_PROFILE", "local").lower()


def _resolve_compute_device() -> str:
    override = (os.getenv("INDEXING_DEVICE") or "").strip().lower()
    if override:
        return override

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def _resolve_hf_home() -> Optional[Path]:
    raw = os.getenv("HF_HOME")
    fallback = PROJECT_ROOT / "_data" / ".hf_home"

    if not raw:
        return None

    candidate = Path(raw).expanduser()

    # local runs should never try to write to HPC scratch defaults.
    if _ENV_PROFILE != "hpc" and candidate.parts and candidate.parts[1:2] == ("scratch",):
        os.environ["HF_HOME"] = str(fallback)
        return fallback

    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError:
        os.environ["HF_HOME"] = str(fallback)
        return fallback

    return candidate

# 1.  PATHS

@dataclass(frozen=True)
class _Paths:
    root:           Path = PROJECT_ROOT
    data:           Path = Path(os.getenv("EVI_DATA_DIR", str(PROJECT_ROOT / "_data")))
    hf_export_cache: Path = Path(os.getenv("EVI_HF_EXPORT_CACHE_DIR", str(PROJECT_ROOT / "_data" / "dataset_index_cache")))
    batches:        Path = Path(os.getenv("EVI_BATCHES_DIR", str(PROJECT_ROOT / "_data" / "unarxive_batches")))
    schemas:        Path = PROJECT_ROOT / "schemas"
    src:            Path = PROJECT_ROOT / "src"
    benchmark_out:  Path = PROJECT_ROOT / "_data" / "benchmark" / "results"
    qdrant_storage: Path = Path(os.getenv("EVI_QDRANT_STORAGE_DIR", str(PROJECT_ROOT / "_data" / "qdrant_storage")))
    qdrant_snapshots: Path = Path(os.getenv("EVI_QDRANT_SNAPSHOTS_DIR", str(PROJECT_ROOT / "_data" / "qdrant_snapshots")))
    model_cache:    Path = PROJECT_ROOT / "_data" / ".model_cache"
    chunks:         Path = PROJECT_ROOT / "_data" / "chunks"
    shards:         Path = Path(os.getenv("EVI_SHARDS_DIR", str(PROJECT_ROOT / "_data" / "shards")))
    manifests:      Path = Path(os.getenv("EVI_MANIFESTS_DIR", str(PROJECT_ROOT / "_data" / "manifests")))
    progress:       Path = Path(os.getenv("EVI_PROGRESS_DIR", str(PROJECT_ROOT / "_data" / "progress")))
    hf_dataset_cache: Path = Path(os.getenv("EVI_HF_DATASET_CACHE_DIR", str(PROJECT_ROOT / "_data" / "unarxive_2024_full")))
    hf_home:        Optional[Path] = field(default_factory=_resolve_hf_home)
    dataset_batch_size: int = int(os.getenv("EVI_DATASET_BATCH_SIZE", "1000"))
    shard_batch_size: int = int(os.getenv("EVI_SHARD_BATCH_SIZE", "128"))

    @property
    def shard_manifest(self) -> Path:
        return self.manifests / "shard_status.jsonl"

    @property
    def ingested_shards(self) -> Path:
        return self.progress / "ingested_shards.jsonl"

    @property
    def run_metadata(self) -> Path:
        return self.manifests / "run_metadata.json"

    @property
    def snapshot_metadata(self) -> Path:
        return self.progress / "snapshot.json"

    @property
    def hf_export_metadata(self) -> Path:
        return self.progress / "hf_index_export.json"

PATHS = _Paths()

# Ensure writable directories exist at import time (safe for all environments)
for _p in (
    PATHS.benchmark_out,
    PATHS.qdrant_storage,
    PATHS.qdrant_snapshots,
    PATHS.model_cache,
    PATHS.chunks,
    PATHS.hf_export_cache,
    PATHS.shards,
    PATHS.manifests,
    PATHS.progress,
    PATHS.hf_dataset_cache,
):
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


# 2.  CHUNKING

@dataclass(frozen=True)
class _Chunking:
    # Abstract 
    # Keep as one chunk; only split if it exceeds `abstract_max_tokens`.
    abstract_max_tokens:    int = 500           # threshold before splitting
    abstract_overlap:       int = 50            # overlap when split is forced

    # Sections / Subsections 
    section_max_tokens:     int = 900           # 1 chunk when ≤ this
    section_window_size:    int = 900           # sliding-window chunk size
    section_overlap_tokens: int = 30            # overlap between windows
    split_at_sentence:      bool = True         # always split on sentence boundaries

    # Tokeniser (used for token-length estimation)
    # "auto"  → chunker calls AutoTokenizer.from_pretrained(active_model.hf_model_id)
    #           giving exact token counts for whichever embedding model is active.
    tokeniser_model_id:     str  = "auto"

    # UID 
    uid_hash_algo:          str  = "sha1"       # chunk_uid = sha1(paper_id + text)

CHUNKING = _Chunking()


# EMBEDDING MODELS

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
    device:           str = field(default_factory=_resolve_compute_device)  # "cuda" | "cpu" | "mps"
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

    "e5-large-v2": EmbeddingModelConfig(
        key             = "e5-large-v2",
        hf_model_id     = "intfloat/e5-large-v2",
        dim             = 1024,
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
        batch_size      = 32,   
    ),

    "embeddinggemma-300m": EmbeddingModelConfig(
        key             = "embeddinggemma-300m",
        hf_model_id     = "google/embeddinggemma-300m",
        dim             = 768,
        e5_prefix_passage  = "",
        e5_prefix_query    = "",
        normalize       = True,
        max_seq_length  = 512,
        batch_size      = 64,
    ),

    
    "jina-v3-nano": EmbeddingModelConfig(
        key="jina-v3-nano",
        hf_model_id="jinaai/jina-embeddings-v5-text-nano-retrieval",
        dim=768,   
        e5_prefix_passage="",
        e5_prefix_query="",
        normalize=True,
        max_seq_length=512,
        batch_size=64,
    ),

    "bge-m3-bm25": EmbeddingModelConfig(
        key             = "bge-m3-bm25",
        hf_model_id     = "BAAI/bge-m3",
        dim             = 1024,
        e5_prefix_passage  = "",
        e5_prefix_query    = "",
        bge_produces_sparse = False,    # use BGE dense only, pair with BM25 at retrieval time
        normalize       = True,
        max_seq_length  = 8192,
        batch_size      = 64,
    ),

    # dense + sparse in a single forward pass

    "bge-m3": EmbeddingModelConfig(
        key             = "bge-m3",
        hf_model_id     = "BAAI/bge-m3",
        dim             = 1024,
        e5_prefix_passage  = "",
        e5_prefix_query    = "",
        bge_produces_sparse = True,     # triggers sparse vector column in Qdrant
        normalize       = True,
        max_seq_length  = 8192,         # BGE-M3 supports long context
        batch_size      = 512,          # fp16 doubles VRAM headroom vs 256
        dtype           = "float16",    # ~2× faster on CUDA vs float32
    ),
}

# Convenience: default model used when none is specified
DEFAULT_EMBEDDING_MODEL: str = "bge-m3"

# RERANKER

@dataclass(frozen=True)
class RerankConfig:
    model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = field(default_factory=_resolve_compute_device)
    batch_size: int = 32
    top_n: int = 30  # candidates to fetch before reranking
    enabled: bool = field(default_factory=lambda: os.getenv("RERANKER_ENABLED", "true").lower() != "false")
    min_score_threshold: float = 0.15  # minimum score to return (filter negatives/low-confidence)

RERANKER = RerankConfig()


@dataclass(frozen=True)
class GraphConfig:
    
    nli_model_id: str = field(
        default_factory=lambda: os.getenv(
            "NLI_MODEL_ID",
            "sileod/deberta-v3-small-tasksource-nli",
        )
    )
    scicite_model_id: str = field(
        default_factory=lambda: os.getenv(
            "SCICITE_MODEL_ID",
            "lostelf/scibert_scivocab_uncased_scicite_finetuned",
        )
    )
    nli_threshold: float = field(
        default_factory=lambda: float(os.getenv("NLI_THRESHOLD", "0.65"))
    )
    nli_contradiction_threshold: float = field(
        default_factory=lambda: float(os.getenv("NLI_CONTRADICTION_THRESHOLD", "0.70"))
    )
    
    # Hop retrieval
    hop_max_chunks_per_claim: int = field(
        default_factory=lambda: int(os.getenv("HOP_MAX_CHUNKS_PER_CLAIM", "2"))
    )
    hop_max_per_build: int = field(
        default_factory=lambda: int(os.getenv("HOP_MAX_PER_BUILD", "50"))
    )

    # Judge evidence trail
    max_evidence_trail_depth: int = field(
        default_factory=lambda: int(os.getenv("MAX_EVIDENCE_TRAIL_DEPTH", "5"))
    )

    # Claim extraction cap per chunk (enforced via LLM prompt)
    max_claims_per_chunk: int = field(
        default_factory=lambda: int(os.getenv("MAX_CLAIMS_PER_CHUNK", "3"))
    )

    # Number of chunks to batch into a single LLM claim extraction call
    claim_extraction_batch_size: int = field(
        default_factory=lambda: int(os.getenv("CLAIM_EXTRACTION_BATCH_SIZE", "14"))
    )


GRAPH_CONFIG = GraphConfig()

# LLM

@dataclass(frozen=True)
class LLMConfig:

    api_base: Optional[str] = field(
        default_factory=lambda: (os.getenv("LLM_API_BASE") or "").strip() or None
    )
    api_key: Optional[str] = field(
        default_factory=lambda: (os.getenv("LLM_API_KEY") or "").strip() or None
    )

    decomposer_model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_DECOMPOSER_MODEL",
            os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        )
    )

    evidence_graph_builder_model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_EVIDENCE_GRAPH_BUILDER_MODEL",
            os.getenv("LLM_MODEL","meta-llama/Llama-3.3-70B-Instruct"),
        )
    )

    judge_model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_JUDGE_MODEL",
            os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        )
    )

    answer_generator_model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_ANSWER_GENERATOR_MODEL",
            os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        )
    )

 
    # Decomposer Agent 1
    decomposer_temperature: float = 0.0
    decomposer_timeout_seconds: float = 300.0
    decomposer_max_retries: int = 3

    # Evidence Graph Builder Agent 2
    evidence_graph_builder_temperature: float = 0.0
    evidence_graph_builder_timeout_seconds: float = 300.0
    evidence_graph_builder_max_retries: int = 3

    # Judge Agent 3
    judge_temperature: float = 0.0
    judge_timeout_seconds: float = 300.0
    judge_max_retries: int = 3

    # Answer Generator Agent 4
    answer_generator_temperature: float = 0.0
    answer_generator_timeout_seconds: float = 300.0
    answer_generator_max_retries: int = 3
    answer_max_claims_total: int = field(
        default_factory=lambda: int(os.getenv("ANSWER_MAX_CLAIMS_TOTAL", "25"))
    )
    answer_min_claims_per_subquery: int = field(
        default_factory=lambda: int(os.getenv("ANSWER_MIN_CLAIMS_PER_SUBQUERY", "2"))
    )

    # Generic / default
    timeout_seconds: float = 300.0
    max_retries: int = 3


@dataclass(frozen=True)
class AgentModelConfig:
    model: str
    temperature: float = 0.0
    timeout_seconds: float = 300.0
    max_retries: int = 3
    max_tokens: int | None = None
    # Answer-generator only
    answer_max_claims_total: int = 12
    answer_min_claims_per_subquery: int = 2


LLM = LLMConfig()

AGENT_MODELS: dict[str, AgentModelConfig] = {

    "decomposer": AgentModelConfig(
        model=LLM.decomposer_model,
        temperature=LLM.decomposer_temperature,
        timeout_seconds=LLM.decomposer_timeout_seconds,
        max_retries=LLM.decomposer_max_retries,
        max_tokens=int(os.getenv("DECOMPOSER_MAX_TOKENS", "4096")),
    ),

    "evidence_graph_builder": AgentModelConfig(
        model=LLM.evidence_graph_builder_model,
        temperature=LLM.evidence_graph_builder_temperature,
        timeout_seconds=LLM.evidence_graph_builder_timeout_seconds,
        max_retries=LLM.evidence_graph_builder_max_retries,
        max_tokens=int(os.getenv("EVIDENCE_GRAPH_BUILDER_MAX_TOKENS", "4096")),
    ),

    "judge": AgentModelConfig(
        model=LLM.judge_model,
        temperature=LLM.judge_temperature,
        timeout_seconds=LLM.judge_timeout_seconds,
        max_retries=LLM.judge_max_retries,
        max_tokens=int(os.getenv("JUDGE_MAX_TOKENS", "4096")),
    ),

    "answer_generator": AgentModelConfig(
        model=LLM.answer_generator_model,
        temperature=LLM.answer_generator_temperature,
        timeout_seconds=LLM.answer_generator_timeout_seconds,
        max_retries=LLM.answer_generator_max_retries,
        max_tokens=int(os.getenv("ANSWER_GENERATOR_MAX_TOKENS", "4096")),
        answer_max_claims_total=LLM.answer_max_claims_total,
        answer_min_claims_per_subquery=LLM.answer_min_claims_per_subquery,
    ),
}


#  QDRANT

@dataclass(frozen=True)
class _QdrantConnection:
    host:       str  = "localhost"
    port:       int  = 6333
    grpc_port:  int  = 6334
    prefer_grpc: bool = False            # Use HTTP by default for better HPC compatibility
    # Set api_key / https_url for cloud Qdrant; leave None for local Docker
    api_key:    Optional[str] = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY")
    )
    url:        Optional[str] = field(
        default_factory=lambda: os.getenv("QDRANT_URL")    # e.g. https://xyz.cloud.qdrant.io
    )
    timeout:    int = field(
        default_factory=lambda: int(os.getenv("QDRANT_TIMEOUT", "60"))
    )


@dataclass(frozen=True)
class _HNSWConfig:
    m:              int = 16    # number of bi-directional links per node
    ef_construct:   int = 64    # size of dynamic candidate list during index build
    # query-time ef is set per-search call; this is the default
    ef:             int = 64    # default search-time ef (recall/latency knob)
    full_scan_threshold: int = 10_000   # switch to exact search below this many vectors


@dataclass(frozen=True)
class _OptimizerConfig:
    deleted_threshold: float = 0.2
    vacuum_min_vector_number: int = 1000
    default_segment_number: int = 0
    max_segment_size: int | None = None
    memmap_threshold: int | None = 20_000
    indexing_threshold: int = 10_000
    flush_interval_sec: int = 15
    max_optimization_threads: int | None = None
    prevent_unoptimized: bool | None = None


@dataclass(frozen=True)
class _WalConfig:
    wal_capacity_mb: int = 32
    wal_segments_ahead: int = 0
    wal_retain_closed: int = 1


@dataclass(frozen=True)
class _QdrantProfile:
    profile:            Literal["local", "hpc"]

    collection_name:    str  = "unarxive_chunks"

    # Distance 
    distance:           str  = "Cosine"         # must match normalize=True

    # HNSW 
    hnsw:               _HNSWConfig = field(default_factory=_HNSWConfig)
    optimizer:          _OptimizerConfig = field(default_factory=_OptimizerConfig)
    wal:                _WalConfig = field(default_factory=_WalConfig)

    # Storage 
    vectors_on_disk:    bool = False            # True → mmap on HPC
    payload_on_disk:    bool = False

    # Scalar quantisation
    # Disabled on laptop (prototype accuracy); enabled on HPC (~4× RAM reduction)
    quantize:             bool = False
    quantize_always_ram:  bool = True            # keep quantised vectors in RAM even if on_disk
    quantize_scalar_type: str  = "int8"          # "int8" | "uint8"

    # Vector field names (must match collection schema)
    dense_vector_name:  str  = "dense"          # named-vector key for dense (BGE-M3 mode)
    sparse_vector_name: str  = "sparse"         # named-vector key for sparse (BGE-M3 mode)

    # BM25 sparse retrieval
    bm25_model:         str  = "Qdrant/bm25"    # server-side tokeniser model

    # Full-text / BM25 index 
    fulltext_field:         str  = "embed_text"           # payload field indexed for BM25
    fulltext_tokenizer:     str  = "word"               # TokenizerType key
    fulltext_min_token_len: int  = 2
    fulltext_max_token_len: int  = 40
    fulltext_lowercase:     bool = True

    # Payload indexes for filtered search
    payload_indexes: tuple[str, ...] = (
        "paper_id_arxiv",
        "paper_doi",           # required for hop retrieval by DOI
        "title",               # paper title — filterable for targeted retrieval
        "authors",             # paper authors list — filterable for targeted retrieval
        "chunk_type",
        "chunk_index",
        "total_chunks",
        "section_title",
        "year",
        "categories",
    )

    # Index type per payload field ("keyword" | "integer"); defaults to "keyword"
    payload_index_types: dict = field(default_factory=lambda: {
        "paper_id_arxiv":    "keyword",
        "paper_doi":         "keyword",
        "title":             "keyword",
        "authors":           "keyword",
        "chunk_type":        "keyword",
        "chunk_index":       "integer",
        "total_chunks":      "integer",
        "section_title":     "keyword",
        "paper.categories":  "keyword",
        "categories":        "keyword",
        "paper.year":        "integer",
        "year":              "integer",
    })

    # Upsert throughput
    upsert_batch_size:  int  = 256              # points per upsert call


# Local — prototype, everything in RAM, no quantisation

QDRANT_LAPTOP: _QdrantProfile = _QdrantProfile(
    profile         = "local",
    hnsw            = _HNSWConfig(m=16, ef_construct=128, ef=64),
    optimizer       = _OptimizerConfig(memmap_threshold=None, flush_interval_sec=15),
    wal             = _WalConfig(wal_capacity_mb=32, wal_segments_ahead=0, wal_retain_closed=1),
    vectors_on_disk = False,
    payload_on_disk = False,
    quantize        = False,
    upsert_batch_size = 32,
)

# HPC — full 2.8 M paper scale, mmap vectors, scalar quantisation

QDRANT_HPC: _QdrantProfile = _QdrantProfile(
    profile         = "hpc",
    hnsw            = _HNSWConfig(m=32, ef_construct=128, ef=64),
    optimizer       = _OptimizerConfig(memmap_threshold=10_000, flush_interval_sec=15),
    wal             = _WalConfig(wal_capacity_mb=32, wal_segments_ahead=0, wal_retain_closed=1),
    vectors_on_disk = True,
    payload_on_disk = True,
    quantize        = True,
    quantize_always_ram = False,
    upsert_batch_size = 128,
)

QDRANT_CONNECTION: _QdrantConnection = _QdrantConnection()

# Active profile
QDRANT_ACTIVE: _QdrantProfile = QDRANT_HPC if _ENV_PROFILE == "hpc" else QDRANT_LAPTOP


def get_qdrant_profile(profile_name: str) -> _QdrantProfile:
    return QDRANT_HPC if profile_name == "hpc" else QDRANT_LAPTOP


@dataclass(frozen=True)
class _QdrantRuntime:
    local_container_name: str = field(default_factory=lambda: os.getenv("QDRANT_CONTAINER_NAME", "evigraph-qdrant"))
    local_image: str = field(default_factory=lambda: os.getenv("QDRANT_IMAGE", "qdrant/qdrant"))
    hpc_instance_name: str = field(default_factory=lambda: os.getenv("QDRANT_INSTANCE_NAME", "evigraph-qdrant"))
    hpc_sif_path: str = field(default_factory=lambda: os.getenv("QDRANT_SIF_PATH", "qdrant.sif"))


QDRANT_RUNTIME = _QdrantRuntime()


@dataclass(frozen=True)
class _HFDataset:
    dataset_id: str = field(default_factory=lambda: os.getenv("INDEXING_HF_DATASET", "ines-besrour/unarxive_2024"))
    split: str = field(default_factory=lambda: os.getenv("INDEXING_HF_SPLIT", "train"))
    batch_size: int = PATHS.dataset_batch_size


HF_DATASET = _HFDataset()


@dataclass(frozen=True)
class _HFIndexExport:
    username: str = field(default_factory=lambda: (os.getenv("INDEXING_HF_USERNAME") or "").strip())
    dense_dataset: str = field(default_factory=lambda: (os.getenv("INDEXING_HF_DENSE_DATASET") or "").strip())
    sparse_dataset: str = field(default_factory=lambda: (os.getenv("INDEXING_HF_SPARSE_DATASET") or "").strip())
    split: str = field(default_factory=lambda: (os.getenv("INDEXING_HF_EXPORT_SPLIT") or "train").strip())
    token: str = field(default_factory=lambda: (os.getenv("HF_TOKEN") or "").strip())


HF_INDEX_EXPORT = _HFIndexExport()


#  RETRIEVAL

@dataclass(frozen=True)
class _Retrieval:
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

    # Section boost: after cross-encoder reranking, chunks whose section_title
    # matches the sub-query's target sections receive a score bonus of
    # top_score * section_boost_gamma.  Set to 0.0 to disable.
    section_boost_gamma: float = 0.10

    # Output
    results_dir:        Path  = PATHS.benchmark_out
    # Each run saves a JSON: results_dir / {run_tag}.json
    # run_tag is built by the benchmark scripts as  "{model_key}_{retrieval_mode}"


RETRIEVAL = _Retrieval()
# backwards-compat alias
BENCHMARK = RETRIEVAL


#  QUICK SANITY CHECK  (python -m src.config)

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
        "QDRANT_RUNTIME":    _dc.asdict(QDRANT_RUNTIME),
        "HF_DATASET":        _dc.asdict(HF_DATASET),
        "RETRIEVAL":         _dc.asdict(RETRIEVAL),
        "EMBEDDING_MODELS":  {k: _dc.asdict(v) for k, v in EMBEDDING_MODELS.items()},
    }

    print(json.dumps(sections, indent=2, default=_default))
