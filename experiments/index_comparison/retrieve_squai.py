from __future__ import annotations
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Paths

_BASE = Path("/data/horse/ws/s3811141-faiss/inbe405h-unarxive")
_FAISS_INDEX   = _BASE / "faiss_index" / "faiss_index"
_SQLITE_DB     = _BASE / "faiss_index" / "index_store.db"
_BM25_DIR      = _BASE / "bm25_retriever"
_E5_CACHE      = _BASE / "cache" / "hf_models"


@dataclass
class RetrievalResult:
    rank: int
    paper_id: str
    title: str
    score: float
    dense_score: float
    bm25_score: float



class _DocStore:

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._cache: Optional[dict[int, dict]] = None

    def _load(self) -> None:
        logger.info("Loading FAISS doc store from SQLite …")
        conn = sqlite3.connect(str(self._db_path))
        cur = conn.cursor()

        # Load vector_id → doc_id mapping
        cur.execute("SELECT id, vector_id FROM document")
        doc_to_vec: dict[str, int] = {}
        for doc_id, vec_id in cur.fetchall():
            if vec_id is not None:
                try:
                    doc_to_vec[doc_id] = int(vec_id)
                except (ValueError, TypeError):
                    pass

        # Load metadata (paper_id, title) per doc_id
        cur.execute("SELECT document_id, name, value FROM meta_document WHERE name IN ('paper_id','title')")
        meta: dict[str, dict] = {}
        for doc_id, name, value in cur.fetchall():
            if doc_id not in meta:
                meta[doc_id] = {}
            # Values are JSON-encoded strings e.g. '"2307.07910"'
            try:
                meta[doc_id][name] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                meta[doc_id][name] = str(value).strip('"')

        conn.close()

        self._cache = {}
        for doc_id, vec_id in doc_to_vec.items():
            m = meta.get(doc_id, {})
            self._cache[vec_id] = {
                "paper_id": m.get("paper_id", ""),
                "title": m.get("title", ""),
            }
        logger.info("  Loaded %d doc entries", len(self._cache))

    def get(self, vec_id: int) -> dict:
        if self._cache is None:
            self._load()
        return self._cache.get(vec_id, {"paper_id": "", "title": ""})


class _BM25Index:

    def __init__(self, bm25_dir: Path):
        self._dir = bm25_dir
        self._loaded = False

    def _load(self) -> None:
        import scipy.sparse as sp

        logger.info("Loading BM25 index …")
        data    = np.load(str(self._dir / "data.csc.index.npy"))
        indices = np.load(str(self._dir / "indices.csc.index.npy"))
        indptr  = np.load(str(self._dir / "indptr.csc.index.npy"))

        params = json.loads((self._dir / "params.index.json").read_text())
        self._num_docs: int = params["num_docs"]

        vocab = json.loads((self._dir / "vocab.index.json").read_text())
        self._vocab: dict[str, int] = vocab
        n_terms = len(vocab)

        # The files are named .csc but the matrix is stored as CSC with shape
        # (num_docs, n_terms-1): indptr has n_terms entries (= n_cols+1 where
        # n_cols = n_terms-1), indices are row (doc) indices (max = num_docs-1).
        # One term in the vocab has no column (the last term is absent).
        self._matrix = sp.csc_matrix(
            (data, indices, indptr),
            shape=(self._num_docs, n_terms - 1),
            dtype=np.float32,
        )

        # corpus.mmindex.json is a plain list of byte offsets, one per doc
        logger.info("  Loading BM25 corpus index …")
        self._offsets: list[int] = json.loads((self._dir / "corpus.mmindex.json").read_text())
        self._corpus_path = self._dir / "corpus.jsonl"

        self._loaded = True
        logger.info("  BM25 ready: %d docs × %d terms", self._num_docs, n_terms)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def score(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if not self._loaded:
            self._load()

        tokens = self._tokenize(query)
        if not tokens:
            return []

        # Sum BM25 scores across query tokens: matrix is (num_docs, n_terms),
        # so column tid gives the BM25 weight of that term across all docs.
        scores = np.zeros(self._num_docs, dtype=np.float32)
        n_cols = self._matrix.shape[1]
        for token in tokens:
            tid = self._vocab.get(token)
            if tid is None or tid >= n_cols:
                continue
            scores += self._matrix.getcol(tid).toarray().ravel()

        # Top-k by score
        if top_k >= self._num_docs:
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]
        return results[:top_k]

    def paper_id_for(self, doc_index: int) -> tuple[str, str]:
        if not self._loaded:
            self._load()
        if doc_index < 0 or doc_index >= len(self._offsets):
            return "", ""
        offset = self._offsets[doc_index]
        with open(self._corpus_path, "rb") as f:
            f.seek(offset)
            line = f.readline()
        doc = json.loads(line)
        return doc.get("paper_id", ""), doc.get("title", "")


class _E5Embedder:
    def __init__(self):
        self._model = None

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer
        model_id = "intfloat/e5-large-v2"
        logger.info("Loading E5-large-v2 …")
        self._model = SentenceTransformer(
            model_id,
            cache_folder=str(_E5_CACHE),
            device="cpu",
        )
        logger.info("  E5-large-v2 loaded")

    def embed_query(self, text: str) -> np.ndarray:
        if self._model is None:
            self._load()
        # E5 requires "query: " prefix for queries
        vec = self._model.encode(
            ["query: " + text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0].astype(np.float32)



class SQuAIRetriever:

    def __init__(self, alpha: float = 0.5, top_k: int = 100):
        self.alpha = alpha
        self.top_k = top_k
        self._faiss_index = None
        self._doc_store = _DocStore(_SQLITE_DB)
        self._bm25 = _BM25Index(_BM25_DIR)
        self._embedder = _E5Embedder()

    def _load_faiss(self) -> None:
        import faiss
        logger.info("Loading FAISS index from %s …", _FAISS_INDEX)
        self._faiss_index = faiss.read_index(str(_FAISS_INDEX))
        logger.info("  FAISS index loaded: %d vectors", self._faiss_index.ntotal)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievalResult]:
        k = top_k or self.top_k
        if self._faiss_index is None:
            self._load_faiss()

        qvec = self._embedder.embed_query(query)
        D, I = self._faiss_index.search(qvec.reshape(1, -1), k)
        dense_scores: dict[str, float] = {}
        dense_vec_ids: dict[str, int] = {}
        for score, vec_id in zip(D[0], I[0]):
            if vec_id < 0:
                continue
            info = self._doc_store.get(int(vec_id))
            pid = info["paper_id"]
            if pid and pid not in dense_scores:
                dense_scores[pid] = float(score)
                dense_vec_ids[pid] = int(vec_id)

        bm25_hits = self._bm25.score(query, top_k=k)
        bm25_raw: dict[str, float] = {}
        bm25_titles: dict[str, str] = {}
        for doc_idx, bm25_score in bm25_hits:
            pid, title = self._bm25.paper_id_for(doc_idx)
            if pid and pid not in bm25_raw:
                bm25_raw[pid] = bm25_score
                bm25_titles[pid] = title

        if bm25_raw:
            max_bm25 = max(bm25_raw.values())
            min_bm25 = min(bm25_raw.values())
            rng = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
            bm25_norm: dict[str, float] = {
                pid: (s - min_bm25) / rng for pid, s in bm25_raw.items()
            }
        else:
            bm25_norm = {}

        if dense_scores:
            max_d = max(dense_scores.values())
            min_d = min(dense_scores.values())
            rng_d = max_d - min_d if max_d > min_d else 1.0
            dense_norm: dict[str, float] = {
                pid: (s - min_d) / rng_d for pid, s in dense_scores.items()
            }
        else:
            dense_norm = {}

        all_pids = set(dense_norm) | set(bm25_norm)
        fused: dict[str, tuple[float, float, float]] = {}
        for pid in all_pids:
            d = dense_norm.get(pid, 0.0)
            b = bm25_norm.get(pid, 0.0)
            fused[pid] = (self.alpha * d + (1 - self.alpha) * b, d, b)

        # Resolve titles: prefer BM25 corpus title, fall back to SQLite
        def _title(pid: str) -> str:
            if pid in bm25_titles:
                return bm25_titles[pid]
            vec_id = dense_vec_ids.get(pid)
            if vec_id is not None:
                return self._doc_store.get(vec_id).get("title", "")
            return ""

        ranked = sorted(fused.items(), key=lambda x: -x[1][0])[:k]
        return [
            RetrievalResult(
                rank=rank + 1,
                paper_id=pid,
                title=_title(pid),
                score=scores[0],
                dense_score=scores[1],
                bm25_score=scores[2],
            )
            for rank, (pid, scores) in enumerate(ranked)
        ]
