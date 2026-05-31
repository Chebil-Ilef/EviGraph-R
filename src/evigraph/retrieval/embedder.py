from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
from tqdm import tqdm

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
from evigraph.config.settings import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS, EmbeddingModelConfig

logger = logging.getLogger(__name__)
load_dotenv()

class BGEOutput(NamedTuple):

    dense:  np.ndarray          # shape (N, 1024) for passages / (1024,) for query
    sparse: list[dict]          # list of {token_id (int): weight (float)} per text

def _l2_normalize(x: np.ndarray) -> np.ndarray:

    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _batch(lst: list, size: int):

    for i in range(0, len(lst), size):
        yield lst[i : i + size]


class Embedder:

    def __init__(self, cfg: EmbeddingModelConfig, model):
        self.cfg   = cfg
        self._model = model
        self._uses_bge_backend = cfg.hf_model_id.lower() == "baai/bge-m3"
        self._is_bge = cfg.bge_produces_sparse

    # Factory 

    @classmethod
    def from_model_key(cls, model_key: str = DEFAULT_EMBEDDING_MODEL) -> "Embedder":

        if model_key not in EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown model key {model_key!r}. "
                f"Available: {list(EMBEDDING_MODELS)}"
            )

        cfg = EMBEDDING_MODELS[model_key]
        cache = str(cfg.local_cache_dir)
        logger.info("Loading model %s from %s on device=%s …", model_key, cache, cfg.device)

        if cfg.hf_model_id.lower() == "baai/bge-m3":
            model = cls._load_bge(cfg, cache)
        else:
            model = cls._load_st(cfg, cache)

        logger.info("Model %s ready.", model_key)
        return cls(cfg, model)

    @staticmethod
    def _load_st(cfg: EmbeddingModelConfig, cache: str):
        from sentence_transformers import SentenceTransformer

        token = os.getenv("HF_TOKEN")
        # Keep all HF cache and token-aware downloads inside the project cache tree.
        os.environ["HF_HOME"] = cache
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache

        model = SentenceTransformer(
            cfg.hf_model_id,
            token=token,
            cache_folder=cache,
            trust_remote_code=True,
            device=cfg.device,
            # model_kwargs={"torch_dtype": torch.float32},
        )
        model.max_seq_length = cfg.max_seq_length
        return model

    @staticmethod
    def _load_bge(cfg: EmbeddingModelConfig, cache: str):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required for bge-m3. "
                "Install it with: pip install flagembedding"
            ) from exc

        # Set HF cache directory for transformers library used by FlagEmbedding
        os.environ["HF_HOME"] = cache
        
        return BGEM3FlagModel(
            cfg.hf_model_id,
            use_fp16=(cfg.dtype in ("float16", "bfloat16")),
        )


    def embed_passages(
        self, texts: list[str]
    ) -> Union[np.ndarray, BGEOutput]:

        if not texts:
            empty = np.empty((0, self.cfg.dim), dtype=np.float32)
            return BGEOutput(empty, []) if self._is_bge else empty

        prefixed = self._apply_passage_prefix(texts)

        if self._uses_bge_backend:
            out = self._encode_bge(prefixed)
            return out if self._is_bge else out.dense
        if self._is_bge:
            return self._encode_bge(prefixed)
        return self._encode_st(prefixed, is_query=False)

    def embed_query(self, text: str) -> Union[np.ndarray, BGEOutput]:

        prefixed = self._apply_query_prefix(text)

        if self._uses_bge_backend:
            out = self._encode_bge([prefixed])
            return BGEOutput(out.dense[0], out.sparse) if self._is_bge else out.dense[0]
        if self._is_bge:
            out = self._encode_bge([prefixed])
            return BGEOutput(out.dense[0], out.sparse)
        vecs = self._encode_st([prefixed], is_query=True)
        return vecs[0]


    def _apply_passage_prefix(self, texts: list[str]) -> list[str]:
        cfg = self.cfg

        if "jina-embeddings-v5" in cfg.hf_model_id.lower():
            return texts

        prefix = cfg.e5_prefix_passage
        return [prefix + t for t in texts] if prefix else texts

    def _apply_query_prefix(self, text: str) -> str:
        cfg = self.cfg

        if cfg.qwen_task_instruction:
            return f"Instruct: {cfg.qwen_task_instruction}\nQuery: {text}"

        elif "jina-embeddings-v5" in cfg.hf_model_id.lower():
            return text

        prefix = cfg.e5_prefix_query
        return prefix + text if prefix else text    
    
    def _encode_st(
        self, texts: list[str], *, is_query: bool
    ) -> np.ndarray:
        """Batch-encode with SentenceTransformer, return float32 ndarray."""
        cfg = self.cfg
        batch_size = cfg.batch_size
        all_vecs: list[np.ndarray] = []

        model_id = cfg.hf_model_id.lower()
        extra_kwargs: dict = {}

        # Jina v5: use prompt_name + truncate_dim
        if "jina-embeddings-v5" in model_id:
            extra_kwargs["prompt_name"] = "query" if is_query else "document"
            extra_kwargs["truncate_dim"] = cfg.dim

        # Older Jina models: use task=
        elif "jina" in model_id:
            extra_kwargs["task"] = "retrieval.query" if is_query else "retrieval.passage"

        num_batches = (len(texts) + batch_size - 1) // batch_size
        for sub in tqdm(
            _batch(texts, batch_size),
            total=num_batches,
            desc=f"Embedding [{cfg.key}]",
            unit="batch",
            leave=False,
        ):
            vecs = self._model.encode(
                sub,
                batch_size=batch_size,
                normalize_embeddings=False,   # we normalize ourselves
                show_progress_bar=False,
                convert_to_numpy=True,
                **extra_kwargs,
            )
            vecs = np.asarray(vecs, dtype=np.float32)

            if vecs.ndim == 1:
                vecs = vecs[None, :]

            if cfg.normalize:
                vecs = _l2_normalize(vecs)

            all_vecs.append(vecs)

        return np.vstack(all_vecs)
    
    def _encode_bge(self, texts: list[str]) -> BGEOutput:

        cfg        = self.cfg
        batch_size = cfg.batch_size

        # Sort by length to minimise padding waste within each batch, then restore order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sorted_texts = [texts[i] for i in order]
        restore = [0] * len(texts)
        for out_pos, orig_pos in enumerate(order):
            restore[orig_pos] = out_pos

        all_dense:  list[np.ndarray] = []
        all_sparse: list[dict]       = []

        for sub in tqdm(
            list(_batch(sorted_texts, batch_size)),
            desc=f"Embedding [bge-m3]",
            unit="batch",
            leave=False,
        ):
            out = self._model.encode(
                sub,
                batch_size=batch_size,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense = np.array(out["dense_vecs"], dtype=np.float32)
            if cfg.normalize:
                dense = _l2_normalize(dense)
            all_dense.append(dense)

            for lw in out["lexical_weights"]:
                all_sparse.append({int(k): float(v) for k, v in lw.items()})

        # Restore original order
        sorted_dense  = np.vstack(all_dense)
        sorted_sparse = all_sparse
        final_dense   = sorted_dense[restore]
        final_sparse  = [sorted_sparse[restore[i]] for i in range(len(texts))]

        return BGEOutput(final_dense, final_sparse)


# CLI smoke test  (python -m src.core.embedder)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    import argparse

    parser = argparse.ArgumentParser(description="Embedder smoke test")
    parser.add_argument(
        "--model", default=DEFAULT_EMBEDDING_MODEL,
        choices=list(EMBEDDING_MODELS),
        help="Model key to test",
    )
    args = parser.parse_args()

    texts = [
        "passage: Contrastive learning improves representation quality.",
        "passage: HNSW is an approximate nearest-neighbour index.",
    ]
    query = "What is contrastive learning?"

    print(f"\nModel  : {args.model}")
    emb = Embedder.from_model_key(args.model)

    result = emb.embed_passages(texts)
    if isinstance(result, BGEOutput):
        dense, sparse = result.dense, result.sparse
        print(f"Passage dense  shape : {dense.shape}")
        print(f"Passage dense  dtype : {dense.dtype}")
        print(f"Passage dense  norm  : {np.linalg.norm(dense, axis=1)}")
        print(f"Sparse[0] keys (first 5): {list(sparse[0].keys())[:5]}")
    else:
        print(f"Passage vecs shape : {result.shape}")
        print(f"Passage vecs dtype : {result.dtype}")
        print(f"Passage L2 norms   : {np.linalg.norm(result, axis=1)}")

    q_result = emb.embed_query(query)
    if isinstance(q_result, BGEOutput):
        print(f"Query dense shape  : {q_result.dense.shape}")
        print(f"Query dense norm   : {np.linalg.norm(q_result.dense):.6f}")
    else:
        print(f"Query vec shape    : {q_result.shape}")
        print(f"Query vec L2 norm  : {np.linalg.norm(q_result):.6f}")

    print("\nDone.")
