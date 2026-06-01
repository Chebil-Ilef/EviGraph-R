from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qdrant_client import QdrantClient
from evigraph.config.settings import QDRANT_ACTIVE, QDRANT_CONNECTION

import re

_FORMULA_RE = re.compile(r"\{\{formula:[^}]+\}\}")
_REF_RE = re.compile(r"\bREF\b")


@dataclass
class ContextGroup:
    category: int                   # 1–4
    domain: str                     # cs | physics | math | stats_other
    paper_ids: list[str]            # arXiv IDs of the contributing papers
    chunk_ids: list[str]            # chunk_uid values
    texts: list[str]                # embed_text values (same order as chunk_ids)
    metadata: dict = field(default_factory=dict)  # extra fields for diagnostics


class QdrantSamplerBase:

    def __init__(self, collection_name: Optional[str] = None) -> None:
        conn = QDRANT_CONNECTION
        if conn.url:
            self._client = QdrantClient(
                url=conn.url, api_key=conn.api_key, timeout=conn.timeout
            )
        else:
            self._client = QdrantClient(
                host=conn.host,
                port=conn.port,
                grpc_port=conn.grpc_port,
                prefer_grpc=conn.prefer_grpc,
                timeout=conn.timeout,
            )
        self._collection = collection_name or QDRANT_ACTIVE.collection_name


    def scroll_all(
        self,
        scroll_filter,
        limit: int = 200,
        with_vectors: bool = False,
    ) -> list:
        points: list = []
        offset = None
        while True:
            batch, next_offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=scroll_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset
        return points

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def is_usable_chunk(text: str, max_formula_ratio: float = 0.40) -> bool:

        if not text:
            return False
        non_ws = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
        if non_ws == 0:
            return False
        placeholder_chars = sum(len(m) for m in _FORMULA_RE.findall(text))
        placeholder_chars += sum(len(m) for m in _REF_RE.findall(text))
        return (placeholder_chars / non_ws) < max_formula_ratio
