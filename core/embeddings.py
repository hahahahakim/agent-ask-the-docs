"""Remote embedding function backed by the 0G PC router (OpenAI-compatible /embeddings).

Shared by core/rag.py (ChromaDB collection) and core/router.py (topic matching)
through the get_embedding_function() singleton, so both use one client and one
query-embedding cache.

Qwen embedding models are instruction-aware: queries are prefixed with a task
instruction, documents are embedded as-is.
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function
from openai import OpenAI

DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding"

QUERY_INSTRUCTION = "Given a question about 0G documentation, retrieve relevant passages"

_BATCH_SIZE = 20  # qwen3.7-text-embedding rejects batches > 20
_QUERY_CACHE_SIZE = 512


@register_embedding_function
class OGEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB embedding function calling the 0G PC router's /embeddings endpoint."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self._client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            timeout=30,
            max_retries=1,
        )
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    # --- embedding -------------------------------------------------------

    def _embed(self, texts: list) -> list:
        vectors: list = []
        for i in range(0, len(texts), _BATCH_SIZE):
            resp = self._client.embeddings.create(model=self.model_name, input=texts[i:i + _BATCH_SIZE])
            for item in sorted(resp.data, key=lambda d: d.index):
                v = np.asarray(item.embedding, dtype=np.float32)
                # L2-normalise so dot product == cosine similarity (core/router.py relies on this)
                vectors.append(v / (np.linalg.norm(v) or 1.0))
        return vectors

    def __call__(self, input: Documents) -> Embeddings:
        """Embed documents (no instruction prefix)."""
        return self._embed(list(input))

    def embed_query(self, input: Documents) -> Embeddings:
        """Embed queries with the task instruction prefix; results are LRU-cached.

        route_query() and query_index() embed the same query on every request, so
        the cache turns that into a single network call.
        """
        queries = list(input)
        missing = [q for q in dict.fromkeys(queries) if q not in self._query_cache]
        if missing:
            prefixed = [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {q}" for q in missing]
            for q, v in zip(missing, self._embed(prefixed)):
                self._query_cache[q] = v
        out = []
        for q in queries:
            self._query_cache.move_to_end(q)
            out.append(self._query_cache[q])
        while len(self._query_cache) > _QUERY_CACHE_SIZE:
            self._query_cache.popitem(last=False)
        return out

    # --- ChromaDB config protocol -----------------------------------------

    @staticmethod
    def name() -> str:
        return "0g_pc_router"

    def default_space(self) -> str:
        return "cosine"

    def get_config(self) -> Dict[str, Any]:
        # Never persist credentials — they are read from the environment.
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "OGEmbeddingFunction":
        return OGEmbeddingFunction(model_name=config.get("model_name"))


_embedding_fn: OGEmbeddingFunction | None = None


def get_embedding_function() -> OGEmbeddingFunction:
    """Return the process-wide embedding function singleton."""
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = OGEmbeddingFunction()
    return _embedding_fn


def collection_name() -> str:
    """ChromaDB collection name scoped to the embedding model.

    Vectors from different models are incompatible (different dimensions and
    spaces), so each model gets its own collection. Switching EMBEDDING_MODEL
    therefore starts from an empty collection that warm_cache() re-populates.
    """
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", get_embedding_function().model_name).strip("-")
    return f"0g_docs__{slug}"
