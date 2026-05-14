"""
RAG (Retrieval-Augmented Generation) module for the 0G Labs Documentation Agent.

Owns the ChromaDB vector index lifecycle:
  - Initialization (lazy singleton PersistentClient + collection)
  - Indexing (chunking and upserting page content)
  - Querying (async semantic search with formatted context)
  - Chunk invalidation (drop_url_chunks for recache flows)

Embedding: ChromaDB's built-in ONNXMiniLM_L6_V2 (384-dim cosine, fully in-process).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

DISTANCE_THRESHOLD = 0.65

# Canonical set of URLs that are indexed into ChromaDB.
# script.py derives _WARM_URLS from this — add new stable doc pages here only.
INDEXABLE_URLS: frozenset = frozenset({
    "https://docs.0g.ai/",
    "https://build.0g.ai/",
    "https://pc.0g.ai/",
    "https://app.0g.ai/",
    "https://docs.0g.ai/developer-hub/building-on-0g/storage/storage-cli",
    "https://docs.0g.ai/developer-hub/building-on-0g/storage/sdk",
    "https://docs.0g.ai/developer-hub/building-on-0g/compute-network/inference",
    "https://docs.0g.ai/developer-hub/building-on-0g/da",
    "https://docs.0g.ai/developer-hub/network-info",
    "https://build.0g.ai/chain",
    "https://build.0g.ai/storage",
    "https://build.0g.ai/compute",
    "https://docs.0g.ai/ai-context",
})

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_embedding_function() -> ONNXMiniLM_L6_V2:
    """Return a ChromaDB ONNXMiniLM_L6_V2 embedding function instance."""
    return ONNXMiniLM_L6_V2()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_collection() -> chromadb.Collection:
    """Return the singleton ChromaDB collection.

    Lazy-initialises the PersistentClient and collection on first call.
    Thread-safe enough for single-worker use.
    """
    global _client, _collection

    if _collection is not None:
        return _collection

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
    _collection = _client.get_or_create_collection(
        name="0g_docs",
        metadata={"hnsw:space": "cosine"},
        embedding_function=_get_embedding_function(),
    )
    return _collection


def index_urls(urls: list, content_map: dict) -> None:
    """Index parsed page content into ChromaDB.

    Synchronous — callers should use asyncio.to_thread when calling from
    an async context.

    Args:
        urls:        List of URLs to consider for indexing.
        content_map: Mapping of url -> formatted page text (as produced by
                     _parse() in script.py, including a ``URL: <url>`` header
                     line).
    """
    col = get_collection()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40,
        separators=["\n\n", "\n", ". ", " "],
    )

    for url in urls:
        # Skip sitemap content — never indexed.
        if "sitemap" in url:
            continue

        try:
            # --- 2a: delete existing chunks for this URL ---
            existing = col.get(where={"url": url})
            existing_ids = existing.get("ids", [])
            if existing_ids:
                col.delete(ids=existing_ids)

            # --- 2b: extract body text (strip header up to "URL: <url>" line) ---
            raw = content_map.get(url, "")
            header_marker = f"URL: {url}"
            marker_pos = raw.find(header_marker)
            if marker_pos != -1:
                # Skip past the marker line (find the newline that ends it)
                after_marker = raw.find("\n", marker_pos)
                body = raw[after_marker + 1:] if after_marker != -1 else ""
            else:
                body = raw

            if not body.strip():
                continue

            # --- 2c: chunk ---
            chunks = splitter.split_text(body)
            if not chunks:
                continue

            # --- 2d: deterministic chunk IDs ---
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            ids = [f"{url_hash}_{i}" for i in range(len(chunks))]

            # --- 2e: upsert ---
            indexed_at = time.time()
            col.upsert(
                ids=ids,
                documents=chunks,
                metadatas=[{"url": url, "indexed_at": indexed_at} for _ in chunks],
            )

        except Exception as e:
            print(f"[rag] index error for {url}: {e}")
            continue


async def query_index(query: str, n_results: int = 6) -> tuple:
    """Query the vector index and return formatted context with min distance.

    Args:
        query:     The natural-language query string.
        n_results: Maximum number of result chunks to retrieve (default 6).

    Returns:
        A ``(context_str, min_distance)`` tuple.
        Returns ``("", 1.0)`` when the index is empty or yields no results.
    """
    col = get_collection()

    count = await asyncio.to_thread(col.count)
    if count == 0:
        return ("", 1.0)

    results = await asyncio.to_thread(
        col.query,
        query_texts=[query],
        n_results=min(n_results, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        return ("", 1.0)

    min_distance = min(distances)

    # --- Group chunks by URL, preserving order and deduplicating URL headers ---
    seen_urls: list[str] = []
    url_chunks: dict[str, list[str]] = {}

    for doc, meta in zip(documents, metadatas):
        url = meta.get("url", "")
        if url not in url_chunks:
            seen_urls.append(url)
            url_chunks[url] = []
        url_chunks[url].append(doc)

    separator = "=" * 72
    sections: list[str] = []

    for url in seen_urls:
        chunks = url_chunks[url]
        chunk_text = "\n\n".join(chunks)
        section = f"### [RAG] {url}\nURL: {url}\n\n{chunk_text}"
        sections.append(section)

    formatted_context = f"\n{separator}\n".join(sections)

    return (formatted_context, min_distance)


def clear_index() -> str:
    """Drop and recreate the ChromaDB collection, removing all indexed chunks.

    Synchronous. Used by the API recache handler to wipe the entire RAG index.
    Returns a human-readable confirmation string.
    """
    global _client, _collection
    if _client is None:
        get_collection()  # ensure initialised
    if _client is not None:
        _client.delete_collection("0g_docs")
        _collection = _client.get_or_create_collection(
            name="0g_docs",
            metadata={"hnsw:space": "cosine"},
            embedding_function=_get_embedding_function(),
        )
        return "RAG index cleared."
    return "RAG index not initialised — nothing to clear."


def drop_url_chunks(url: str) -> None:
    """Delete all indexed chunks for the given URL.

    Synchronous. Used when ``recache <url>`` is called from the CLI.

    Args:
        url: The URL whose chunks should be removed from the index.
    """
    col = get_collection()
    existing = col.get(where={"url": url})
    existing_ids = existing.get("ids", [])
    if existing_ids:
        col.delete(ids=existing_ids)
