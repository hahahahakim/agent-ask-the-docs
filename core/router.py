"""Semantic router — matches queries to documentation topics via embedding similarity.

Replaces the keyword-based _route_query() with embedding cosine similarity
against natural-language topic descriptions, using the same ONNXMiniLM_L6_V2
model already used for RAG indexing.

Topic embeddings are lazily computed and cached in-process on first call.
"""

from __future__ import annotations

import numpy as np
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

# ---------------------------------------------------------------------------
# Topic registry
# ---------------------------------------------------------------------------

TOPICS = [
    {
        "id": "storage-sdk",
        "description": (
            "uploading and downloading files using the 0G Storage SDK or CLI, "
            "file storage, indexer node, turbo rpc endpoint, standard rpc endpoint, "
            "flow contract, zgs, storage node"
        ),
        "urls": [
            "https://docs.0g.ai/developer-hub/building-on-0g/storage/sdk",
            "https://docs.0g.ai/developer-hub/building-on-0g/storage/storage-cli",
            "https://docs.0g.ai/developer-hub/network-info",
            "https://build.0g.ai/storage",
        ],
    },
    {
        "id": "da-integration",
        "description": (
            "submitting blobs to the 0G Data Availability layer, DA node integration, "
            "blob submission, data availability, da integration, 0G DA"
        ),
        "urls": [
            "https://docs.0g.ai/developer-hub/building-on-0g/da-integration",
        ],
    },
    {
        "id": "compute-network",
        "description": (
            "AI models available on 0G, inference API endpoint, compute network, "
            "LLM endpoint, list models, available models, inference model names, "
            "DeepSeek Qwen Whisper GLM, mainnet testnet models, compute SDK"
        ),
        "urls": [
            "https://pc.0g.ai/playground",
            "https://docs.0g.ai/developer-hub/building-on-0g/compute-network/inference",
            "https://build.0g.ai/compute",
        ],
    },
    {
        "id": "ai-overview",
        "description": (
            "what is 0G, 0G AI infrastructure overview, decentralized AI network, "
            "how 0G works, 0G ecosystem, AI context"
        ),
        "urls": [
            "https://docs.0g.ai/ai-context",
            "https://docs.0g.ai/",
            "https://build.0g.ai/",
        ],
    },
    {
        "id": "private-computer",
        "description": (
            "0G Private Computer, verifiable AI inference, pc.0g.ai, PC API, "
            "PC credits, verifiable inference, trusted execution"
        ),
        "urls": [
            "https://pc.0g.ai/",
            "https://pc.0g.ai/playground",
        ],
    },
    {
        "id": "network-info",
        "description": (
            "0G chain ID, RPC URL, network endpoints, contract addresses, "
            "network configuration, RPC endpoint, testnet mainnet endpoints"
        ),
        "urls": [
            "https://docs.0g.ai/developer-hub/network-info",
        ],
    },
    {
        "id": "staking",
        "description": (
            "delegate tokens on 0G, undelegation, staking, validator node, "
            "run a validator node, node operator, how to run a node on 0G, "
            "EVM smart contracts, Solidity, Hardhat, Foundry, "
            "deploy contract, chain info"
        ),
        "urls": [
            "https://docs.0g.ai/developer-hub/building-on-0g/contracts-on-0g/staking-interfaces",
            "https://build.0g.ai/chain",
            "https://app.0g.ai/",
        ],
    },
    {
        "id": "blog",
        "description": (
            "0G blog, announcements, news, 0G Pay, ecosystem updates, "
            "partnership announcements, go to market"
        ),
        # sitemap.xml is kept as a live-fetch fallback; individual post URLs
        # discovered from the sitemap are indexed into ChromaDB by warm_cache().
        "urls": [
            "https://0g.ai/sitemap.xml",
        ],
    },
]

# ---------------------------------------------------------------------------
# Similarity threshold
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# Embedding singleton
# ---------------------------------------------------------------------------

_embed_fn = None
_topic_embeddings = None  # list of numpy arrays, lazily computed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_query(query: str) -> list:
    """Return deduplicated URLs for all topics semantically similar to *query*.

    Uses cosine similarity between the query embedding and pre-computed topic
    description embeddings. Topics above SIMILARITY_THRESHOLD contribute their
    URLs to the result.

    Returns an empty list if no topic exceeds the threshold.
    """
    global _embed_fn, _topic_embeddings

    try:
        # Lazy-init embedding function singleton
        if _embed_fn is None:
            _embed_fn = ONNXMiniLM_L6_V2()

        # Lazy-compute topic description embeddings
        if _topic_embeddings is None:
            descriptions = [t["description"] for t in TOPICS]
            _topic_embeddings = np.array(_embed_fn(descriptions))

        # Embed the query
        query_emb = np.array(_embed_fn([query])[0])

        # Cosine similarity — MiniLM produces normalized vectors, so dot product == cosine sim
        similarities = _topic_embeddings @ query_emb

        # Collect URLs for top topics above the threshold (deduplicated, order-preserving)
        _MAX_TOPICS = 2
        _MAX_URLS = 4

        # Pair each above-threshold topic with its score, then sort descending
        matched = [
            (similarities[i], topic)
            for i, topic in enumerate(TOPICS)
            if similarities[i] >= SIMILARITY_THRESHOLD
        ]
        matched.sort(key=lambda x: x[0], reverse=True)

        # Take top _MAX_TOPICS, collect URLs, cap at _MAX_URLS
        urls: list = []
        for _, topic in matched[:_MAX_TOPICS]:
            for u in topic["urls"]:
                if u not in urls and len(urls) < _MAX_URLS:
                    urls.append(u)

        return urls

    except Exception:
        return []
