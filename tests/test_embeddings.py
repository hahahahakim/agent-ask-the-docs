"""Unit tests for core/embeddings.py — the OpenAI client is mocked, no network."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core import embeddings
from core.embeddings import OGEmbeddingFunction


class _FakeEmbeddings:
    def __init__(self):
        self.calls: list = []

    def create(self, model, input):
        self.calls.append(list(input))
        # Unnormalised, deterministic vectors; returned out of order to test index sorting
        data = [SimpleNamespace(index=i, embedding=[float(len(t)), 3.0, 4.0]) for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))


def _make_fn(model="qwen3.7-text-embedding"):
    fn = OGEmbeddingFunction(model_name=model)
    fake = _FakeEmbeddings()
    fn._client = SimpleNamespace(embeddings=fake)
    return fn, fake


def test_documents_are_normalised_and_unprefixed():
    fn, fake = _make_fn()
    out = fn(["a", "bbb"])
    assert fake.calls == [["a", "bbb"]]
    for v in out:
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5
    # order preserved despite shuffled response
    assert out[0][0] < out[1][0]


def test_documents_are_batched():
    fn, fake = _make_fn()
    fn([f"doc {i}" for i in range(130)])
    assert [len(c) for c in fake.calls] == [20, 20, 20, 20, 20, 20, 10]


def test_query_is_prefixed_and_cached():
    fn, fake = _make_fn()
    first = fn.embed_query(["how do I upload?"])
    second = fn.embed_query(["how do I upload?"])
    assert len(fake.calls) == 1
    assert fake.calls[0][0].startswith("Instruct: ")
    assert fake.calls[0][0].endswith("Query: how do I upload?")
    assert np.array_equal(first[0], second[0])


def test_config_roundtrip_has_no_secrets():
    fn, _ = _make_fn("qwen3-embedding-0.6b")
    cfg = fn.get_config()
    assert cfg == {"model_name": "qwen3-embedding-0.6b"}
    assert OGEmbeddingFunction.build_from_config(cfg).model_name == "qwen3-embedding-0.6b"


def test_collection_name_is_scoped_to_model(monkeypatch):
    fn, _ = _make_fn("qwen3.7-text-embedding")
    monkeypatch.setattr(embeddings, "_embedding_fn", fn)
    assert embeddings.collection_name() == "0g_docs__qwen3-7-text-embedding"
