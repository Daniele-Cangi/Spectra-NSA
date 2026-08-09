from pathlib import Path

import numpy as np

from spectra_v3.cache import CacheIdentity, EmbeddingCache, embedding_cache_key
from spectra_v3.encoders import CachedEncoder, EncoderSpec


class FakeEncoder:
    def __init__(self) -> None:
        self.spec = EncoderSpec("fake", "commit-1", normalize=False)
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.asarray(
            [[len(text), ord(text[0])] for text in texts], dtype=np.float32
        )


def test_cache_key_is_revision_and_normalization_aware() -> None:
    base = CacheIdentity("model", "revision-a", True)

    assert embedding_cache_key(base, "text") != embedding_cache_key(
        CacheIdentity("model", "revision-b", True), "text"
    )
    assert embedding_cache_key(base, "text") != embedding_cache_key(
        CacheIdentity("model", "revision-a", False), "text"
    )
    assert embedding_cache_key(base, "text") != embedding_cache_key(
        CacheIdentity("model", "revision-a", True, "torch=other"), "text"
    )


def test_embedding_cache_round_trip(tmp_path: Path) -> None:
    identity = CacheIdentity("model", "commit", True, "runtime-v1")
    with EmbeddingCache(tmp_path / "embeddings.sqlite") as cache:
        assert cache.get(identity, "hello") is None
        cache.put(identity, "hello", [1.0, 2.0, 3.0])

        np.testing.assert_allclose(cache.get(identity, "hello"), [1.0, 2.0, 3.0])
        assert cache.count() == 1


def test_cached_encoder_deduplicates_misses_and_preserves_order(tmp_path: Path) -> None:
    underlying = FakeEncoder()
    with EmbeddingCache(tmp_path / "embeddings.sqlite") as cache:
        encoder = CachedEncoder(underlying, cache)
        first = encoder.encode(["a", "bb", "a"])
        second = encoder.encode(["bb", "a"])

    assert underlying.calls == [["a", "bb"]]
    assert encoder.stats() == {
        "lookup_hits": 2,
        "lookup_misses": 3,
        "encoded_unique_texts": 2,
    }
    np.testing.assert_allclose(first, [[1, 97], [2, 98], [1, 97]])
    np.testing.assert_allclose(second, [[2, 98], [1, 97]])
