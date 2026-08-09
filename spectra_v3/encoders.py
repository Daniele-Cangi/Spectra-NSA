from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from importlib.util import find_spec
import json
import platform
import sys
from typing import Any, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from .cache import CacheIdentity, EmbeddingCache
from .response import l2_normalize


@dataclass(frozen=True)
class EncoderSpec:
    model_name: str
    revision: str
    normalize: bool = True

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if not self.revision.strip():
            raise ValueError("revision cannot be empty")

    @property
    def cache_identity(self) -> CacheIdentity:
        return CacheIdentity(self.model_name, self.revision, self.normalize)


class TextEncoder(Protocol):
    spec: EncoderSpec

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]: ...


class SentenceTransformerEncoder:
    """Lazy adapter so numerical Phase 0 tests do not require PyTorch."""

    def __init__(
        self,
        spec: EncoderSpec,
        *,
        device: str | None = None,
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if find_spec("sentence_transformers") is None:
            raise RuntimeError(
                "Sentence Transformers is required for encoding. "
                "Install the phase0 extra with: pip install -e '.[phase0]'"
            )

        self.spec = spec
        self.device = device
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        import torch

        resolved_request = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_name = (
            torch.cuda.get_device_name(0)
            if resolved_request.startswith("cuda") and torch.cuda.is_available()
            else None
        )
        self.cache_identity = CacheIdentity(
            model_name=spec.model_name,
            revision=spec.revision,
            normalize=spec.normalize,
            inference_fingerprint=json.dumps(
                {
                    "batch_size": batch_size,
                    "device": resolved_request,
                    "device_name": device_name,
                    "sentence_transformers": version("sentence-transformers"),
                    "torch": version("torch"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def _ensure_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.spec.model_name,
                revision=self.spec.revision,
                device=self.device,
                trust_remote_code=False,
            )
            self._tokenizer = self._model.tokenizer
        return self._model

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.spec.model_name,
                revision=self.spec.revision,
                trust_remote_code=False,
            )
        return self._tokenizer

    def runtime_metadata(self) -> dict[str, str | bool | None]:
        import torch

        resolved_device = str(self._model.device) if self._model is not None else None
        cuda_available = torch.cuda.is_available()
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": version("torch"),
            "sentence_transformers": version("sentence-transformers"),
            "numpy": version("numpy"),
            "requested_device": self.device,
            "batch_size": self.batch_size,
            "resolved_device": resolved_device,
            "model_loaded": self._model is not None,
            "tokenizer_loaded": self._tokenizer is not None,
            "cuda_available": cuda_available,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
            "cache_inference_fingerprint": self.cache_identity.inference_fingerprint,
        }

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        if not texts:
            raise ValueError("texts cannot be empty")
        embeddings = self._ensure_model().encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.spec.normalize,
            show_progress_bar=self.show_progress,
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(texts):
            raise RuntimeError("encoder returned an invalid embedding matrix")
        if not np.isfinite(array).all():
            raise RuntimeError("encoder returned non-finite embeddings")
        if self.spec.normalize:
            array = l2_normalize(array, axis=1).astype(np.float32)
        return array

    def tokenize_ids(self, texts: Sequence[str]) -> list[list[int]]:
        if not texts:
            raise ValueError("texts cannot be empty")
        encoded = self._ensure_tokenizer()(
            list(texts),
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        return [[int(token) for token in ids] for ids in encoded["input_ids"]]


class CachedEncoder:
    def __init__(self, encoder: TextEncoder, cache: EmbeddingCache) -> None:
        self.encoder = encoder
        self.cache = cache
        self.spec = encoder.spec
        self.cache_identity = getattr(encoder, "cache_identity", self.spec.cache_identity)
        self.cache_lookup_hits = 0
        self.cache_lookup_misses = 0
        self.encoded_unique_texts = 0

    def stats(self) -> dict[str, int]:
        return {
            "lookup_hits": self.cache_lookup_hits,
            "lookup_misses": self.cache_lookup_misses,
            "encoded_unique_texts": self.encoded_unique_texts,
        }

    def tokenize_ids(self, texts: Sequence[str]) -> list[list[int]] | None:
        tokenize = getattr(self.encoder, "tokenize_ids", None)
        if tokenize is None:
            return None
        return tokenize(texts)

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        if not texts:
            raise ValueError("texts cannot be empty")
        if any(not text for text in texts):
            raise ValueError("texts cannot contain empty values")

        resolved: list[npt.NDArray[np.float32] | None] = [None] * len(texts)
        missing: dict[str, list[int]] = {}
        for index, value in enumerate(texts):
            cached = self.cache.get(self.cache_identity, value)
            if cached is not None:
                self.cache_lookup_hits += 1
                resolved[index] = cached
            else:
                self.cache_lookup_misses += 1
                missing.setdefault(value, []).append(index)

        if missing:
            missing_texts = list(missing)
            encoded = self.encoder.encode(missing_texts)
            self.encoded_unique_texts += len(missing_texts)
            if encoded.shape[0] != len(missing_texts):
                raise RuntimeError("encoder returned the wrong number of embeddings")
            for text, vector in zip(missing_texts, encoded, strict=True):
                self.cache.put(self.cache_identity, text, vector)
                for index in missing[text]:
                    resolved[index] = np.asarray(vector, dtype=np.float32)

        if any(vector is None for vector in resolved):
            raise RuntimeError("failed to resolve all embeddings")
        return np.stack([vector for vector in resolved if vector is not None])
