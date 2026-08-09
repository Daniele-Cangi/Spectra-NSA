from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class CacheIdentity:
    model_name: str
    revision: str
    normalize: bool
    inference_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if not self.revision.strip():
            raise ValueError("revision cannot be empty")


def embedding_cache_key(identity: CacheIdentity, text: str) -> str:
    if not text:
        raise ValueError("text cannot be empty")
    payload = {
        "model_name": identity.model_name,
        "revision": identity.revision,
        "normalize": identity.normalize,
        "inference_fingerprint": identity.inference_fingerprint,
        "text": text,
    }
    serialized = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Small revision-aware SQLite cache for deterministic Phase 0 measurements."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                revision TEXT NOT NULL,
                normalize INTEGER NOT NULL,
                inference_fingerprint TEXT NOT NULL DEFAULT '',
                text_hash TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                vector BLOB NOT NULL,
                created_utc TEXT NOT NULL
            )
            """
        )
        columns = {
            row[1] for row in self._connection.execute("PRAGMA table_info(embeddings)")
        }
        if "inference_fingerprint" not in columns:
            self._connection.execute(
                "ALTER TABLE embeddings ADD COLUMN "
                "inference_fingerprint TEXT NOT NULL DEFAULT ''"
            )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def get(self, identity: CacheIdentity, text: str) -> npt.NDArray[np.float32] | None:
        key = embedding_cache_key(identity, text)
        row = self._connection.execute(
            "SELECT dimension, dtype, vector FROM embeddings WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        dimension, dtype_name, payload = row
        dtype = np.dtype(dtype_name)
        vector = np.frombuffer(payload, dtype=dtype)
        if vector.shape != (dimension,):
            raise RuntimeError(f"corrupt embedding cache entry: {key}")
        return vector.astype(np.float32, copy=True)

    def put(self, identity: CacheIdentity, text: str, vector: npt.ArrayLike) -> None:
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("cached embedding must be a non-empty vector")
        if not np.isfinite(array).all():
            raise ValueError("cached embedding must contain only finite values")

        key = embedding_cache_key(identity, text)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self._connection.execute(
            """
            INSERT OR REPLACE INTO embeddings (
                cache_key, model_name, revision, normalize, text_hash,
                inference_fingerprint, dimension, dtype, vector, created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                identity.model_name,
                identity.revision,
                int(identity.normalize),
                text_hash,
                identity.inference_fingerprint,
                int(array.size),
                array.dtype.str,
                array.tobytes(order="C"),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._connection.commit()

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return int(row[0])
