from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from .encoders import EncoderSpec, TextEncoder
from .features import OrbitFeatures, extract_orbit_features
from .interventions import InterventionOrbit
from .response import l2_normalize
from .token_diagnostics import TokenDiagnostics, token_sequence_diagnostics


@dataclass(frozen=True)
class OrbitMeasurement:
    orbit: InterventionOrbit
    encoder_spec: EncoderSpec
    orbit_features: OrbitFeatures
    weighting: str
    token_diagnostics: tuple[TokenDiagnostics, ...] | None
    context_base_similarity: float | None
    context_similarity_deltas: npt.NDArray[np.float64] | None

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        interventions = []
        for index, item in enumerate(self.orbit.interventions):
            record: dict[str, Any] = {
                "family": item.family,
                "expected_relation": item.expected_relation.value,
                "strength": item.strength,
                "generator_id": item.generator_id,
                "verification_status": item.verification_status.value,
                "metadata": dict(item.metadata),
                "response_norm": float(
                    self.orbit_features.unweighted_response_norms[index]
                ),
                "measurement_response_norm": float(
                    self.orbit_features.global_measurement.response_norms[index]
                ),
            }
            if include_text:
                record["transformed_text"] = item.transformed_text
            if self.token_diagnostics is not None:
                record.update(self.token_diagnostics[index].to_dict())
            if self.context_similarity_deltas is not None:
                delta = float(self.context_similarity_deltas[index])
                record["context_similarity_delta"] = delta
                record["abs_context_similarity_delta"] = abs(delta)
            interventions.append(record)

        result: dict[str, Any] = {
            "base_id": self.orbit.base_id,
            "source": self.orbit.source,
            "metadata": dict(self.orbit.metadata),
            "encoder": {
                "model_name": self.encoder_spec.model_name,
                "revision": self.encoder_spec.revision,
                "normalize": self.encoder_spec.normalize,
            },
            "measurement": {"weighting": self.weighting},
            "interventions": interventions,
            **self.orbit_features.to_dict(),
        }
        if include_text:
            result["base_text"] = self.orbit.base_text
        if self.context_base_similarity is not None:
            result["context"] = {"base_similarity": self.context_base_similarity}
            if include_text:
                result["context"]["text"] = self.orbit.context_text
        return result


def measure_orbit(
    orbit: InterventionOrbit,
    encoder: TextEncoder,
    *,
    rank: int = 8,
    weighting: str = "uniform",
) -> OrbitMeasurement:
    content_texts = [
        orbit.base_text,
        *(item.transformed_text for item in orbit.interventions),
    ]
    texts = [orbit.context_text, *content_texts] if orbit.context_text else content_texts
    embeddings = encoder.encode(texts)
    tokenize = getattr(encoder, "tokenize_ids", None)
    token_ids = tokenize(texts) if tokenize is not None else None
    if orbit.context_text:
        context_embedding = embeddings[0]
        content_embeddings = embeddings[1:]
        content_token_ids = token_ids[1:] if token_ids is not None else None
    else:
        context_embedding = None
        content_embeddings = embeddings
        content_token_ids = token_ids
    return measure_orbit_from_embeddings(
        orbit,
        encoder.spec,
        content_embeddings,
        token_ids=content_token_ids,
        context_embedding=context_embedding,
        rank=rank,
        weighting=weighting,
    )


def measure_orbit_from_embeddings(
    orbit: InterventionOrbit,
    encoder_spec: EncoderSpec,
    embeddings: npt.ArrayLike,
    *,
    token_ids: Sequence[Sequence[int]] | None = None,
    context_embedding: npt.ArrayLike | None = None,
    rank: int = 8,
    weighting: str = "uniform",
) -> OrbitMeasurement:
    array = np.asarray(embeddings)
    expected = 1 + len(orbit.interventions)
    if array.ndim != 2 or array.shape[0] != expected:
        raise RuntimeError("encoder returned an invalid embedding matrix")
    orbit_features = extract_orbit_features(
        array[0],
        array[1:],
        orbit.interventions,
        rank=rank,
        weighting=weighting,
    )
    token_diagnostics: tuple[TokenDiagnostics, ...] | None = None
    if token_ids is not None:
        if len(token_ids) != expected:
            raise RuntimeError("tokenizer returned the wrong number of sequences")
        token_diagnostics = tuple(
            token_sequence_diagnostics(token_ids[0], transformed_ids)
            for transformed_ids in token_ids[1:]
        )
    context_base_similarity: float | None = None
    context_similarity_deltas: npt.NDArray[np.float64] | None = None
    if context_embedding is not None:
        context = l2_normalize(context_embedding)
        normalized_content = l2_normalize(array, axis=1)
        if context.shape[0] != normalized_content.shape[1]:
            raise ValueError("context and content embedding dimensions must match")
        similarities = normalized_content @ context
        context_base_similarity = float(similarities[0])
        context_similarity_deltas = np.asarray(
            similarities[1:] - similarities[0], dtype=np.float64
        )
    return OrbitMeasurement(
        orbit=orbit,
        encoder_spec=encoder_spec,
        orbit_features=orbit_features,
        weighting=weighting,
        token_diagnostics=token_diagnostics,
        context_base_similarity=context_base_similarity,
        context_similarity_deltas=context_similarity_deltas,
    )
