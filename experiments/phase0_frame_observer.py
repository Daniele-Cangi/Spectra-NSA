from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from experiments.phase0_late_interaction import _load_orbits
from spectra_v3.cache import EmbeddingCache
from spectra_v3.encoders import CachedEncoder, EncoderSpec, SentenceTransformerEncoder
from spectra_v3.frame_variables import (
    SemanticFrame,
    extract_semantic_frame,
    fixed_frame_distance,
    frame_compatibility,
)
from spectra_v3.interventions import InterventionOrbit


def _frame_texts(frame: SemanticFrame) -> tuple[str, ...]:
    return tuple(
        value
        for value in (frame.span, frame.predicate, frame.actor, frame.patient)
        if value
    )


def _prepare_frames(
    orbits: Sequence[InterventionOrbit],
) -> list[tuple[SemanticFrame, SemanticFrame, list[SemanticFrame]]]:
    prepared = []
    for orbit in orbits:
        query = extract_semantic_frame(orbit.context_text, orbit.context_text)
        base = extract_semantic_frame(orbit.context_text, orbit.base_text)
        transformed = [
            extract_semantic_frame(orbit.context_text, item.transformed_text)
            for item in orbit.interventions
        ]
        prepared.append((query, base, transformed))
    return prepared


def _embedding_lookup(
    texts: Sequence[str], embeddings: np.ndarray
) -> Callable[[str, str], float]:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError("embedding matrix does not match frame texts")
    lookup = {text: embeddings[index] for index, text in enumerate(texts)}

    def similarity(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        return float(lookup[left] @ lookup[right])

    return similarity


def observe_orbits(
    orbits: Sequence[InterventionOrbit],
    prepared: Sequence[tuple[SemanticFrame, SemanticFrame, list[SemanticFrame]]],
    similarity: Callable[[str, str], float],
) -> list[dict[str, Any]]:
    if len(orbits) != len(prepared):
        raise ValueError("orbits and prepared frames have different lengths")
    rows: list[dict[str, Any]] = []
    for orbit, (query, base, transformed_frames) in zip(
        orbits, prepared, strict=True
    ):
        if len(transformed_frames) != len(orbit.interventions):
            raise ValueError(f"frame count mismatch for {orbit.base_id}")
        base_state = frame_compatibility(query, base, similarity)
        base_span_similarity = similarity(query.span, base.span)
        interventions = []
        for item, frame in zip(
            orbit.interventions, transformed_frames, strict=True
        ):
            state = frame_compatibility(query, frame, similarity)
            drops = state.drops_from(base_state)
            span_similarity = similarity(query.span, frame.span)
            interventions.append(
                {
                    "family": item.family,
                    "expected_relation": item.expected_relation.value,
                    "generator_id": item.generator_id,
                    "selected_span_similarity": span_similarity,
                    "selected_span_similarity_drop": max(
                        base_span_similarity - span_similarity, 0.0
                    ),
                    "frame": frame.to_dict(),
                    "frame_compatibility": state.to_dict(),
                    "frame_coordinate_drops": drops,
                    "frame_distance": fixed_frame_distance(drops),
                    "observer_reliability": state.reliability,
                }
            )
        rows.append(
            {
                "base_id": orbit.base_id,
                "metadata": {
                    "template_id": orbit.metadata.get("template_id"),
                    "predicate_family": orbit.metadata.get("predicate_family"),
                    "evaluation_partition": orbit.metadata.get(
                        "evaluation_partition"
                    ),
                },
                "query_frame": query.to_dict(),
                "base_frame": base.to_dict(),
                "base_selected_span_similarity": base_span_similarity,
                "base_frame_compatibility": base_state.to_dict(),
                "interventions": interventions,
            }
        )
    return rows


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paired: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for row in rows:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in row["interventions"]:
            grouped[(item["family"], item["expected_relation"])].append(item)
        for family in {key[0] for key in grouped}:
            critical = grouped.get((family, "change"), [])
            preserving = grouped.get((family, "preserve"), [])
            if not critical or not preserving:
                continue
            paired[family].append(
                (
                    float(np.mean([x["frame_distance"] for x in critical])),
                    float(np.mean([x["frame_distance"] for x in preserving])),
                    float(
                        np.mean(
                            [x["selected_span_similarity_drop"] for x in critical]
                        )
                    ),
                    float(
                        np.mean(
                            [x["selected_span_similarity_drop"] for x in preserving]
                        )
                    ),
                )
            )
    return {
        family: {
            "orbit_count": len(values),
            "frame_critical_mean": float(np.mean([x[0] for x in values])),
            "frame_preserve_mean": float(np.mean([x[1] for x in values])),
            "frame_critical_greater_fraction": float(
                np.mean([x[0] > x[1] for x in values])
            ),
            "span_critical_mean": float(np.mean([x[2] for x in values])),
            "span_preserve_mean": float(np.mean([x[3] for x in values])),
        }
        for family, values in sorted(paired.items())
    }


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.revision.casefold() in {"head", "latest", "main", "master"}:
        raise ValueError("revision must identify an immutable model snapshot")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    output = args.output.resolve()
    summary_path = output.with_name(f"{output.name}.summary.json")
    existing = [path for path in (output, summary_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite: {', '.join(map(str, existing))}")
    output.parent.mkdir(parents=True, exist_ok=True)
    orbits = _load_orbits(args.input)
    prepared = _prepare_frames(orbits)
    texts = list(
        dict.fromkeys(
            text
            for bundle in prepared
            for frame in (bundle[0], bundle[1], *bundle[2])
            for text in _frame_texts(frame)
        )
    )

    spec = EncoderSpec(args.model, args.revision, normalize=True)
    base_encoder = SentenceTransformerEncoder(
        spec,
        device=args.device,
        batch_size=args.batch_size,
        show_progress=args.show_progress,
    )
    started = perf_counter()
    with EmbeddingCache(args.cache) as cache:
        encoder = CachedEncoder(base_encoder, cache)
        embeddings = encoder.encode(texts)
        cache_stats = encoder.stats()
    rows = observe_orbits(orbits, prepared, _embedding_lookup(texts, embeddings))

    temporary = output.with_name(f"{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(output)
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "claim_eligible": False,
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "orbit_count": len(orbits),
        "observer": "selected-span predicate/argument frame with fixed monotone norm",
        "elapsed_seconds": perf_counter() - started,
        "encoder": {
            "model_name": spec.model_name,
            "revision": spec.revision,
            "runtime": base_encoder.runtime_metadata(),
            "cache": cache_stats,
        },
        "matched_axis_diagnostics": _summarize(rows),
        "output": str(output),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, summary_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure explicit predicate/argument frame variables."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output, summary = run(args)
    print(f"wrote {output}")
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
