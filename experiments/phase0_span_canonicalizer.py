from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from experiments.phase0_late_interaction import _load_orbits
from spectra_v3.cache import EmbeddingCache
from spectra_v3.encoders import CachedEncoder, EncoderSpec, SentenceTransformerEncoder
from spectra_v3.lexical_variables import select_relevant_span


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for item in row["interventions"]:
            grouped[(item["family"], item["expected_relation"])].append(
                float(item["span_similarity_drop"])
            )
        for family in {key[0] for key in grouped}:
            critical = grouped.get((family, "change"))
            preserving = grouped.get((family, "preserve"))
            if critical and preserving:
                pairs[family].append((float(np.mean(critical)), float(np.mean(preserving))))
    return {
        family: {
            "orbit_count": len(values),
            "critical_drop_mean": float(np.mean([value[0] for value in values])),
            "preserve_drop_mean": float(np.mean([value[1] for value in values])),
            "critical_greater_fraction": float(
                np.mean([value[0] > value[1] for value in values])
            ),
        }
        for family, values in sorted(pairs.items())
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure pooled semantic compatibility on the query-relevant span."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser


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

    texts: list[str] = []
    spans: list[tuple[str, list[str]]] = []
    for orbit in orbits:
        base_span = select_relevant_span(orbit.context_text, orbit.base_text)
        transformed_spans = [
            select_relevant_span(orbit.context_text, item.transformed_text)
            for item in orbit.interventions
        ]
        texts.extend([orbit.context_text, base_span, *transformed_spans])
        spans.append((base_span, transformed_spans))

    spec = EncoderSpec(args.model, args.revision, normalize=True)
    base_encoder = SentenceTransformerEncoder(
        spec,
        device=args.device,
        batch_size=args.batch_size,
        show_progress=False,
    )
    started = perf_counter()
    with EmbeddingCache(args.cache) as cache:
        encoder = CachedEncoder(base_encoder, cache)
        embeddings = encoder.encode(texts)
        cache_stats = encoder.stats()

    rows = []
    cursor = 0
    for orbit, (_, transformed_spans) in zip(orbits, spans, strict=True):
        width = 2 + len(orbit.interventions)
        query_embedding = embeddings[cursor]
        base_similarity = float(query_embedding @ embeddings[cursor + 1])
        interventions = []
        for index, (item, _) in enumerate(
            zip(orbit.interventions, transformed_spans, strict=True)
        ):
            similarity = float(query_embedding @ embeddings[cursor + 2 + index])
            delta = similarity - base_similarity
            interventions.append(
                {
                    "family": item.family,
                    "expected_relation": item.expected_relation.value,
                    "generator_id": item.generator_id,
                    "span_similarity": similarity,
                    "span_similarity_delta": delta,
                    "span_similarity_drop": -delta,
                    "abs_span_similarity_delta": abs(delta),
                }
            )
        rows.append(
            {
                "base_id": orbit.base_id,
                "base_span_similarity": base_similarity,
                "interventions": interventions,
            }
        )
        cursor += width

    temporary = output.with_name(f"{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(output)
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "orbit_count": len(orbits),
        "elapsed_seconds": perf_counter() - started,
        "observer": "query-to-selected-span pooled cosine",
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output, summary = run(args)
    print(f"wrote {output}")
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
