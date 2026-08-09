from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from spectra_v3.cache import EmbeddingCache
from spectra_v3.encoders import CachedEncoder, EncoderSpec, SentenceTransformerEncoder
from spectra_v3.interventions import InterventionOrbit
from spectra_v3.pipeline import measure_orbit_from_embeddings
from spectra_v3.response import DEFAULT_SPECTRUM_ATOL, DEFAULT_SPECTRUM_RTOL


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure Spectra v3 semantic response spectra from intervention orbits."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input orbit JSONL file."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output feature JSONL file."
    )
    parser.add_argument(
        "--model", required=True, help="Sentence Transformers model name."
    )
    parser.add_argument(
        "--revision",
        required=True,
        help=(
            "Immutable model revision or commit hash; floating defaults are "
            "intentionally forbidden."
        ),
    )
    parser.add_argument(
        "--cache", type=Path, required=True, help="SQLite embedding cache."
    )
    parser.add_argument(
        "--rank", type=int, default=8, help="Maximum retained response rank."
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--weighting",
        choices=("uniform", "strength"),
        default="uniform",
        help="Response-row weighting; uniform is the non-leaking default.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include raw base and transformed texts in derived output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output and manifest.",
    )
    return parser


def _load_orbits(path: Path) -> list[InterventionOrbit]:
    orbits: list[InterventionOrbit] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("record must be a JSON object")
                orbits.append(InterventionOrbit.from_dict(value))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid orbit at {path}:{line_number}: {exc}"
                ) from exc
    if not orbits:
        raise ValueError(f"input contains no intervention orbits: {path}")
    base_ids = [orbit.base_id for orbit in orbits]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError("base_id values must be unique within an input file")
    return orbits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_targets_available(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing artifact(s): {joined}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    started = perf_counter()
    if args.rank <= 0:
        raise ValueError("rank must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.revision.casefold() in {"head", "latest", "main", "master"}:
        raise ValueError("revision must identify an immutable model snapshot")
    if not args.input.is_file():
        raise FileNotFoundError(f"input file does not exist: {args.input}")

    output = args.output.resolve()
    manifest = output.with_name(f"{output.name}.manifest.json")
    _ensure_targets_available((output, manifest), overwrite=args.overwrite)
    orbits = _load_orbits(args.input)

    spec = EncoderSpec(model_name=args.model, revision=args.revision, normalize=True)
    base_encoder = SentenceTransformerEncoder(
        spec,
        device=args.device,
        batch_size=args.batch_size,
        show_progress=args.show_progress,
    )

    temporary_output = output.with_name(f"{output.name}.tmp")
    intervention_count = 0
    try:
        with (
            EmbeddingCache(args.cache) as cache,
            temporary_output.open("w", encoding="utf-8", newline="\n") as handle,
        ):
            encoder = CachedEncoder(base_encoder, cache)
            all_texts: list[str] = []
            for orbit in orbits:
                if orbit.context_text:
                    all_texts.append(orbit.context_text)
                all_texts.extend(
                    [
                        orbit.base_text,
                        *(item.transformed_text for item in orbit.interventions),
                    ]
                )
            all_embeddings = encoder.encode(all_texts)
            all_token_ids = encoder.tokenize_ids(all_texts)
            cursor = 0
            for orbit in orbits:
                width = 1 + len(orbit.interventions)
                if orbit.context_text:
                    context_embedding = all_embeddings[cursor]
                    content_start = cursor + 1
                else:
                    context_embedding = None
                    content_start = cursor
                end = content_start + width
                measurement = measure_orbit_from_embeddings(
                    orbit,
                    spec,
                    all_embeddings[content_start:end],
                    token_ids=(
                        all_token_ids[content_start:end]
                        if all_token_ids is not None
                        else None
                    ),
                    context_embedding=context_embedding,
                    rank=args.rank,
                    weighting=args.weighting,
                )
                handle.write(
                    json.dumps(
                        measurement.to_dict(include_text=args.include_text),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                intervention_count += len(orbit.interventions)
                cursor = end
            cache_entries = cache.count()
            cache_stats = encoder.stats()
        temporary_output.replace(output)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise

    _write_json(
        manifest,
        {
            "schema_version": 4,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input": {
                "path": str(args.input.resolve()),
                "sha256": _sha256(args.input),
                "orbit_count": len(orbits),
                "intervention_count": intervention_count,
            },
            "encoder": {
                "model_name": spec.model_name,
                "revision": spec.revision,
                "normalize": spec.normalize,
                "cache_inference_fingerprint": (
                    base_encoder.cache_identity.inference_fingerprint
                ),
            },
            "runtime": base_encoder.runtime_metadata(),
            "measurement": {
                "rank": args.rank,
                "tangent_projection": True,
                "weighting": args.weighting,
                "spectrum_noise_atol": DEFAULT_SPECTRUM_ATOL,
                "spectrum_noise_rtol": DEFAULT_SPECTRUM_RTOL,
                "include_text": args.include_text,
            },
            "cache": {
                "path": str(args.cache.resolve()),
                "entry_count": cache_entries,
                **cache_stats,
            },
            "elapsed_seconds": perf_counter() - started,
            "output": str(output),
        },
    )
    return output, manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output, manifest = run(args)
    print(f"wrote {output}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
