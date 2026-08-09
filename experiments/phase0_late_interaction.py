from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from spectra_v3.interventions import InterventionOrbit


def maxsim_score(query_tokens: torch.Tensor, document_tokens: torch.Tensor) -> float:
    if query_tokens.ndim != 2 or document_tokens.ndim != 2:
        raise ValueError("token embeddings must be two-dimensional")
    if query_tokens.shape[1] != document_tokens.shape[1]:
        raise ValueError("query and document token dimensions must match")
    if query_tokens.shape[0] == 0 or document_tokens.shape[0] == 0:
        raise ValueError("token embedding sequences cannot be empty")
    query = functional.normalize(query_tokens.float(), dim=1)
    document = functional.normalize(document_tokens.float(), dim=1)
    similarities = query @ document.T
    return float(similarities.max(dim=1).values.mean().item())


def _load_orbits(path: Path) -> list[InterventionOrbit]:
    orbits: list[InterventionOrbit] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            orbit = InterventionOrbit.from_dict(value)
            if not orbit.context_text:
                raise ValueError(f"orbit has no context_text at {path}:{line_number}")
            orbits.append(orbit)
    if not orbits:
        raise ValueError("input contains no orbits")
    return orbits


def _without_special_tokens(
    embeddings: torch.Tensor, special_mask: Sequence[int]
) -> torch.Tensor:
    mask = torch.tensor(
        [not bool(value) for value in special_mask],
        dtype=torch.bool,
        device=embeddings.device,
    )
    if mask.shape[0] != embeddings.shape[0]:
        raise RuntimeError("special-token mask and embeddings have different lengths")
    return embeddings[mask]


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for row in rows:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for intervention in row["interventions"]:
            grouped[
                (intervention["family"], intervention["expected_relation"])
            ].append(intervention)
        for family in {key[0] for key in grouped}:
            critical = grouped.get((family, "change"))
            preserving = grouped.get((family, "preserve"))
            if not critical or not preserving:
                continue
            pairs[family].append(
                (
                    float(np.mean([item["abs_maxsim_delta"] for item in critical])),
                    float(np.mean([item["abs_maxsim_delta"] for item in preserving])),
                    float(np.mean([item["maxsim_delta"] for item in critical])),
                    float(np.mean([item["maxsim_delta"] for item in preserving])),
                )
            )

    result: dict[str, Any] = {}
    for family, values in sorted(pairs.items()):
        array = np.asarray(values, dtype=np.float64)
        result[family] = {
            "orbit_count": int(array.shape[0]),
            "critical_abs_delta_mean": float(array[:, 0].mean()),
            "preserve_abs_delta_mean": float(array[:, 1].mean()),
            "critical_abs_greater_fraction": float(np.mean(array[:, 0] > array[:, 1])),
            "critical_signed_delta_mean": float(array[:, 2].mean()),
            "preserve_signed_delta_mean": float(array[:, 3].mean()),
            "critical_more_negative_fraction": float(np.mean(array[:, 2] < array[:, 3])),
        }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate query-document token late interaction on intervention orbits."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda")
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
    for orbit in orbits:
        texts.extend(
            [
                orbit.context_text,
                orbit.base_text,
                *(item.transformed_text for item in orbit.interventions),
            ]
        )

    from sentence_transformers import SentenceTransformer

    started = perf_counter()
    model = SentenceTransformer(
        args.model,
        revision=args.revision,
        device=args.device,
        trust_remote_code=False,
    )
    token_embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        output_value="token_embeddings",
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    tokenized = model.tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_special_tokens_mask=True,
    )
    cleaned = [
        _without_special_tokens(embedding, mask)
        for embedding, mask in zip(
            token_embeddings,
            tokenized["special_tokens_mask"],
            strict=True,
        )
    ]

    rows: list[dict[str, Any]] = []
    cursor = 0
    for orbit in orbits:
        query = cleaned[cursor]
        base = cleaned[cursor + 1]
        base_score = maxsim_score(query, base)
        interventions = []
        for index, intervention in enumerate(orbit.interventions):
            score = maxsim_score(query, cleaned[cursor + 2 + index])
            delta = score - base_score
            interventions.append(
                {
                    "family": intervention.family,
                    "expected_relation": intervention.expected_relation.value,
                    "generator_id": intervention.generator_id,
                    "maxsim_score": score,
                    "maxsim_delta": delta,
                    "abs_maxsim_delta": abs(delta),
                }
            )
        rows.append(
            {
                "base_id": orbit.base_id,
                "base_maxsim_score": base_score,
                "interventions": interventions,
            }
        )
        cursor += 2 + len(orbit.interventions)

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
        "encoder": {
            "model_name": args.model,
            "revision": args.revision,
            "torch": version("torch"),
            "sentence_transformers": version("sentence-transformers"),
            "device": str(model.device),
            "batch_size": args.batch_size,
        },
        "elapsed_seconds": perf_counter() - started,
        "scoring": "query_token_to_document_token_mean_max_cosine",
        "matched_family_diagnostics": _summarize(rows),
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
