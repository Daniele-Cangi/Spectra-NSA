from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.phase0_late_interaction import _load_orbits
from spectra_v3.semantic_variables import relational_state_from_probabilities


def find_entailment_index(id_to_label: Mapping[int, str]) -> int:
    matches = [
        int(index)
        for index, label in id_to_label.items()
        if "entail" in str(label).casefold()
    ]
    if len(matches) != 1:
        raise ValueError(f"cannot identify entailment label: {dict(id_to_label)}")
    return matches[0]


def labelled_probabilities(
    scores: np.ndarray, id_to_label: Mapping[int, str]
) -> dict[str, float]:
    if scores.ndim != 1 or scores.shape[0] != len(id_to_label):
        raise ValueError("score vector and label mapping have different sizes")
    return {
        str(id_to_label[index]).casefold(): float(scores[index])
        for index in sorted(id_to_label)
    }


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for row in rows:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in row["interventions"]:
            grouped[(item["family"], item["expected_relation"])].append(item)
        for family in {key[0] for key in grouped}:
            critical = grouped.get((family, "change"))
            preserving = grouped.get((family, "preserve"))
            if not critical or not preserving:
                continue
            pairs[family].append(
                (
                    float(np.mean([item["entailment_drop"] for item in critical])),
                    float(np.mean([item["entailment_drop"] for item in preserving])),
                    float(np.mean([item["entailment_delta"] for item in critical])),
                    float(np.mean([item["entailment_delta"] for item in preserving])),
                )
            )

    result: dict[str, Any] = {}
    for family, values in sorted(pairs.items()):
        array = np.asarray(values, dtype=np.float64)
        result[family] = {
            "orbit_count": int(array.shape[0]),
            "critical_drop_mean": float(array[:, 0].mean()),
            "preserve_drop_mean": float(array[:, 1].mean()),
            "critical_drop_greater_fraction": float(np.mean(array[:, 0] > array[:, 1])),
            "critical_delta_mean": float(array[:, 2].mean()),
            "preserve_delta_mean": float(array[:, 3].mean()),
        }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a frozen NLI cross-encoder baseline.")
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

    pairs: list[tuple[str, str]] = []
    for orbit in orbits:
        pairs.append((orbit.base_text, orbit.context_text))
        pairs.extend(
            (item.transformed_text, orbit.context_text)
            for item in orbit.interventions
        )

    from sentence_transformers import CrossEncoder

    started = perf_counter()
    model = CrossEncoder(
        args.model,
        revision=args.revision,
        device=args.device,
        trust_remote_code=False,
    )
    labels = {int(key): str(value) for key, value in model.model.config.id2label.items()}
    entailment_index = find_entailment_index(labels)
    scores = np.asarray(
        model.predict(
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=False,
            apply_softmax=True,
            convert_to_numpy=True,
        ),
        dtype=np.float64,
    )
    if scores.shape != (len(pairs), len(labels)):
        raise RuntimeError("NLI model returned an invalid score matrix")

    rows: list[dict[str, Any]] = []
    cursor = 0
    for orbit in orbits:
        base_probabilities = labelled_probabilities(scores[cursor], labels)
        base_state = relational_state_from_probabilities(base_probabilities)
        base_entailment = float(scores[cursor, entailment_index])
        interventions = []
        for index, item in enumerate(orbit.interventions):
            transformed_scores = scores[cursor + 1 + index]
            probabilities = labelled_probabilities(transformed_scores, labels)
            state = relational_state_from_probabilities(probabilities)
            entailment = float(transformed_scores[entailment_index])
            delta = entailment - base_entailment
            interventions.append(
                {
                    "family": item.family,
                    "expected_relation": item.expected_relation.value,
                    "generator_id": item.generator_id,
                    "entailment_probability": entailment,
                    "entailment_delta": delta,
                    "entailment_drop": -delta,
                    "probabilities": probabilities,
                    "probability_deltas": {
                        label: probability - base_probabilities[label]
                        for label, probability in probabilities.items()
                    },
                    "relational_state": state.to_dict(),
                    "relational_state_delta": state.delta_from(base_state),
                }
            )
        rows.append(
            {
                "base_id": orbit.base_id,
                "base_entailment_probability": base_entailment,
                "base_probabilities": base_probabilities,
                "base_relational_state": base_state.to_dict(),
                "interventions": interventions,
            }
        )
        cursor += 1 + len(orbit.interventions)

    temporary = output.with_name(f"{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(output)

    summary = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "orbit_count": len(orbits),
        "encoder": {
            "model_name": args.model,
            "revision": args.revision,
            "sentence_transformers": version("sentence-transformers"),
            "device": str(model.device),
            "batch_size": args.batch_size,
            "id_to_label": labels,
            "entailment_index": entailment_index,
        },
        "elapsed_seconds": perf_counter() - started,
        "pair_order": "premise=document,hypothesis=context",
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
