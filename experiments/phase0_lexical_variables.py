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
from spectra_v3.lexical_variables import lexical_state_distance, lexical_variable_state


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for item in row["interventions"]:
            grouped[(item["family"], item["expected_relation"])].append(
                float(item["lexical_state_distance"])
            )
        for family in {key[0] for key in grouped}:
            critical = grouped.get((family, "change"))
            preserving = grouped.get((family, "preserve"))
            if critical and preserving:
                pairs[family].append((float(np.mean(critical)), float(np.mean(preserving))))
    return {
        family: {
            "orbit_count": len(values),
            "critical_distance_mean": float(np.mean([value[0] for value in values])),
            "preserve_distance_mean": float(np.mean([value[1] for value in values])),
            "critical_greater_fraction": float(
                np.mean([value[0] > value[1] for value in values])
            ),
        }
        for family, values in sorted(pairs.items())
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure a family-blind lexical semantic-variable state."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    output = args.output.resolve()
    summary_path = output.with_name(f"{output.name}.summary.json")
    existing = [path for path in (output, summary_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite: {', '.join(map(str, existing))}")
    output.parent.mkdir(parents=True, exist_ok=True)
    orbits = _load_orbits(args.input)
    started = perf_counter()
    rows = []
    for orbit in orbits:
        base_state = lexical_variable_state(orbit.context_text, orbit.base_text)
        interventions = []
        for item in orbit.interventions:
            state = lexical_variable_state(orbit.context_text, item.transformed_text)
            interventions.append(
                {
                    "family": item.family,
                    "expected_relation": item.expected_relation.value,
                    "generator_id": item.generator_id,
                    "lexical_state": state.to_dict(),
                    "lexical_state_delta": state.delta_from(base_state),
                    "lexical_state_distance": lexical_state_distance(base_state, state),
                }
            )
        rows.append(
            {
                "base_id": orbit.base_id,
                "base_lexical_state": base_state.to_dict(),
                "interventions": interventions,
            }
        )

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
        "observer": "family-blind lexical variable state",
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
