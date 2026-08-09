from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def describe(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    if not np.isfinite(array).all():
        raise ValueError("summary input contains non-finite values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "q10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
        "zero_fraction": float(np.mean(array <= 1e-12)),
    }


def _pearson(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    right = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if left.std() <= 1e-12 or right.std() <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def summarize_feature_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("feature rows cannot be empty")

    orbit_metrics: dict[str, list[float]] = defaultdict(list)
    group_norms: dict[str, list[float]] = defaultdict(list)
    group_token_edits: dict[str, list[float]] = defaultdict(list)
    group_context_deltas: dict[str, list[float]] = defaultdict(list)
    group_signed_context_deltas: dict[str, list[float]] = defaultdict(list)
    response_token_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    generator_norms: dict[str, list[float]] = defaultdict(list)
    family_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    family_token_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    family_context_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    family_signed_context_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    rank_counts: Counter[int] = Counter()
    intervention_count = 0

    selected_metrics = (
        "response.global.total_energy",
        "response.global.effective_rank",
        "response.global.spectral_entropy",
        "response.critical_invariant_energy_contrast",
        "response.critical_to_invariant_log_energy_ratio",
    )
    for row in rows:
        features = row["features"]
        for name in selected_metrics:
            if name in features:
                orbit_metrics[name].append(float(features[name]))
        rank_counts[int(features["response.global.numerical_rank"])] += 1

        by_family_relation: dict[tuple[str, str], list[float]] = defaultdict(list)
        by_family_relation_tokens: dict[tuple[str, str], list[float]] = defaultdict(list)
        by_family_relation_context: dict[tuple[str, str], list[float]] = defaultdict(list)
        by_family_relation_signed_context: dict[tuple[str, str], list[float]] = defaultdict(
            list
        )
        for intervention in row["interventions"]:
            family = str(intervention["family"])
            relation = str(intervention["expected_relation"])
            norm = float(intervention["response_norm"])
            group_norms[f"{family}.{relation}"].append(norm)
            generator_norms[str(intervention["generator_id"])].append(norm)
            by_family_relation[(family, relation)].append(norm)
            if "normalized_token_edit_distance" in intervention:
                token_edit = float(intervention["normalized_token_edit_distance"])
                group_token_edits[f"{family}.{relation}"].append(token_edit)
                response_token_pairs[f"{family}.{relation}"].append((norm, token_edit))
                response_token_pairs["all"].append((norm, token_edit))
                by_family_relation_tokens[(family, relation)].append(token_edit)
            if "abs_context_similarity_delta" in intervention:
                context_delta = float(intervention["abs_context_similarity_delta"])
                group_context_deltas[f"{family}.{relation}"].append(context_delta)
                by_family_relation_context[(family, relation)].append(context_delta)
            if "context_similarity_delta" in intervention:
                signed_delta = float(intervention["context_similarity_delta"])
                group_signed_context_deltas[f"{family}.{relation}"].append(
                    signed_delta
                )
                by_family_relation_signed_context[(family, relation)].append(
                    signed_delta
                )
            intervention_count += 1

        for family in {key[0] for key in by_family_relation}:
            preserving = by_family_relation.get((family, "preserve"))
            changing = by_family_relation.get((family, "change"))
            if preserving and changing:
                family_pairs[family].append(
                    (float(np.mean(changing)), float(np.mean(preserving)))
                )
                preserving_tokens = by_family_relation_tokens.get((family, "preserve"))
                changing_tokens = by_family_relation_tokens.get((family, "change"))
                if preserving_tokens and changing_tokens:
                    family_token_pairs[family].append(
                        (
                            float(np.mean(changing_tokens)),
                            float(np.mean(preserving_tokens)),
                        )
                    )
                preserving_context = by_family_relation_context.get((family, "preserve"))
                changing_context = by_family_relation_context.get((family, "change"))
                if preserving_context and changing_context:
                    family_context_pairs[family].append(
                        (
                            float(np.mean(changing_context)),
                            float(np.mean(preserving_context)),
                        )
                    )
                preserving_signed = by_family_relation_signed_context.get(
                    (family, "preserve")
                )
                changing_signed = by_family_relation_signed_context.get(
                    (family, "change")
                )
                if preserving_signed and changing_signed:
                    family_signed_context_pairs[family].append(
                        (
                            float(np.mean(changing_signed)),
                            float(np.mean(preserving_signed)),
                        )
                    )

    paired_summary: dict[str, Any] = {}
    for family, pairs in sorted(family_pairs.items()):
        critical = np.asarray([pair[0] for pair in pairs])
        preserving = np.asarray([pair[1] for pair in pairs])
        gaps = critical - preserving
        paired_summary[family] = {
            "orbit_count": len(pairs),
            "critical_mean": float(critical.mean()),
            "preserve_mean": float(preserving.mean()),
            "mean_gap": float(gaps.mean()),
            "critical_greater_fraction": float(np.mean(gaps > 0.0)),
        }
        token_pairs = family_token_pairs.get(family)
        if token_pairs:
            paired_summary[family].update(
                {
                    "critical_token_edit_mean": float(
                        np.mean([pair[0] for pair in token_pairs])
                    ),
                    "preserve_token_edit_mean": float(
                        np.mean([pair[1] for pair in token_pairs])
                    ),
                }
            )
        context_pairs = family_context_pairs.get(family)
        if context_pairs:
            critical_context = np.asarray([pair[0] for pair in context_pairs])
            preserving_context = np.asarray([pair[1] for pair in context_pairs])
            context_gaps = critical_context - preserving_context
            paired_summary[family].update(
                {
                    "critical_context_delta_mean": float(critical_context.mean()),
                    "preserve_context_delta_mean": float(preserving_context.mean()),
                    "context_critical_greater_fraction": float(
                        np.mean(context_gaps > 0.0)
                    ),
                }
            )
        signed_pairs = family_signed_context_pairs.get(family)
        if signed_pairs:
            critical_signed = np.asarray([pair[0] for pair in signed_pairs])
            preserving_signed = np.asarray([pair[1] for pair in signed_pairs])
            paired_summary[family].update(
                {
                    "critical_similarity_delta_mean": float(critical_signed.mean()),
                    "preserve_similarity_delta_mean": float(preserving_signed.mean()),
                    "critical_more_negative_fraction": float(
                        np.mean(critical_signed < preserving_signed)
                    ),
                }
            )

    return {
        "orbit_count": len(rows),
        "intervention_count": intervention_count,
        "rank_counts": {str(key): value for key, value in sorted(rank_counts.items())},
        "orbit_metrics": {
            name: describe(values) for name, values in sorted(orbit_metrics.items())
        },
        "family_relation_response_norms": {
            name: describe(values) for name, values in sorted(group_norms.items())
        },
        "family_relation_token_edits": {
            name: describe(values) for name, values in sorted(group_token_edits.items())
        },
        "family_relation_context_similarity_deltas": {
            name: describe(values) for name, values in sorted(group_context_deltas.items())
        },
        "family_relation_signed_context_similarity_deltas": {
            name: describe(values)
            for name, values in sorted(group_signed_context_deltas.items())
        },
        "response_token_edit_pearson": {
            name: _pearson(values) for name, values in sorted(response_token_pairs.items())
        },
        "generator_response_norms": {
            name: describe(values) for name, values in sorted(generator_norms.items())
        },
        "matched_family_diagnostics": paired_summary,
    }


def _load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Phase 0 feature JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(input_: Path, output: Path, *, overwrite: bool) -> Path:
    if not input_.is_file():
        raise FileNotFoundError(f"input does not exist: {input_}")
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_feature_rows(_load(input_))
    summary.update(
        {
            "schema_version": 1,
            "input": str(input_.resolve()),
            "input_sha256": hashlib.sha256(input_.read_bytes()).hexdigest(),
        }
    )
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run(args.input, args.output, overwrite=args.overwrite)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
