from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    recall_score,
    roc_auc_score,
)

from experiments.phase0_late_interaction import _load_orbits
from spectra_v3.semantic_variables import semantic_change_from_metadata


FRAME_CHANGE_THRESHOLD = 0.35
FRAME_AMBIGUITY_LOW = 0.08
FRAME_AMBIGUITY_HIGH = FRAME_CHANGE_THRESHOLD
FRAME_RELIABILITY_THRESHOLD = 0.75
NLI_CHANGE_THRESHOLD = 0.35


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"input contains no records: {path}")
    return rows


def _index_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in rows:
        for item in row["interventions"]:
            key = (str(row["base_id"]), str(item["generator_id"]))
            if key in result:
                raise ValueError(f"duplicate intervention observation: {key}")
            result[key] = item
    return result


def build_records(
    orbit_path: Path,
    frame_rows: Sequence[dict[str, Any]],
    nli_rows: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    frame_lookup = _index_rows(frame_rows)
    nli_lookup = _index_rows(nli_rows) if nli_rows is not None else {}
    records = []
    for orbit in _load_orbits(orbit_path):
        for item in orbit.interventions:
            annotation = semantic_change_from_metadata(item.metadata)
            annotation.validate_relation(item.expected_relation)
            key = (orbit.base_id, item.generator_id)
            if key not in frame_lookup:
                raise ValueError(f"missing frame observation: {key}")
            frame = frame_lookup[key]
            if str(frame["expected_relation"]) != item.expected_relation.value:
                raise ValueError(f"frame label mismatch: {key}")
            record = {
                "base_id": orbit.base_id,
                "generator_id": item.generator_id,
                "semantic_axis": annotation.axis.value,
                "role": str(item.metadata["adversarial_role"]),
                "template_id": str(orbit.metadata["template_id"]),
                "predicate_family": str(orbit.metadata["predicate_family"]),
                "evaluation_partition": str(
                    orbit.metadata["evaluation_partition"]
                ),
                "label": int(item.expected_relation.value == "change"),
                "selected_span_score": float(
                    frame["selected_span_similarity_drop"]
                ),
                "frame_score": float(frame["frame_distance"]),
                "observer_reliability": float(frame["observer_reliability"]),
            }
            if nli_rows is not None:
                if key not in nli_lookup:
                    raise ValueError(f"missing NLI observation: {key}")
                record["nli_score"] = max(
                    min(float(nli_lookup[key]["entailment_drop"]), 1.0), 0.0
                )
            records.append(record)
    if not records:
        raise ValueError("no frame records found")
    return records


def _binary_metrics(
    labels: np.ndarray, scores: np.ndarray, *, threshold: float
) -> dict[str, float]:
    predictions = scores >= threshold
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "false_positive_rate": float(
            np.mean(predictions[labels == 0]) if np.any(labels == 0) else 0.0
        ),
    }


def _matched_pair_accuracy(
    records: Sequence[dict[str, Any]], score_name: str
) -> float:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for record in records:
        if record["role"] in {"critical", "control"}:
            grouped[(record["base_id"], record["semantic_axis"])][
                record["role"]
            ] = float(record[score_name])
    if not grouped or any(set(pair) != {"critical", "control"} for pair in grouped.values()):
        raise ValueError("matched metric requires one critical-control pair per axis")
    return float(
        np.mean(
            [pair["critical"] > pair["control"] for pair in grouped.values()]
        )
    )


def _slice_metrics(
    records: Sequence[dict[str, Any]], score_name: str, *, threshold: float
) -> dict[str, Any]:
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    scores = np.asarray(
        [record[score_name] for record in records], dtype=np.float64
    )
    result = _binary_metrics(labels, scores, threshold=threshold)
    result["record_count"] = len(records)
    result["matched_pair_accuracy"] = _matched_pair_accuracy(records, score_name)
    return result


def _role_error_rates(
    records: Sequence[dict[str, Any]], score_name: str, *, threshold: float
) -> dict[str, float]:
    result = {}
    for role in ("critical", "control", "invariant"):
        selected = [record for record in records if record["role"] == role]
        predictions = [record[score_name] >= threshold for record in selected]
        if role == "critical":
            result["critical_recall"] = float(np.mean(predictions))
        else:
            result[f"{role}_false_positive_rate"] = float(np.mean(predictions))
    return result


def _evaluate_view(
    records: Sequence[dict[str, Any]], score_name: str, *, threshold: float
) -> dict[str, Any]:
    result = {
        "score_name": score_name,
        "threshold": threshold,
        "overall": _slice_metrics(records, score_name, threshold=threshold),
        "role_errors": _role_error_rates(
            records, score_name, threshold=threshold
        ),
    }
    for field in (
        "semantic_axis",
        "evaluation_partition",
        "predicate_family",
        "template_id",
    ):
        result[f"per_{field}"] = {
            value: _slice_metrics(
                [record for record in records if record[field] == value],
                score_name,
                threshold=threshold,
            )
            for value in sorted({record[field] for record in records})
        }
    return result


def _add_cascade_scores(records: Sequence[dict[str, Any]]) -> float:
    fallback_count = 0
    for record in records:
        frame_score = float(record["frame_score"])
        ambiguous = FRAME_AMBIGUITY_LOW < frame_score < FRAME_AMBIGUITY_HIGH
        unreliable = (
            float(record["observer_reliability"])
            < FRAME_RELIABILITY_THRESHOLD
        )
        fallback = ambiguous or unreliable
        record["fallback_used"] = fallback
        record["cascade_score"] = (
            float(record["nli_score"]) if fallback else frame_score
        )
        fallback_count += int(fallback)
    return fallback_count / len(records)


def _gate_report(result: dict[str, Any]) -> dict[str, Any]:
    span = result["views"]["selected_span"]
    preferred_name = "cascade" if "cascade" in result["views"] else "frame_fixed"
    preferred = result["views"][preferred_name]
    span_relation_auc = span["per_semantic_axis"]["relation"]["auroc"]
    preferred_relation_auc = preferred["per_semantic_axis"]["relation"]["auroc"]
    relation_gain = preferred_relation_auc - span_relation_auc
    remaining_error = 1.0 - span_relation_auc
    relation_error_reduction = (
        relation_gain / remaining_error if remaining_error > 0.0 else 0.0
    )
    held_partitions = (
        "held-predicate",
        "held-template",
        "double-holdout",
    )
    held_auc = min(
        preferred["per_evaluation_partition"][name]["auroc"]
        for name in held_partitions
    )
    nonrelation_axes = ("direction", "modality", "scope")
    nonrelation_margin = min(
        preferred["per_semantic_axis"][axis]["auroc"]
        - span["per_semantic_axis"][axis]["auroc"]
        for axis in nonrelation_axes
    )
    absolute_relation_check = relation_gain >= 0.05
    relative_relation_check = relation_error_reduction >= 0.50
    checks = {
        "relation_improvement_feasibility_corrected": (
            absolute_relation_check or relative_relation_check
        ),
        "held_partition_auroc_at_least_0_95": held_auc >= 0.95,
        "critical_recall_at_least_0_98": (
            preferred["role_errors"]["critical_recall"] >= 0.98
        ),
        "invariant_false_positive_rate_at_most_0_05": (
            preferred["role_errors"]["invariant_false_positive_rate"] <= 0.05
        ),
        "nonrelation_auroc_degradation_at_most_0_01": (
            nonrelation_margin >= -0.01
        ),
    }
    if preferred_name == "cascade":
        checks["nli_fallback_rate_at_most_0_30"] = (
            result["cascade"]["fallback_rate"] <= 0.30
        )
    return {
        "preferred_view": preferred_name,
        "passed": all(checks.values()),
        "gate_revision": (
            "v1.1: absolute +0.05 AUROC, or at least 50% reduction of the "
            "remaining AUROC error when +0.05 exceeds the available headroom"
        ),
        "superseded_absolute_check": {
            "relation_auroc_gain_at_least_0_05": absolute_relation_check,
            "feasible_from_baseline": remaining_error >= 0.05,
        },
        "checks": checks,
        "measurements": {
            "relation_auroc_gain": relation_gain,
            "relation_remaining_error_reduction": relation_error_reduction,
            "minimum_held_partition_auroc": held_auc,
            "minimum_nonrelation_auroc_margin": nonrelation_margin,
        },
    }


def _error_audit(
    records: Sequence[dict[str, Any]], score_name: str, *, threshold: float
) -> dict[str, Any]:
    fields = (
        "base_id",
        "generator_id",
        "semantic_axis",
        "role",
        "template_id",
        "predicate_family",
        "evaluation_partition",
        "label",
        "frame_score",
        "nli_score",
        "cascade_score",
        "fallback_used",
    )
    false_negatives = []
    false_positives = []
    for record in records:
        prediction = float(record[score_name]) >= threshold
        target = false_negatives if record["label"] and not prediction else None
        if not record["label"] and prediction:
            target = false_positives
        if target is not None:
            target.append(
                {
                    field: record[field]
                    for field in fields
                    if field in record
                }
            )
    return {
        "false_negative_count": len(false_negatives),
        "false_positive_count": len(false_positives),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
    }


def evaluate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    mutable_records = [dict(record) for record in records]
    result = {
        "schema_version": 1,
        "development_only": True,
        "claim_eligible": False,
        "record_count": len(records),
        "orbit_count": len({record["base_id"] for record in records}),
        "predeclared_thresholds": {
            "frame_change": FRAME_CHANGE_THRESHOLD,
            "frame_ambiguity_low": FRAME_AMBIGUITY_LOW,
            "frame_ambiguity_high": FRAME_AMBIGUITY_HIGH,
            "frame_reliability": FRAME_RELIABILITY_THRESHOLD,
            "nli_change": NLI_CHANGE_THRESHOLD,
        },
        "views": {
            "selected_span": _evaluate_view(
                mutable_records,
                "selected_span_score",
                threshold=FRAME_AMBIGUITY_LOW,
            ),
            "frame_fixed": _evaluate_view(
                mutable_records,
                "frame_score",
                threshold=FRAME_CHANGE_THRESHOLD,
            ),
        },
    }
    if all("nli_score" in record for record in mutable_records):
        result["views"]["nli"] = _evaluate_view(
            mutable_records, "nli_score", threshold=NLI_CHANGE_THRESHOLD
        )
        fallback_rate = _add_cascade_scores(mutable_records)
        result["views"]["cascade"] = _evaluate_view(
            mutable_records,
            "cascade_score",
            threshold=FRAME_CHANGE_THRESHOLD,
        )
        result["cascade"] = {
            "fallback_rate": fallback_rate,
            "fallback_count": int(round(fallback_rate * len(mutable_records))),
            "direct_count": int(round((1.0 - fallback_rate) * len(mutable_records))),
            "fallback_by_axis": {
                axis: int(
                    sum(
                        record["fallback_used"]
                        for record in mutable_records
                        if record["semantic_axis"] == axis
                    )
                )
                for axis in sorted(
                    {record["semantic_axis"] for record in mutable_records}
                )
            },
            "fallback_by_role": {
                role: int(
                    sum(
                        record["fallback_used"]
                        for record in mutable_records
                        if record["role"] == role
                    )
                )
                for role in sorted({record["role"] for record in mutable_records})
            },
        }
        result["error_audit"] = _error_audit(
            mutable_records,
            "cascade_score",
            threshold=FRAME_CHANGE_THRESHOLD,
        )
    result["decision_gate"] = _gate_report(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed frame variables and the optional NLI cascade."
    )
    parser.add_argument("--orbits", type=Path, required=True)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--nli", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = build_records(
        args.orbits,
        _load_jsonl(args.frame),
        _load_jsonl(args.nli) if args.nli else None,
    )
    result = evaluate(records)
    result["inputs"] = {
        "orbits": str(args.orbits.resolve()),
        "frame": str(args.frame.resolve()),
        "nli": str(args.nli.resolve()) if args.nli else None,
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
