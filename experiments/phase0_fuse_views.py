from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = (
    "response_norm",
    "abs_context_similarity_delta",
    "context_similarity_delta",
    "abs_maxsim_delta",
    "maxsim_delta",
    "normalized_token_edit_distance",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_records(
    response_rows: Sequence[dict[str, Any]], late_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    late_lookup = {
        (row["base_id"], item["generator_id"]): item
        for row in late_rows
        for item in row["interventions"]
    }
    records: list[dict[str, Any]] = []
    matched_families = {"entity", "negation", "quantity", "time_modality"}
    for row in response_rows:
        for item in row["interventions"]:
            if item["family"] not in matched_families:
                continue
            key = (row["base_id"], item["generator_id"])
            if key not in late_lookup:
                raise ValueError(f"missing late-interaction record: {key}")
            late = late_lookup[key]
            records.append(
                {
                    "base_id": row["base_id"],
                    "family": item["family"],
                    "label": int(item["expected_relation"] == "change"),
                    "features": [
                        float(item["response_norm"]),
                        float(item["abs_context_similarity_delta"]),
                        float(item["context_similarity_delta"]),
                        float(late["abs_maxsim_delta"]),
                        float(late["maxsim_delta"]),
                        float(item["normalized_token_edit_distance"]),
                    ],
                }
            )
    if not records:
        raise ValueError("no matched intervention records found")
    return records


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def _fit(train_x: np.ndarray, train_y: np.ndarray):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=0),
    )
    return model.fit(train_x, train_y)


def evaluate(records: Sequence[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    matrix = np.asarray([record["features"] for record in records], dtype=np.float64)
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    groups = np.asarray([record["base_id"] for record in records])
    families = np.asarray([record["family"] for record in records])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_indices, test_indices = next(splitter.split(matrix, labels, groups))
    model = _fit(matrix[train_indices], labels[train_indices])
    probabilities = model.predict_proba(matrix[test_indices])[:, 1]

    baselines: dict[str, Any] = {}
    for index, name in enumerate(FEATURE_NAMES):
        values = matrix[test_indices, index]
        auc = float(roc_auc_score(labels[test_indices], values))
        baselines[name] = {
            "raw_auroc": auc,
            "orientation_free_auroc": max(auc, 1.0 - auc),
        }

    per_family: dict[str, Any] = {}
    for family in sorted(set(families[test_indices])):
        mask = families[test_indices] == family
        per_family[family] = _metrics(labels[test_indices][mask], probabilities[mask])

    leave_one_family_out: dict[str, Any] = {}
    for family in sorted(set(families)):
        train_mask = families != family
        test_mask = families == family
        held_model = _fit(matrix[train_mask], labels[train_mask])
        held_probabilities = held_model.predict_proba(matrix[test_mask])[:, 1]
        leave_one_family_out[family] = _metrics(
            labels[test_mask], held_probabilities
        )

    classifier = model.named_steps["logisticregression"]
    return {
        "schema_version": 1,
        "development_only": True,
        "claim_eligible": False,
        "record_count": len(records),
        "orbit_count": len(set(groups)),
        "feature_names": list(FEATURE_NAMES),
        "grouped_split": {
            "seed": seed,
            "train_orbits": len(set(groups[train_indices])),
            "test_orbits": len(set(groups[test_indices])),
            "combined": _metrics(labels[test_indices], probabilities),
            "per_family": per_family,
            "standardized_coefficients": {
                name: float(value)
                for name, value in zip(
                    FEATURE_NAMES, classifier.coef_[0], strict=True
                )
            },
        },
        "single_feature_baselines": baselines,
        "leave_one_family_out": leave_one_family_out,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse Phase 0 response views.")
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--late", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = build_records(_load_jsonl(args.response), _load_jsonl(args.late))
    result = evaluate(records, seed=args.seed)
    result["response_input"] = str(args.response.resolve())
    result["late_input"] = str(args.late.resolve())
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
