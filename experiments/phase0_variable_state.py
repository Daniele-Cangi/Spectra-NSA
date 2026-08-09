from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spectra_v3.interventions import ExpectedRelation
from spectra_v3.lexical_variables import LEXICAL_STATE_NAMES
from spectra_v3.semantic_variables import (
    RELATIONAL_STATE_NAMES,
    semantic_change_from_metadata,
)

CHEAP_FEATURE_NAMES = (
    "response_norm",
    "abs_context_similarity_delta",
    "context_similarity_delta",
    "abs_maxsim_delta",
    "maxsim_delta",
    "normalized_token_edit_distance",
)
STATE_FEATURE_NAMES = tuple(
    name
    for state_name in RELATIONAL_STATE_NAMES
    for name in (f"{state_name}_delta", f"abs_{state_name}_delta")
)
LEXICAL_FEATURE_NAMES = tuple(
    name
    for state_name in LEXICAL_STATE_NAMES
    for name in (f"lexical_{state_name}_delta", f"abs_lexical_{state_name}_delta")
) + ("lexical_state_distance",)
SPAN_FEATURE_NAMES = (
    "span_similarity_delta",
    "span_similarity_drop",
    "abs_span_similarity_delta",
)
CANONICAL_DISTANCE_NAME = "canonical_state_distance"
FEATURE_VIEWS = {
    "cheap": CHEAP_FEATURE_NAMES,
    "lexical_distance": ("lexical_state_distance",),
    "lexical_state": LEXICAL_FEATURE_NAMES,
    "span_state": SPAN_FEATURE_NAMES,
    "fixed_canonical_distance": (CANONICAL_DISTANCE_NAME,),
    "canonical_state": LEXICAL_FEATURE_NAMES + SPAN_FEATURE_NAMES,
    "relational_state": STATE_FEATURE_NAMES,
    "cheap_plus_lexical": CHEAP_FEATURE_NAMES + LEXICAL_FEATURE_NAMES,
    "cheap_plus_span": CHEAP_FEATURE_NAMES + SPAN_FEATURE_NAMES,
    "combined": CHEAP_FEATURE_NAMES + STATE_FEATURE_NAMES,
    "all": (
        CHEAP_FEATURE_NAMES
        + LEXICAL_FEATURE_NAMES
        + SPAN_FEATURE_NAMES
        + STATE_FEATURE_NAMES
    ),
}
EXCLUDED_DESIGN_FIELDS = (
    "expected_relation",
    "family",
    "query_relevant",
    "semantic_axis",
    "template_id",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _lookup(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["base_id"]), str(item["generator_id"])): item
        for row in rows
        for item in row["interventions"]
    }


def _state_features(item: Mapping[str, Any]) -> dict[str, float]:
    deltas = item.get("relational_state_delta")
    if not isinstance(deltas, Mapping):
        raise ValueError("NLI record has no relational_state_delta")
    result: dict[str, float] = {}
    for name in RELATIONAL_STATE_NAMES:
        delta_name = f"{name}_delta"
        delta = float(deltas[delta_name])
        result[delta_name] = delta
        result[f"abs_{name}_delta"] = abs(delta)
    return result


def _lexical_features(item: Mapping[str, Any]) -> dict[str, float]:
    deltas = item.get("lexical_state_delta")
    if not isinstance(deltas, Mapping):
        raise ValueError("lexical record has no lexical_state_delta")
    result: dict[str, float] = {}
    for name in LEXICAL_STATE_NAMES:
        delta = float(deltas[f"{name}_delta"])
        result[f"lexical_{name}_delta"] = delta
        result[f"abs_lexical_{name}_delta"] = abs(delta)
    result["lexical_state_distance"] = float(item["lexical_state_distance"])
    return result


def build_records(
    response_rows: Sequence[dict[str, Any]],
    late_rows: Sequence[dict[str, Any]],
    nli_rows: Sequence[dict[str, Any]],
    lexical_rows: Sequence[dict[str, Any]],
    span_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    late_lookup = _lookup(late_rows)
    nli_lookup = _lookup(nli_rows)
    lexical_lookup = _lookup(lexical_rows)
    span_lookup = _lookup(span_rows)
    records: list[dict[str, Any]] = []
    for row in response_rows:
        template_id = str(row.get("metadata", {}).get("template_id", ""))
        if not template_id:
            raise ValueError(f"orbit {row['base_id']} has no template_id")
        for item in row["interventions"]:
            annotation = semantic_change_from_metadata(item.get("metadata", {}))
            relation = str(item["expected_relation"])
            annotation.validate_relation(ExpectedRelation(relation))
            key = (str(row["base_id"]), str(item["generator_id"]))
            if (
                key not in late_lookup
                or key not in nli_lookup
                or key not in lexical_lookup
                or key not in span_lookup
            ):
                raise ValueError(f"missing joined response view: {key}")
            late = late_lookup[key]
            nli = nli_lookup[key]
            lexical = lexical_lookup[key]
            span = span_lookup[key]
            lexical_distance = float(lexical["lexical_state_distance"])
            span_drop = float(span["span_similarity_drop"])
            features = {
                "response_norm": float(item["response_norm"]),
                "abs_context_similarity_delta": float(
                    item["abs_context_similarity_delta"]
                ),
                "context_similarity_delta": float(item["context_similarity_delta"]),
                "abs_maxsim_delta": float(late["abs_maxsim_delta"]),
                "maxsim_delta": float(late["maxsim_delta"]),
                "normalized_token_edit_distance": float(
                    item["normalized_token_edit_distance"]
                ),
                **_lexical_features(lexical),
                "span_similarity_delta": float(span["span_similarity_delta"]),
                "span_similarity_drop": float(span["span_similarity_drop"]),
                "abs_span_similarity_delta": float(
                    span["abs_span_similarity_delta"]
                ),
                CANONICAL_DISTANCE_NAME: lexical_distance
                + 0.5 * max(span_drop, 0.0),
                **_state_features(nli),
            }
            records.append(
                {
                    "base_id": str(row["base_id"]),
                    "template_id": template_id,
                    "semantic_axis": annotation.axis.value,
                    "query_relevant": annotation.query_relevant,
                    "label": int(relation == "change"),
                    "features": features,
                    "nli_entailment_drop": float(nli["entailment_drop"]),
                }
            )
    if not records:
        raise ValueError("no semantic-variable records found")
    for names in FEATURE_VIEWS.values():
        if set(names) & set(EXCLUDED_DESIGN_FIELDS):
            raise RuntimeError("experimental design field leaked into model features")
    return records


def _matrix(
    records: Sequence[dict[str, Any]], names: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [[record["features"][name] for name in names] for record in records],
        dtype=np.float64,
    )


def _fit(matrix: np.ndarray, labels: np.ndarray):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=0),
    ).fit(matrix, labels)


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def _matched_pair_accuracy(
    records: Sequence[dict[str, Any]],
    indices: np.ndarray,
    scores: np.ndarray,
) -> float:
    pairs: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, score in zip(indices, scores, strict=True):
        record = records[int(index)]
        key = (record["base_id"], record["semantic_axis"])
        pairs[key][record["label"]].append(float(score))
    if not pairs or any(set(values) != {0, 1} for values in pairs.values()):
        raise ValueError("matched-pair metric requires one critical and one control record")
    return float(
        np.mean(
            [np.mean(values[1]) > np.mean(values[0]) for values in pairs.values()]
        )
    )


def _binary_view(
    records: Sequence[dict[str, Any]],
    names: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    matrix = _matrix(records, names)
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    groups = np.asarray([record["base_id"] for record in records])
    axes = np.asarray([record["semantic_axis"] for record in records])
    templates = np.asarray([record["template_id"] for record in records])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_indices, test_indices = next(splitter.split(matrix, labels, groups))
    model = _fit(matrix[train_indices], labels[train_indices])
    probabilities = model.predict_proba(matrix[test_indices])[:, 1]

    combined_metrics = _binary_metrics(labels[test_indices], probabilities)
    combined_metrics["matched_pair_accuracy"] = _matched_pair_accuracy(
        records, test_indices, probabilities
    )
    per_axis = {}
    for axis in sorted(set(axes[test_indices])):
        mask = axes[test_indices] == axis
        metrics = _binary_metrics(
            labels[test_indices][mask], probabilities[mask]
        )
        metrics["matched_pair_accuracy"] = _matched_pair_accuracy(
            records, test_indices[mask], probabilities[mask]
        )
        per_axis[axis] = metrics

    leave_one_axis_out = {}
    for axis in sorted(set(axes)):
        train_mask = axes != axis
        test_mask = axes == axis
        held_model = _fit(matrix[train_mask], labels[train_mask])
        held_probabilities = held_model.predict_proba(matrix[test_mask])[:, 1]
        metrics = _binary_metrics(
            labels[test_mask], held_probabilities
        )
        metrics["matched_pair_accuracy"] = _matched_pair_accuracy(
            records, np.flatnonzero(test_mask), held_probabilities
        )
        leave_one_axis_out[axis] = metrics

    leave_one_template_out = {}
    for template in sorted(set(templates)):
        train_mask = templates != template
        test_mask = templates == template
        held_model = _fit(matrix[train_mask], labels[train_mask])
        held_probabilities = held_model.predict_proba(matrix[test_mask])[:, 1]
        metrics = _binary_metrics(
            labels[test_mask], held_probabilities
        )
        metrics["matched_pair_accuracy"] = _matched_pair_accuracy(
            records, np.flatnonzero(test_mask), held_probabilities
        )
        leave_one_template_out[template] = metrics

    return {
        "feature_names": list(names),
        "grouped_split": {
            "train_orbits": len(set(groups[train_indices])),
            "test_orbits": len(set(groups[test_indices])),
            "combined": combined_metrics,
            "per_axis": per_axis,
        },
        "leave_one_axis_out": leave_one_axis_out,
        "leave_one_template_out": leave_one_template_out,
    }


def _axis_decoding(
    records: Sequence[dict[str, Any]], names: Sequence[str], *, seed: int
) -> dict[str, Any]:
    critical = [record for record in records if record["label"] == 1]
    matrix = _matrix(critical, names)
    labels = np.asarray([record["semantic_axis"] for record in critical])
    groups = np.asarray([record["base_id"] for record in critical])
    templates = np.asarray([record["template_id"] for record in critical])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_indices, test_indices = next(splitter.split(matrix, labels, groups))
    model = _fit(matrix[train_indices], labels[train_indices])
    prediction = model.predict(matrix[test_indices])

    held_templates = {}
    for template in sorted(set(templates)):
        train_mask = templates != template
        test_mask = templates == template
        held_model = _fit(matrix[train_mask], labels[train_mask])
        held_prediction = held_model.predict(matrix[test_mask])
        held_templates[template] = {
            "accuracy": float(accuracy_score(labels[test_mask], held_prediction)),
            "macro_f1": float(
                f1_score(labels[test_mask], held_prediction, average="macro")
            ),
        }
    return {
        "grouped_split": {
            "accuracy": float(accuracy_score(labels[test_indices], prediction)),
            "macro_f1": float(f1_score(labels[test_indices], prediction, average="macro")),
        },
        "leave_one_template_out": held_templates,
    }


def _effective_rank(singular_values: np.ndarray) -> float:
    total = float(singular_values.sum())
    if total <= 1e-12:
        return 0.0
    probabilities = singular_values / total
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return float(np.exp(entropy))


def _semantic_jacobian(
    records: Sequence[dict[str, Any]], names: Sequence[str]
) -> dict[str, float]:
    raw = _matrix(records, names)
    scale = raw.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (raw - raw.mean(axis=0)) / scale
    grouped: dict[str, dict[str, dict[int, list[np.ndarray]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for record, vector in zip(records, standardized, strict=True):
        grouped[record["base_id"]][record["semantic_axis"]][record["label"]].append(
            vector
        )

    ranks: list[float] = []
    alignments: list[float] = []
    for axes in grouped.values():
        rows = []
        for axis in sorted(axes):
            if set(axes[axis]) != {0, 1}:
                raise ValueError(f"axis {axis} has no critical-control pair")
            rows.append(
                np.mean(axes[axis][1], axis=0) - np.mean(axes[axis][0], axis=0)
            )
        matrix = np.asarray(rows, dtype=np.float64)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        ranks.append(_effective_rank(singular_values))
        normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        cosine = np.abs(normalized @ normalized.T)
        upper = cosine[np.triu_indices(cosine.shape[0], k=1)]
        alignments.append(float(upper.mean()))
    return {
        "mean_effective_rank": float(np.mean(ranks)),
        "mean_absolute_cross_axis_cosine": float(np.mean(alignments)),
    }


def evaluate(records: Sequence[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    annotation_labels = np.asarray(
        [int(record["query_relevant"]) for record in records], dtype=np.int64
    )
    nli_drop = np.asarray(
        [record["nli_entailment_drop"] for record in records], dtype=np.float64
    )
    views = {
        view: _binary_view(records, names, seed=seed)
        for view, names in FEATURE_VIEWS.items()
    }
    nli_metrics = _binary_metrics(labels, nli_drop)
    nli_metrics["matched_pair_accuracy"] = _matched_pair_accuracy(
        records, np.arange(len(records)), nli_drop
    )
    return {
        "schema_version": 1,
        "development_only": True,
        "claim_eligible": False,
        "record_count": len(records),
        "orbit_count": len({record["base_id"] for record in records}),
        "feature_leakage_audit": {
            "excluded_design_fields": list(EXCLUDED_DESIGN_FIELDS),
            "passed": True,
        },
        "annotation_contract_accuracy": float(
            accuracy_score(labels, annotation_labels)
        ),
        "nli_entailment_drop_baseline": nli_metrics,
        "views": views,
        "axis_decoding": {
            view: _axis_decoding(records, names, seed=seed)
            for view, names in FEATURE_VIEWS.items()
        },
        "semantic_jacobian": {
            view: _semantic_jacobian(records, names)
            for view, names in FEATURE_VIEWS.items()
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate task-relative semantic-variable response views."
    )
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--late", type=Path, required=True)
    parser.add_argument("--nli", type=Path, required=True)
    parser.add_argument("--lexical", type=Path, required=True)
    parser.add_argument("--span", type=Path, required=True)
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
    records = build_records(
        _load_jsonl(args.response),
        _load_jsonl(args.late),
        _load_jsonl(args.nli),
        _load_jsonl(args.lexical),
        _load_jsonl(args.span),
    )
    result = evaluate(records, seed=args.seed)
    result["inputs"] = {
        "response": str(args.response.resolve()),
        "late": str(args.late.resolve()),
        "nli": str(args.nli.resolve()),
        "lexical": str(args.lexical.resolve()),
        "span": str(args.span.resolve()),
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
