from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt


FloatVector = npt.NDArray[np.float64]


def _arrays(
    feature_rows: Sequence[Mapping[str, float]], names: Sequence[str]
) -> FloatVector:
    matrix = np.asarray(
        [[float(row[name]) for name in names] for row in feature_rows],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("invalid feature matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix contains non-finite values")
    return matrix


def best_f1_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    from sklearn.metrics import f1_score

    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0], probability, [1.0])))
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        value = float(f1_score(y, probability >= threshold, zero_division=0))
        candidate = (value, -abs(float(threshold) - 0.5))
        if candidate > (best[0], -abs(best[1] - 0.5)):
            best = (value, float(threshold))
    return best[1]


def calibration_error(
    labels: Sequence[int], scores: Sequence[float], bins: int = 10
) -> float:
    y = np.asarray(labels, dtype=np.float64)
    probability = np.asarray(scores, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if np.any(mask):
            result += float(mask.mean()) * abs(
                float(probability[mask].mean()) - float(y[mask].mean())
            )
    return result


def risk_coverage(
    labels: Sequence[int], scores: Sequence[float]
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(scores, dtype=np.float64)
    prediction = probability >= 0.5
    error = (prediction != y).astype(np.float64)
    confidence = np.abs(probability - 0.5)
    order = np.argsort(-confidence, kind="stable")
    cumulative_risk = np.cumsum(error[order]) / np.arange(1, len(y) + 1)
    coverage = np.arange(1, len(y) + 1) / len(y)
    aurc = float(np.trapezoid(cumulative_risk, coverage))
    failure = error.sum()
    recalls = {}
    uncertainty_order = np.argsort(confidence, kind="stable")
    for budget in (0.05, 0.10, 0.20, 0.40):
        count = max(1, int(round(len(y) * budget)))
        found = float(error[uncertainty_order[:count]].sum())
        recalls[f"failure_recall_at_{int(budget * 100)}pct"] = (
            found / float(failure) if failure else 0.0
        )
    return {"aurc": aurc, **recalls}


def binary_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        roc_auc_score,
    )

    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y)) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y, probability))
        auprc = float(average_precision_score(y, probability))
    prediction = probability >= threshold
    return {
        "n": float(len(y)),
        "positive_rate": float(y.mean()),
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": float(accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "threshold": float(threshold),
        "ece_10": calibration_error(y, probability),
        **risk_coverage(y, probability),
    }


@dataclass
class FittedScores:
    feature_names: list[str]
    dev: FloatVector
    test: FloatVector
    threshold: float


def fit_logistic(
    train_rows: Sequence[Mapping[str, float]],
    train_labels: Sequence[int],
    dev_rows: Sequence[Mapping[str, float]],
    dev_labels: Sequence[int],
    test_rows: Sequence[Mapping[str, float]],
    feature_names: Sequence[str],
) -> FittedScores:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=1701,
            solver="lbfgs",
        ),
    )
    model.fit(_arrays(train_rows, feature_names), np.asarray(train_labels))
    dev_score = model.predict_proba(_arrays(dev_rows, feature_names))[:, 1]
    test_score = model.predict_proba(_arrays(test_rows, feature_names))[:, 1]
    threshold = best_f1_threshold(dev_labels, dev_score)
    return FittedScores(
        feature_names=list(feature_names),
        dev=np.asarray(dev_score, dtype=np.float64),
        test=np.asarray(test_score, dtype=np.float64),
        threshold=threshold,
    )


def grouped_bootstrap_delta(
    labels: Sequence[int],
    baseline: Sequence[float],
    candidate: Sequence[float],
    groups: Sequence[str],
    *,
    metric: str = "auroc",
    replicates: int = 1000,
) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(labels, dtype=np.int64)
    base = np.asarray(baseline, dtype=np.float64)
    trial = np.asarray(candidate, dtype=np.float64)
    grouped: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        grouped.setdefault(group, []).append(index)
    group_names = sorted(grouped)
    seed_material = "\x1f".join(group_names).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    random = np.random.default_rng(seed)
    scorer = roc_auc_score if metric == "auroc" else average_precision_score
    deltas = []
    for _ in range(replicates):
        sampled = random.choice(group_names, size=len(group_names), replace=True)
        indices = [item for group in sampled for item in grouped[str(group)]]
        sampled_y = y[indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        deltas.append(
            float(scorer(sampled_y, trial[indices]))
            - float(scorer(sampled_y, base[indices]))
        )
    if not deltas:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    values = np.asarray(deltas)
    observed = float(scorer(y, trial) - scorer(y, base))
    return {
        "delta": observed,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "replicates": float(len(values)),
    }


def similarity_slices(
    dev_cosine: Sequence[float], test_cosine: Sequence[float]
) -> dict[str, npt.NDArray[np.bool_]]:
    dev = np.asarray(dev_cosine, dtype=np.float64)
    test = np.asarray(test_cosine, dtype=np.float64)
    quartiles = np.quantile(dev, [0.25, 0.5, 0.75])
    return {
        "high_similarity": test >= quartiles[2],
        "similarity_q1": test < quartiles[0],
        "similarity_q2": (test >= quartiles[0]) & (test < quartiles[1]),
        "similarity_q3": (test >= quartiles[1]) & (test < quartiles[2]),
        "similarity_q4": test >= quartiles[2],
    }


def selective_cascade(
    dev_labels: Sequence[int],
    dev_cheap: Sequence[float],
    dev_nli: Sequence[float],
    test_labels: Sequence[int],
    test_cheap: Sequence[float],
    test_nli: Sequence[float],
    test_base: Sequence[float],
    *,
    max_call_fraction: float = 0.40,
) -> dict[str, Any]:
    del dev_nli  # NLI labels/scores never determine the routing threshold.
    dev_uncertainty = 0.5 - np.abs(np.asarray(dev_cheap) - 0.5)
    route_threshold = float(np.quantile(dev_uncertainty, 1.0 - max_call_fraction))
    test_cheap_array = np.asarray(test_cheap, dtype=np.float64)
    test_nli_array = np.asarray(test_nli, dtype=np.float64)
    uncertainty = 0.5 - np.abs(test_cheap_array - 0.5)
    route = uncertainty >= route_threshold
    cascade = np.where(route, test_nli_array, test_cheap_array)

    count = int(round(len(route) * max_call_fraction))
    random_rank = np.asarray(
        [
            int.from_bytes(hashlib.sha256(str(index).encode()).digest()[:8], "big")
            for index in range(len(route))
        ]
    )
    random_route = np.zeros(len(route), dtype=bool)
    random_route[np.argsort(random_rank)[:count]] = True
    random_score = np.where(random_route, test_nli_array, test_cheap_array)

    base_array = np.asarray(test_base, dtype=np.float64)
    base_route = np.zeros(len(route), dtype=bool)
    base_route[np.argsort(-base_array)[:count]] = True
    base_score = np.where(base_route, test_nli_array, test_cheap_array)
    threshold = best_f1_threshold(dev_labels, dev_cheap)
    return {
        "route_threshold": route_threshold,
        "call_fraction": float(route.mean()),
        "routed": int(route.sum()),
        "score": cascade,
        "metrics": binary_metrics(test_labels, cascade, threshold),
        "random_metrics": binary_metrics(test_labels, random_score, threshold),
        "base_similarity_metrics": binary_metrics(
            test_labels, base_score, threshold
        ),
        "route_mask": route,
    }
