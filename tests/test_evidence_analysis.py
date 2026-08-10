from __future__ import annotations

import numpy as np

from spectra_v3.evidence_analysis import (
    binary_metrics,
    grouped_bootstrap_delta,
    selective_cascade,
)


def test_grouped_bootstrap_uses_group_units() -> None:
    labels = [0, 0, 1, 1, 0, 1]
    baseline = [0.4, 0.3, 0.6, 0.7, 0.45, 0.55]
    candidate = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7]
    groups = ["a", "a", "b", "b", "c", "c"]
    result = grouped_bootstrap_delta(
        labels, baseline, candidate, groups, replicates=100
    )
    assert result["delta"] >= 0.0
    assert result["replicates"] > 0


def test_selective_cascade_respects_frozen_budget() -> None:
    labels = np.asarray([0, 1] * 50)
    cheap = np.linspace(0.01, 0.99, 100)
    nli = labels * 0.98 + (1 - labels) * 0.02
    result = selective_cascade(
        labels,
        cheap,
        nli,
        labels,
        cheap,
        nli,
        cheap,
        max_call_fraction=0.40,
    )
    assert result["call_fraction"] <= 0.41
    assert len(result["score"]) == len(labels)


def test_binary_metrics_include_cost_relevant_risk_outputs() -> None:
    result = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], 0.5)
    assert result["auroc"] == 1.0
    assert "aurc" in result
    assert "failure_recall_at_40pct" in result
