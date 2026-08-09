import pytest

from experiments.phase0_summarize import describe, summarize_feature_rows


def test_describe_reports_quantiles_and_zero_fraction() -> None:
    summary = describe([0.0, 1.0, 2.0])

    assert summary["count"] == 3
    assert summary["median"] == pytest.approx(1.0)
    assert summary["zero_fraction"] == pytest.approx(1 / 3)


def test_summary_compares_matched_family_relations() -> None:
    row = {
        "features": {
            "response.global.numerical_rank": 2.0,
            "response.global.total_energy": 0.5,
        },
        "interventions": [
            {
                "family": "negation",
                "expected_relation": "change",
                "response_norm": 0.8,
                "normalized_token_edit_distance": 0.1,
                "abs_context_similarity_delta": 0.7,
                "context_similarity_delta": -0.7,
                "generator_id": "critical",
            },
            {
                "family": "negation",
                "expected_relation": "preserve",
                "response_norm": 0.2,
                "normalized_token_edit_distance": 0.3,
                "abs_context_similarity_delta": 0.1,
                "context_similarity_delta": -0.1,
                "generator_id": "control",
            },
        ],
    }

    summary = summarize_feature_rows([row])

    diagnostic = summary["matched_family_diagnostics"]["negation"]
    assert diagnostic["mean_gap"] == pytest.approx(0.6)
    assert diagnostic["critical_greater_fraction"] == pytest.approx(1.0)
    assert diagnostic["critical_token_edit_mean"] == pytest.approx(0.1)
    assert diagnostic["preserve_token_edit_mean"] == pytest.approx(0.3)
    assert diagnostic["context_critical_greater_fraction"] == pytest.approx(1.0)
    assert diagnostic["critical_more_negative_fraction"] == pytest.approx(1.0)
