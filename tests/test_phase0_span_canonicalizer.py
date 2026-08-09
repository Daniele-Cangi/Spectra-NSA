from experiments.phase0_span_canonicalizer import _summarize


def test_span_summary_compares_critical_with_all_controls() -> None:
    rows = [
        {
            "interventions": [
                {
                    "family": "relation",
                    "expected_relation": "change",
                    "span_similarity_drop": 0.8,
                },
                {
                    "family": "relation",
                    "expected_relation": "preserve",
                    "span_similarity_drop": 0.1,
                },
                {
                    "family": "relation",
                    "expected_relation": "preserve",
                    "span_similarity_drop": 0.3,
                },
            ]
        }
    ]

    result = _summarize(rows)["relation"]

    assert result["critical_drop_mean"] == 0.8
    assert result["preserve_drop_mean"] == 0.2
    assert result["critical_greater_fraction"] == 1.0
