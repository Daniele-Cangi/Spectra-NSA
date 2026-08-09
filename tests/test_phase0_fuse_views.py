from experiments.phase0_fuse_views import FEATURE_NAMES, build_records


def test_build_records_joins_views_by_generator() -> None:
    response = [
        {
            "base_id": "base-1",
            "interventions": [
                {
                    "family": "negation",
                    "generator_id": "neg-v1",
                    "expected_relation": "change",
                    "response_norm": 0.5,
                    "abs_context_similarity_delta": 0.2,
                    "context_similarity_delta": -0.2,
                    "normalized_token_edit_distance": 0.1,
                }
            ],
        }
    ]
    late = [
        {
            "base_id": "base-1",
            "interventions": [
                {
                    "generator_id": "neg-v1",
                    "abs_maxsim_delta": 0.3,
                    "maxsim_delta": -0.3,
                }
            ],
        }
    ]

    records = build_records(response, late)

    assert records[0]["label"] == 1
    assert len(records[0]["features"]) == len(FEATURE_NAMES)
