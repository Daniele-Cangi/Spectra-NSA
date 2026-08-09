from experiments.phase0_variable_state import (
    CHEAP_FEATURE_NAMES,
    CANONICAL_DISTANCE_NAME,
    EXCLUDED_DESIGN_FIELDS,
    FEATURE_VIEWS,
    LEXICAL_FEATURE_NAMES,
    SPAN_FEATURE_NAMES,
    STATE_FEATURE_NAMES,
    build_records,
    evaluate,
)
from spectra_v3.semantic_variables import SemanticAxis, SemanticVariableChange


def _change(axis: SemanticAxis, relevant: bool) -> dict:
    return SemanticVariableChange(
        axis=axis,
        frame_id="shipment" if relevant else "report",
        query_relevant=relevant,
        value_changed=True,
        before_value="before",
        after_value="after",
    ).to_dict()


def test_build_records_excludes_oracle_design_fields() -> None:
    response = [
        {
            "base_id": "base-1",
            "metadata": {"template_id": "template-a"},
            "interventions": [
                {
                    "generator_id": "entity-critical",
                    "expected_relation": "change",
                    "response_norm": 0.4,
                    "abs_context_similarity_delta": 0.2,
                    "context_similarity_delta": -0.2,
                    "normalized_token_edit_distance": 0.1,
                    "metadata": {"semantic_change": _change(SemanticAxis.ENTITY, True)},
                }
            ],
        }
    ]
    late = [
        {
            "base_id": "base-1",
            "interventions": [
                {
                    "generator_id": "entity-critical",
                    "abs_maxsim_delta": 0.3,
                    "maxsim_delta": -0.3,
                }
            ],
        }
    ]
    deltas = {
        "support_delta": -1.0,
        "contradiction_delta": 0.8,
        "neutrality_delta": 0.0,
        "commitment_delta": 0.0,
        "entropy_delta": 0.1,
    }
    nli = [
        {
            "base_id": "base-1",
            "interventions": [
                {
                    "generator_id": "entity-critical",
                    "entailment_drop": 0.9,
                    "relational_state_delta": deltas,
                }
            ],
        }
    ]

    lexical_deltas = {
        "content_coverage_delta": -0.1,
        "entity_coverage_delta": -0.5,
        "numeric_coverage_delta": 0.0,
        "polarity_alignment_delta": 0.0,
        "scope_overlap_delta": -0.1,
    }
    lexical = [
        {
            "base_id": "base-1",
            "interventions": [
                {
                    "generator_id": "entity-critical",
                    "lexical_state_delta": lexical_deltas,
                    "lexical_state_distance": 0.5,
                }
            ],
        }
    ]

    span = [
        {
            "base_id": "base-1",
            "interventions": [
                {
                    "generator_id": "entity-critical",
                    "span_similarity_delta": -0.4,
                    "span_similarity_drop": 0.4,
                    "abs_span_similarity_delta": 0.4,
                }
            ],
        }
    ]

    records = build_records(response, late, nli, lexical, span)

    assert records[0]["label"] == 1
    assert records[0]["semantic_axis"] == "entity"
    assert not set(records[0]["features"]) & set(EXCLUDED_DESIGN_FIELDS)
    assert set(records[0]["features"]) == set(
        CHEAP_FEATURE_NAMES
        + LEXICAL_FEATURE_NAMES
        + SPAN_FEATURE_NAMES
        + STATE_FEATURE_NAMES
        + (CANONICAL_DISTANCE_NAME,)
    )


def test_evaluate_reports_transfer_and_jacobian_views() -> None:
    axes = ("entity", "polarity", "quantity", "time")
    templates = ("a", "b", "c", "d")
    records = []
    for base_index in range(16):
        for axis_index, axis in enumerate(axes):
            for label in (0, 1):
                features = {}
                for index, name in enumerate(CHEAP_FEATURE_NAMES):
                    features[name] = label * 2.0 + axis_index * 0.1 + index * 0.01
                for index, name in enumerate(LEXICAL_FEATURE_NAMES):
                    features[name] = label * 2.5 + axis_index * 0.1 + index * 0.01
                for index, name in enumerate(SPAN_FEATURE_NAMES):
                    features[name] = label * 2.7 + axis_index * 0.1 + index * 0.01
                features[CANONICAL_DISTANCE_NAME] = label * 3.0 + axis_index * 0.1
                for index, name in enumerate(STATE_FEATURE_NAMES):
                    features[name] = label * 3.0 + axis_index * 0.2 + index * 0.01
                records.append(
                    {
                        "base_id": f"base-{base_index}",
                        "template_id": templates[base_index % len(templates)],
                        "semantic_axis": axis,
                        "query_relevant": bool(label),
                        "label": label,
                        "features": features,
                        "nli_entailment_drop": float(label),
                    }
                )

    result = evaluate(records, seed=3)

    assert result["annotation_contract_accuracy"] == 1.0
    assert set(result["views"]) == set(FEATURE_VIEWS)
    assert result["views"]["combined"]["grouped_split"]["combined"]["auroc"] == 1.0
    assert result["semantic_jacobian"]["combined"]["mean_effective_rank"] > 0.0
