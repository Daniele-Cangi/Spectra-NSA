from experiments.phase0_frame_evaluate import evaluate


def _records(with_nli: bool) -> list[dict]:
    records = []
    axes = ("relation", "direction", "scope", "modality")
    partitions = (
        "development",
        "held-predicate",
        "held-template",
        "double-holdout",
    )
    for orbit_index in range(8):
        for axis in axes:
            for role, label in (("critical", 1), ("control", 0), ("invariant", 0)):
                frame_score = 0.2 if axis == "relation" and label else float(label)
                record = {
                    "base_id": f"base-{orbit_index}",
                    "semantic_axis": axis,
                    "role": role,
                    "template_id": f"template-{orbit_index % 4}",
                    "predicate_family": f"predicate-{orbit_index % 4}",
                    "evaluation_partition": partitions[orbit_index % 4],
                    "label": label,
                    "selected_span_score": 0.05 * label,
                    "frame_score": frame_score,
                    "observer_reliability": 1.0,
                }
                if with_nli:
                    record["nli_score"] = float(label)
                records.append(record)
    return records


def test_evaluator_reports_locked_slices_and_fixed_view() -> None:
    result = evaluate(_records(with_nli=False))

    assert result["views"]["frame_fixed"]["overall"]["auroc"] == 1.0
    assert set(result["views"]["frame_fixed"]["per_evaluation_partition"]) == {
        "development",
        "held-predicate",
        "held-template",
        "double-holdout",
    }
    assert "cascade" not in result["views"]


def test_cascade_falls_back_only_for_ambiguous_frame_scores() -> None:
    result = evaluate(_records(with_nli=True))

    assert result["cascade"]["fallback_rate"] == 1.0 / 12.0
    assert result["views"]["cascade"]["role_errors"]["critical_recall"] == 1.0
    assert result["views"]["cascade"]["role_errors"][
        "invariant_false_positive_rate"
    ] == 0.0
    assert result["decision_gate"]["superseded_absolute_check"][
        "feasible_from_baseline"
    ] is False
    assert result["error_audit"]["false_negative_count"] == 0


def test_human_locked_records_use_the_frozen_human_gate() -> None:
    records = _records(with_nli=True)
    for record in records:
        group = int(record["base_id"].rsplit("-", 1)[-1]) % 3
        record["evaluation_partition"] = "human-locked"
        record["author_id_hash"] = f"author-{group}"
        record["source_group"] = f"source-{group}"
        record["selected_span_score"] = 0.05 * record["label"]
        if record["semantic_axis"] == "relation":
            record["selected_span_score"] = 0.0

    result = evaluate(records)

    assert result["decision_gate"]["gate_name"] == "human-frame-v1"
    assert result["decision_gate"]["passed"] is True
    assert set(result["views"]["cascade"]["per_author_id_hash"]) == {
        "author-0",
        "author-1",
        "author-2",
    }

    for record in records:
        record["selected_span_score"] = float(record["label"])
    perfect_baseline = evaluate(records)
    assert perfect_baseline["decision_gate"]["checks"][
        "relation_error_reduction_at_least_0_50_or_perfect"
    ] is True
