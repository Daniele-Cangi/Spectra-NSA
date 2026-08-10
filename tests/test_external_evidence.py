from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spectra_v3.external_evidence import (
    EvidenceRecord,
    assert_external_path,
    audit_group_leakage,
    deterministic_group_sample,
    embedding_pair_features,
    feature_groups,
    load_condaqa,
    orbit_spectral_features,
    purge_cross_split_groups,
    scalar_pair_features,
    structural_features,
    validate_feature_names,
)


def _record(index: int, group: str, label: int = 0) -> EvidenceRecord:
    return EvidenceRecord(
        dataset="fixture",
        split="test",
        example_id=f"item-{index}",
        group_id=group,
        left_text="Alice follows Bob.",
        right_text="Bob follows Alice.",
        label=label,
    )


def test_human_locked_paths_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen human"):
        assert_external_path(tmp_path / "pilot-v4-source-seed-distribution")


def test_group_sample_never_splits_an_orbit() -> None:
    records = [
        _record(index, f"group-{index // 3}", index % 2) for index in range(12)
    ]
    selected = deterministic_group_sample(records, 7)
    counts = {}
    for record in selected:
        counts[record.group_id] = counts.get(record.group_id, 0) + 1
    assert selected
    assert all(count == 3 for count in counts.values())
    assert len(selected) <= 7


def test_group_overlap_audit_is_explicit() -> None:
    result = audit_group_leakage(
        {
            "train": [_record(0, "shared")],
            "dev": [_record(1, "dev")],
            "test": [_record(2, "shared")],
        }
    )
    assert result["group_overlap"]["test__train"] == 1


def test_group_purge_prioritizes_test_then_dev() -> None:
    controlled = purge_cross_split_groups(
        {
            "train": [_record(0, "shared"), _record(1, "train")],
            "dev": [_record(2, "shared"), _record(3, "dev")],
            "test": [_record(4, "shared"), _record(5, "test")],
        }
    )
    assert {row.group_id for row in controlled["test"]} == {"shared", "test"}
    assert {row.group_id for row in controlled["dev"]} == {"dev"}
    assert {row.group_id for row in controlled["train"]} == {"train"}


def test_leakage_prone_feature_names_are_rejected() -> None:
    for name in (
        "f3.axis_code",
        "f2.intervention_role",
        "f1.target_label",
        "f0.dataset_id",
    ):
        with pytest.raises(ValueError, match="forbidden"):
            validate_feature_names([name])


def test_feature_ladder_is_nested_and_label_blind() -> None:
    record = _record(1, "one", 1)
    features = scalar_pair_features(record.left_text, record.right_text)
    features.update(embedding_pair_features([2.0, 0.0], [0.0, 3.0]))
    features.update(structural_features(record))
    features.update(
        orbit_spectral_features(
            [1.0, 0.0, 0.0],
            [[0.9, 0.1, 0.0], [0.9, 0.0, 0.1], [0.8, 0.1, 0.1]],
        )
    )
    groups = feature_groups(list(features))
    assert set(groups["F0"]) < set(groups["F0+F1"])
    assert set(groups["F0+F1"]) < set(groups["F0+F1+F3"])
    assert set(groups["F0+F1+F3"]) < set(groups["F0+F1+F2+F3"])
    validate_feature_names(features)
    assert all(np.isfinite(value) for value in features.values())


def test_condaqa_adapter_uses_human_answer_equality(tmp_path: Path) -> None:
    root = tmp_path / "public-data"
    source = root / "condaqa"
    source.mkdir(parents=True)
    rows = []
    for role, answer in enumerate(("YES", "YES", "NO", "YES")):
        rows.append(
            {
                "PassageID": 3,
                "QuestionID": "q1",
                "PassageEditID": role,
                "SampleID": role,
                "sentence1": f"Passage version {role}.",
                "sentence2": "Is this supported?",
                "label": answer,
            }
        )
    (source / "test.json").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    records = load_condaqa(root, "test")
    assert [record.label for record in records] == [1, 0, 1]
    assert len({record.group_id for record in records}) == 1
    assert {record.orbit_role for record in records} == {
        "paraphrase",
        "scope",
        "affirmative",
    }
    assert all("label" not in record.left_text.casefold() for record in records)
