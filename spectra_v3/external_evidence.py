from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence
import zipfile

import numpy as np
import numpy.typing as npt

from .frame_variables import (
    extract_semantic_frame,
    fixed_frame_distance,
    frame_compatibility,
)
from .response import l2_normalize, measure_response


PROTOCOL_ID = "existing-human-evidence-v1"
SAMPLING_SEED = "spectra-existing-human-evidence-v1"
FORBIDDEN_FEATURE_FRAGMENTS = frozenset(
    {
        "answer",
        "axis",
        "dataset",
        "expected",
        "generator",
        "intervention_role",
        "label",
        "model_label",
        "passage_edit_id",
        "role",
        "target",
        "template",
    }
)
FROZEN_FILES = {
    "condaqa/train.json": (
        "f5f8a9d1b64a6c3b111b00d2e125f1d7f4da8b50acfdbc788d2901e2e43d79ec"
    ),
    "condaqa/dev.json": (
        "b9ec9ab7453fbdc61fbed988119c03f507341ac495f1f5f95b13b27f59e639a2"
    ),
    "condaqa/test.json": (
        "ffd7f4ad85689b2317283430a226cd759c3ccc40070e6a32652e7ae84ae90e8e"
    ),
    "paws/train-00000-of-00001.parquet": (
        "8dc9ad3e5f30ad9a86b290fe236d528ef23a5751fec9a35d99cbacf68ba277cf"
    ),
    "paws/validation-00000-of-00001.parquet": (
        "7760d829453764ba342a6f562809a8ed21c2c3eec3fd9ffa544089f145d42f6d"
    ),
    "paws/test-00000-of-00001.parquet": (
        "ae342ff12bb84b84b95f468abf5db6cb7c7bd578271299fe9c99be75b8132f4d"
    ),
    "anli_v1.0.zip": (
        "e5c058f2bb4e6190b0651badca2c590e45db95248f8b28cde674615ee40820bf"
    ),
}

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_NEGATION = re.compile(
    r"\b(?:no|not|never|neither|nor|without|cannot|can't|won't|don't|didn't)\b",
    re.IGNORECASE,
)
_MODAL = re.compile(
    r"\b(?:may|might|must|could|should|would|possibly|perhaps|certainly)\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_external_path(path: Path) -> None:
    folded = str(path.resolve()).casefold().replace("_", "-")
    forbidden = (
        "source-seed",
        "source-seeded",
        "human-locked",
        "author-bundle",
        "pilot-v4",
    )
    if any(fragment in folded for fragment in forbidden):
        raise ValueError(
            "refusing to access frozen human/source-seeded material: "
            f"{path}"
        )


@dataclass(frozen=True)
class EvidenceRecord:
    dataset: str
    split: str
    example_id: str
    group_id: str
    left_text: str
    right_text: str
    label: int
    slices: tuple[str, ...] = ()
    orbit_id: str | None = None
    orbit_role: str | None = None
    query_text: str | None = None
    reference_text: str | None = None
    candidate_text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("label must be binary")
        required = (
            self.dataset,
            self.split,
            self.example_id,
            self.group_id,
            self.left_text,
            self.right_text,
        )
        if any(not value.strip() for value in required):
            raise ValueError("record identity and text fields cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["slices"] = list(self.slices)
        result["metadata"] = dict(self.metadata)
        return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _answer(value: Any) -> str:
    return normalize_text(str(value)).strip(" .\"'")


def load_condaqa(raw_root: Path, split: str) -> list[EvidenceRecord]:
    source = raw_root / "condaqa" / f"{split}.json"
    assert_external_path(source)
    rows = _read_jsonl(source)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["PassageID"]), str(row["QuestionID"]))
        grouped[key].append(row)

    records: list[EvidenceRecord] = []
    role_names = {"1": "paraphrase", "2": "scope", "3": "affirmative"}
    for key, orbit in sorted(grouped.items()):
        by_role = {str(row["PassageEditID"]): row for row in orbit}
        if set(by_role) != {"0", "1", "2", "3"}:
            continue
        original = by_role["0"]
        query = str(original["sentence2"])
        reference = str(original["sentence1"])
        orbit_id = f"condaqa:{key[0]}:{key[1]}"
        for role in ("1", "2", "3"):
            edited = by_role[role]
            candidate = str(edited["sentence1"])
            role_name = role_names[role]
            records.append(
                EvidenceRecord(
                    dataset="condaqa",
                    split=split,
                    example_id=f"{orbit_id}:{role}",
                    group_id=orbit_id,
                    left_text=f"Question: {query}\nPassage: {reference}",
                    right_text=f"Question: {query}\nPassage: {candidate}",
                    label=int(_answer(original["label"]) == _answer(edited["label"])),
                    slices=("all", f"edit_{role_name}"),
                    orbit_id=orbit_id,
                    orbit_role=role_name,
                    query_text=query,
                    reference_text=reference,
                    candidate_text=candidate,
                    metadata={
                        "passage_id": str(original["PassageID"]),
                        "question_id": str(original["QuestionID"]),
                        "sample_id": str(edited["SampleID"]),
                    },
                )
            )
    return records


def load_paws(raw_root: Path, split: str) -> list[EvidenceRecord]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency message
        raise RuntimeError("PAWS requires the human-evidence optional extra") from exc
    disk_split = "validation" if split == "dev" else split
    source = raw_root / "paws" / f"{disk_split}-00000-of-00001.parquet"
    assert_external_path(source)
    rows = pq.read_table(source).to_pylist()
    records = []
    for row in rows:
        left = str(row["sentence1"])
        right = str(row["sentence2"])
        canonical = "\x1f".join(sorted((normalize_text(left), normalize_text(right))))
        group_id = f"paws:{stable_hash(canonical)}"
        records.append(
            EvidenceRecord(
                dataset="paws_wiki",
                split=split,
                example_id=f"paws:{row['id']}",
                group_id=group_id,
                left_text=left,
                right_text=right,
                label=int(row["label"]),
                slices=("all",),
                metadata={"source_id": str(row["id"])},
            )
        )
    return records


def load_anli(raw_root: Path, split: str) -> list[EvidenceRecord]:
    source = raw_root / "anli_v1.0.zip"
    assert_external_path(source)
    records = []
    with zipfile.ZipFile(source) as archive:
        for round_name in ("R1", "R2", "R3"):
            member = f"anli_v1.0/{round_name}/{split}.jsonl"
            lines = archive.read(member).decode("utf-8").splitlines()
            for line in lines:
                if not line.strip():
                    continue
                row = json.loads(line)
                context = str(row["context"])
                hypothesis = str(row["hypothesis"])
                records.append(
                    EvidenceRecord(
                        dataset="anli",
                        split=split,
                        example_id=f"anli:{row['uid']}",
                        group_id=f"anli:{stable_hash(normalize_text(context))}",
                        left_text=context,
                        right_text=hypothesis,
                        label=int(row["label"] == "e"),
                        slices=("all", f"round_{round_name.casefold()}"),
                        metadata={
                            "round": round_name,
                            "original_label": str(row["label"]),
                            "genre": str(row.get("genre", "")),
                        },
                    )
                )
    return records


def load_records(raw_root: Path, dataset: str, split: str) -> list[EvidenceRecord]:
    loaders = {
        "condaqa": load_condaqa,
        "paws_wiki": load_paws,
        "anli": load_anli,
    }
    try:
        loader = loaders[dataset]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset: {dataset}") from exc
    return loader(raw_root, split)


def deterministic_group_sample(
    records: Sequence[EvidenceRecord], cap: int
) -> list[EvidenceRecord]:
    if cap <= 0:
        raise ValueError("cap must be positive")
    grouped: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)
    ordered = sorted(
        grouped,
        key=lambda group: stable_hash(f"{SAMPLING_SEED}\x1f{group}"),
    )
    selected: list[EvidenceRecord] = []
    for group in ordered:
        candidate = grouped[group]
        if selected and len(selected) + len(candidate) > cap:
            continue
        selected.extend(candidate)
        if len(selected) >= cap:
            break
    return sorted(selected, key=lambda row: row.example_id)


def audit_group_leakage(
    split_records: Mapping[str, Sequence[EvidenceRecord]],
) -> dict[str, Any]:
    group_sets = {
        split: {record.group_id for record in records}
        for split, records in split_records.items()
    }
    overlaps: dict[str, int] = {}
    split_names = sorted(group_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlaps[f"{left}__{right}"] = len(
                group_sets[left].intersection(group_sets[right])
            )
    return {
        "records": {
            split: len(records) for split, records in split_records.items()
        },
        "groups": {split: len(groups) for split, groups in group_sets.items()},
        "group_overlap": overlaps,
    }


def purge_cross_split_groups(
    split_records: Mapping[str, Sequence[EvidenceRecord]],
) -> dict[str, list[EvidenceRecord]]:
    """Keep test authoritative, then dev, and remove shared groups upstream."""

    required = {"train", "dev", "test"}
    if set(split_records) != required:
        raise ValueError(f"expected exactly these splits: {sorted(required)}")
    test = list(split_records["test"])
    test_groups = {record.group_id for record in test}
    dev = [
        record for record in split_records["dev"]
        if record.group_id not in test_groups
    ]
    held_out = test_groups | {record.group_id for record in dev}
    train = [
        record for record in split_records["train"]
        if record.group_id not in held_out
    ]
    return {"train": train, "dev": dev, "test": test}


def verify_frozen_files(raw_root: Path) -> dict[str, dict[str, Any]]:
    assert_external_path(raw_root)
    result = {}
    for relative, expected in FROZEN_FILES.items():
        path = raw_root / relative
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"frozen dataset hash mismatch for {relative}: {actual}"
            )
        result[relative] = {
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return result


def audit_dataset(raw_root: Path, dataset: str) -> dict[str, Any]:
    splits = {
        split: load_records(raw_root, dataset, split)
        for split in ("train", "dev", "test")
    }
    result = audit_group_leakage(splits)
    controlled = purge_cross_split_groups(splits)
    result["after_group_purge"] = audit_group_leakage(controlled)
    result["labels"] = {
        split: dict(sorted(Counter(row.label for row in rows).items()))
        for split, rows in splits.items()
    }
    return result


def word_tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD.finditer(text)]


def scalar_pair_features(left: str, right: str) -> dict[str, float]:
    left_tokens = word_tokens(left)
    right_tokens = word_tokens(right)
    left_set, right_set = set(left_tokens), set(right_tokens)
    union = left_set | right_set
    intersection = left_set & right_set
    shorter = max(min(len(left_set), len(right_set)), 1)
    longer = max(max(len(left_set), len(right_set)), 1)
    return {
        "f0.left_chars": float(len(left)),
        "f0.right_chars": float(len(right)),
        "f0.char_delta": float(abs(len(left) - len(right))),
        "f0.left_tokens": float(len(left_tokens)),
        "f0.right_tokens": float(len(right_tokens)),
        "f0.token_delta": float(abs(len(left_tokens) - len(right_tokens))),
        "f0.token_ratio": float(min(len(left_tokens), len(right_tokens)) / max(
            max(len(left_tokens), len(right_tokens)), 1
        )),
        "f1.jaccard": float(len(intersection) / max(len(union), 1)),
        "f1.containment": float(len(intersection) / shorter),
        "f1.vocab_ratio": float(shorter / longer),
        "f1.edit_similarity": float(
            SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()
        ),
        "f1.negation_left": float(len(_NEGATION.findall(left))),
        "f1.negation_right": float(len(_NEGATION.findall(right))),
        "f1.negation_delta": float(
            abs(len(_NEGATION.findall(left)) - len(_NEGATION.findall(right)))
        ),
        "f1.modal_delta": float(
            abs(len(_MODAL.findall(left)) - len(_MODAL.findall(right)))
        ),
        "f1.punctuation_delta": float(
            abs(len(_PUNCT.findall(left)) - len(_PUNCT.findall(right)))
        ),
    }


def validate_feature_names(names: Iterable[str]) -> None:
    violations = []
    for name in names:
        folded = name.casefold()
        if any(fragment in folded for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
            violations.append(name)
    if violations:
        raise ValueError(f"forbidden leakage-prone feature names: {sorted(violations)}")


def cosine(left: npt.ArrayLike, right: npt.ArrayLike) -> float:
    left_vector = l2_normalize(np.asarray(left, dtype=np.float64))
    right_vector = l2_normalize(np.asarray(right, dtype=np.float64))
    return float(np.clip(left_vector @ right_vector, -1.0, 1.0))


def embedding_pair_features(
    left: npt.ArrayLike, right: npt.ArrayLike
) -> dict[str, float]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    left_unit = l2_normalize(left_array)
    right_unit = l2_normalize(right_array)
    return {
        "f0.base_cosine": float(left_unit @ right_unit),
        "f0.left_norm": float(np.linalg.norm(left_array)),
        "f0.right_norm": float(np.linalg.norm(right_array)),
        "f0.norm_delta": float(
            abs(np.linalg.norm(left_array) - np.linalg.norm(right_array))
        ),
        "f1.response_norm": float(np.linalg.norm(right_unit - left_unit)),
    }


def orbit_spectral_features(
    base: npt.ArrayLike, transformed: npt.ArrayLike
) -> dict[str, float]:
    measurement = measure_response(base, transformed, rank=3)
    source = measurement.feature_dict(prefix="orbit")
    result = {"f2." + key.removeprefix("orbit."): value for key, value in source.items()}
    validate_feature_names(result)
    return result


def _string_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio())


def structural_texts(record: EvidenceRecord) -> tuple[str, ...]:
    query = record.query_text or record.left_text
    reference = record.reference_text or record.left_text
    candidate = record.candidate_text or record.right_text
    reference_frame = extract_semantic_frame(query, reference)
    candidate_frame = extract_semantic_frame(query, candidate)
    return (
        reference_frame.predicate,
        reference_frame.actor,
        reference_frame.patient,
        reference_frame.span,
        candidate_frame.predicate,
        candidate_frame.actor,
        candidate_frame.patient,
        candidate_frame.span,
    )


def structural_features(
    record: EvidenceRecord,
    similarity: Callable[[str, str], float] = _string_similarity,
) -> dict[str, float]:
    query = record.query_text or record.left_text
    reference = record.reference_text or record.left_text
    candidate = record.candidate_text or record.right_text
    reference_frame = extract_semantic_frame(query, reference)
    candidate_frame = extract_semantic_frame(query, candidate)
    compatibility = frame_compatibility(
        reference_frame, candidate_frame, similarity
    )
    values = compatibility.to_dict()
    drops = {
        f"{name}_drop": max(1.0 - float(value), 0.0)
        for name, value in values.items()
        if name != "reliability"
    }
    result = {
        f"f3.{name}": float(value) for name, value in values.items()
    }
    result.update({f"f3.{name}": float(value) for name, value in drops.items()})
    result["f3.fixed_frame_distance"] = fixed_frame_distance(drops)
    result["f3.reference_missing"] = float(reference_frame.reliability == 0.0)
    result["f3.candidate_missing"] = float(candidate_frame.reliability == 0.0)
    result["f3.span_similarity"] = similarity(
        reference_frame.span, candidate_frame.span
    )
    validate_feature_names(result)
    return result


def fixed_monotone_score(features: Mapping[str, float]) -> float:
    incompatibilities = [
        0.5 * max(1.0 - features["f0.base_cosine"], 0.0),
        features["f3.fixed_frame_distance"],
        0.5 * max(1.0 - features["f3.span_similarity"], 0.0),
    ]
    reliability = features["f3.reliability"]
    penalty = 0.25 * (1.0 - reliability)
    return float(max(incompatibilities) + penalty)


def feature_groups(feature_names: Sequence[str]) -> dict[str, list[str]]:
    names = sorted(feature_names)
    validate_feature_names(names)
    by_prefix = {
        prefix: [name for name in names if name.startswith(f"{prefix}.")]
        for prefix in ("f0", "f1", "f2", "f3")
    }
    groups = {
        "F0": by_prefix["f0"],
        "F0+F1": by_prefix["f0"] + by_prefix["f1"],
        "F0+F1+F3": by_prefix["f0"] + by_prefix["f1"] + by_prefix["f3"],
    }
    if by_prefix["f2"]:
        groups["F0+F1+F2"] = (
            by_prefix["f0"] + by_prefix["f1"] + by_prefix["f2"]
        )
        groups["F0+F1+F2+F3"] = names
    return groups


def json_lines(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
        for row in rows
    )


def finite_feature_row(features: Mapping[str, float]) -> None:
    validate_feature_names(features)
    invalid = [name for name, value in features.items() if not math.isfinite(value)]
    if invalid:
        raise ValueError(f"non-finite features: {invalid}")
