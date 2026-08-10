from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from spectra_v3.cache import EmbeddingCache
from spectra_v3.encoders import (
    CachedEncoder,
    EncoderSpec,
    SentenceTransformerEncoder,
)
from spectra_v3.evidence_analysis import (
    best_f1_threshold,
    binary_metrics,
    fit_logistic,
    grouped_bootstrap_delta,
    selective_cascade,
    similarity_slices,
)
from spectra_v3.external_evidence import (
    PROTOCOL_ID,
    EvidenceRecord,
    audit_dataset,
    cosine,
    deterministic_group_sample,
    embedding_pair_features,
    feature_groups,
    finite_feature_row,
    fixed_monotone_score,
    json_lines,
    load_records,
    orbit_spectral_features,
    purge_cross_split_groups,
    scalar_pair_features,
    structural_features,
    structural_texts,
    verify_frozen_files,
)


DATASETS = ("condaqa", "paws_wiki", "anli")
SPLITS = ("train", "dev", "test")
SAMPLE_CAPS = {
    "condaqa": {"train": 3000, "dev": 600, "test": 1200},
    "paws_wiki": {"train": 3000, "dev": 1000, "test": 1200},
    "anli": {"train": 3000, "dev": 1000, "test": 1200},
}
ENCODERS = {
    "minilm": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "prefix": "",
    },
    "e5": {
        "model_name": "intfloat/e5-base-v2",
        "revision": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        "prefix": "query: ",
    },
}
NLI = {
    "model_name": "cross-encoder/nli-deberta-v3-small",
    "revision": "fa2804872c3b4bd748f38c0185cc85775361e735",
}
BOOTSTRAP_REPLICATES = 1000
MAX_NLI_FRACTION = 0.40


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], text=True, encoding="utf-8"
    ).strip()


def _require_frozen_checkout(protocol_commit: str) -> str:
    if not protocol_commit or protocol_commit.casefold() in {
        "head",
        "main",
        "master",
        "latest",
    }:
        raise ValueError("--protocol-commit must be an explicit immutable SHA")
    current = _git("rev-parse", "HEAD")
    resolved = _git("rev-parse", protocol_commit)
    if current != resolved:
        raise RuntimeError(
            f"checkout {current} does not equal protocol commit {resolved}"
        )
    if _git("status", "--porcelain"):
        raise RuntimeError("Stage B requires a clean protocol-commit checkout")
    return current


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _audit(args: argparse.Namespace) -> int:
    raw_root = args.raw_root.resolve()
    output = args.output.resolve()
    result = {
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "structural audit only; no encoder or final metric was run",
        "files": verify_frozen_files(raw_root),
        "datasets": {
            dataset: audit_dataset(raw_root, dataset) for dataset in DATASETS
        },
    }
    _write_json(output, result)
    print(f"wrote structural audit {output}")
    return 0


def _sample_records(raw_root: Path) -> dict[str, dict[str, list[EvidenceRecord]]]:
    result: dict[str, dict[str, list[EvidenceRecord]]] = {}
    for dataset in DATASETS:
        result[dataset] = {}
        loaded = {
            split: load_records(raw_root, dataset, split) for split in SPLITS
        }
        controlled = purge_cross_split_groups(loaded)
        for split in SPLITS:
            result[dataset][split] = deterministic_group_sample(
                controlled[split], SAMPLE_CAPS[dataset][split]
            )
    return result


def _formatted(text: str, prefix: str) -> str:
    return prefix + text


def _encoded_map(
    encoder: CachedEncoder,
    texts: Sequence[str],
    prefix: str,
) -> dict[str, np.ndarray]:
    unique = sorted({text for text in texts if text})
    vectors = encoder.encode([_formatted(text, prefix) for text in unique])
    return {
        text: np.asarray(vector, dtype=np.float32)
        for text, vector in zip(unique, vectors, strict=True)
    }


def _feature_rows(
    records: Sequence[EvidenceRecord],
    vectors: Mapping[str, np.ndarray],
) -> list[dict[str, float]]:
    orbit_members: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in records:
        if record.orbit_id:
            orbit_members[record.orbit_id].append(record)
    spectra = {}
    for orbit_id, members in orbit_members.items():
        members = sorted(members, key=lambda row: row.example_id)
        if len(members) != 3:
            raise ValueError(f"sampled orbit is incomplete: {orbit_id}")
        spectra[orbit_id] = orbit_spectral_features(
            vectors[members[0].left_text],
            np.stack([vectors[row.right_text] for row in members]),
        )

    def semantic_similarity(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        return cosine(vectors[left], vectors[right])

    result = []
    for record in records:
        features = scalar_pair_features(record.left_text, record.right_text)
        features.update(
            embedding_pair_features(
                vectors[record.left_text], vectors[record.right_text]
            )
        )
        features.update(structural_features(record, semantic_similarity))
        if record.orbit_id:
            features.update(spectra[record.orbit_id])
        features["f4.fixed_compatibility"] = 1.0 - min(
            fixed_monotone_score(features), 1.0
        )
        finite_feature_row(features)
        result.append(features)
    return result


def _all_encoder_texts(
    sampled: Mapping[str, Mapping[str, Sequence[EvidenceRecord]]]
) -> list[str]:
    texts = set()
    for by_split in sampled.values():
        for records in by_split.values():
            for record in records:
                texts.add(record.left_text)
                texts.add(record.right_text)
                texts.update(text for text in structural_texts(record) if text)
    return sorted(texts)


def _nli_scores(
    records: Sequence[EvidenceRecord],
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sentence_transformers import CrossEncoder

    pairs = []
    pair_counts = []
    for record in records:
        pairs.append((record.left_text, record.right_text))
        if record.dataset in {"paws_wiki", "condaqa"}:
            pairs.append((record.right_text, record.left_text))
            pair_counts.append(2)
        else:
            pair_counts.append(1)
    started = perf_counter()
    model = CrossEncoder(
        NLI["model_name"],
        revision=NLI["revision"],
        device=device,
        trust_remote_code=False,
    )
    labels = {
        int(key): str(value) for key, value in model.model.config.id2label.items()
    }
    matches = [
        index for index, label in labels.items() if "entail" in label.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(f"cannot resolve NLI entailment class: {labels}")
    probabilities = np.asarray(
        model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=True,
            apply_softmax=True,
            convert_to_numpy=True,
        ),
        dtype=np.float64,
    )
    entailment = probabilities[:, matches[0]]
    scores = []
    cursor = 0
    for count in pair_counts:
        scores.append(float(np.min(entailment[cursor : cursor + count])))
        cursor += count
    elapsed = perf_counter() - started
    metadata = {
        **NLI,
        "id_to_label": labels,
        "entailment_index": matches[0],
        "pair_order": (
            "bidirectional-min for PAWS/CONDAQA; premise-to-hypothesis for ANLI"
        ),
        "logical_examples": len(records),
        "model_pairs": len(pairs),
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "examples_per_second": len(records) / elapsed,
    }
    return np.asarray(scores, dtype=np.float64), metadata


def _masked_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    mask: np.ndarray,
) -> dict[str, float] | None:
    if int(mask.sum()) < 20 or len(np.unique(labels[mask])) < 2:
        return None
    return binary_metrics(labels[mask], scores[mask], threshold)


def _evaluate_one(
    dataset: str,
    records: Mapping[str, Sequence[EvidenceRecord]],
    features: Mapping[str, Sequence[Mapping[str, float]]],
    nli: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = {
        split: np.asarray([row.label for row in records[split]], dtype=np.int64)
        for split in SPLITS
    }
    names = feature_groups(list(features["train"][0]))
    fitted = {}
    metrics: dict[str, Any] = {"task1": {}, "task2": {}, "bootstrap": {}}
    for group, columns in names.items():
        fit = fit_logistic(
            features["train"],
            labels["train"],
            features["dev"],
            labels["dev"],
            features["test"],
            columns,
        )
        fitted[group] = fit
        metrics["task1"][group] = {
            "all": binary_metrics(labels["test"], fit.test, fit.threshold)
        }

    dev_cosine = [row["f0.base_cosine"] for row in features["dev"]]
    test_cosine = np.asarray(
        [row["f0.base_cosine"] for row in features["test"]]
    )
    masks = similarity_slices(dev_cosine, test_cosine)
    for record_slice in sorted(
        {name for row in records["test"] for name in row.slices if name != "all"}
    ):
        masks[record_slice] = np.asarray(
            [record_slice in row.slices for row in records["test"]]
        )
    for group, fit in fitted.items():
        for slice_name, mask in masks.items():
            value = _masked_metrics(
                labels["test"], fit.test, fit.threshold, mask
            )
            if value is not None:
                metrics["task1"][group][slice_name] = value

    fixed_dev = np.asarray(
        [row["f4.fixed_compatibility"] for row in features["dev"]]
    )
    fixed_test = np.asarray(
        [row["f4.fixed_compatibility"] for row in features["test"]]
    )
    fixed_threshold = best_f1_threshold(labels["dev"], fixed_dev)
    metrics["task1"]["F4-fixed"] = {
        "all": binary_metrics(labels["test"], fixed_test, fixed_threshold)
    }

    nli_threshold = best_f1_threshold(labels["dev"], nli["dev"])
    metrics["task1"]["F5-NLI"] = {
        "all": binary_metrics(labels["test"], nli["test"], nli_threshold)
    }

    base_threshold = best_f1_threshold(labels["dev"], dev_cosine)
    error_labels = {
        split: (
            (np.asarray([row["f0.base_cosine"] for row in features[split]])
             >= base_threshold)
            != labels[split]
        ).astype(np.int64)
        for split in SPLITS
    }
    for group in ("F0", "F0+F1", "F0+F1+F3"):
        fit = fit_logistic(
            features["train"],
            error_labels["train"],
            features["dev"],
            error_labels["dev"],
            features["test"],
            names[group],
        )
        metrics["task2"][group] = binary_metrics(
            error_labels["test"], fit.test, fit.threshold
        )

    primary = fitted["F0+F1+F3"]
    baseline = fitted["F0+F1"]
    groups = [row.group_id for row in records["test"]]
    metrics["bootstrap"]["variables_vs_scalar"] = grouped_bootstrap_delta(
        labels["test"], baseline.test, primary.test, groups,
        replicates=BOOTSTRAP_REPLICATES,
    )
    if "F0+F1+F2+F3" in fitted:
        metrics["bootstrap"]["spectrum_beyond_variables"] = (
            grouped_bootstrap_delta(
                labels["test"],
                primary.test,
                fitted["F0+F1+F2+F3"].test,
                groups,
                replicates=BOOTSTRAP_REPLICATES,
            )
        )
        cheap = fitted["F0+F1+F2+F3"]
    else:
        cheap = primary

    cascade = selective_cascade(
        labels["dev"], cheap.dev, nli["dev"],
        labels["test"], cheap.test, nli["test"], test_cosine,
        max_call_fraction=MAX_NLI_FRACTION,
    )
    metrics["cascade"] = {
        key: value
        for key, value in cascade.items()
        if key not in {"score", "route_mask"}
    }
    raw = {
        "labels": labels["test"],
        "scores": {group: fit.test for group, fit in fitted.items()},
        "fixed": fixed_test,
        "nli": nli["test"],
        "cascade": cascade["score"],
        "route_mask": cascade["route_mask"],
        "base_threshold": base_threshold,
    }
    return metrics, raw


def _environment() -> dict[str, Any]:
    import torch

    packages = (
        "numpy",
        "pyarrow",
        "scikit-learn",
        "sentence-transformers",
        "torch",
        "transformers",
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: version(name) for name in packages},
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "pid": os.getpid(),
    }


def _run(args: argparse.Namespace) -> int:
    protocol_commit = _require_frozen_checkout(args.protocol_commit)
    raw_root = args.raw_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Stage B output: {output}")
    verify_frozen_files(raw_root)
    sampled = _sample_records(raw_root)
    output.mkdir(parents=True)
    started = perf_counter()
    timings: dict[str, Any] = {}
    all_metrics: dict[str, Any] = {}
    raw_outputs: dict[str, Any] = {}
    encoder_metadata: dict[str, Any] = {}
    nli_cache: dict[str, dict[str, np.ndarray]] = {}
    nli_metadata: dict[str, Any] = {}

    for dataset in DATASETS:
        nli_cache[dataset] = {}
        for split in ("dev", "test"):
            scores, metadata = _nli_scores(
                sampled[dataset][split],
                device=args.device,
                batch_size=args.nli_batch_size,
            )
            nli_cache[dataset][split] = scores
            nli_metadata[f"{dataset}:{split}"] = metadata

    all_texts = _all_encoder_texts(sampled)
    for encoder_name, identity in ENCODERS.items():
        encoder_started = perf_counter()
        spec = EncoderSpec(
            identity["model_name"], identity["revision"], normalize=False
        )
        base_encoder = SentenceTransformerEncoder(
            spec,
            device=args.device,
            batch_size=args.encoder_batch_size,
            show_progress=True,
        )
        cache_path = output / "cache" / f"{encoder_name}.sqlite3"
        with EmbeddingCache(cache_path) as cache:
            cached = CachedEncoder(base_encoder, cache)
            vectors = _encoded_map(cached, all_texts, identity["prefix"])
            encoder_metadata[encoder_name] = {
                **identity,
                "normalize_embeddings": False,
                "e5_prefix_policy": identity["prefix"] or None,
                "runtime": base_encoder.runtime_metadata(),
                "cache": cached.stats(),
                "unique_texts": len(all_texts),
            }
        all_metrics[encoder_name] = {}
        raw_outputs[encoder_name] = {}
        encoder_features = {}
        for dataset in DATASETS:
            feature_by_split = {
                split: _feature_rows(sampled[dataset][split], vectors)
                for split in SPLITS
            }
            encoder_features[dataset] = feature_by_split
            metrics, raw = _evaluate_one(
                dataset,
                sampled[dataset],
                feature_by_split,
                nli_cache[dataset],
            )
            all_metrics[encoder_name][dataset] = metrics
            raw_outputs[encoder_name][dataset] = raw
            rows = []
            for index, record in enumerate(sampled[dataset]["test"]):
                row = {
                    **record.to_dict(),
                    "features": feature_by_split["test"][index],
                    "scores": {
                        name: float(values[index])
                        for name, values in raw["scores"].items()
                    },
                    "fixed_score": float(raw["fixed"][index]),
                    "nli_score": float(raw["nli"][index]),
                    "cascade_score": float(raw["cascade"][index]),
                    "cascade_routed": bool(raw["route_mask"][index]),
                }
                rows.append(row)
            path = output / "raw" / encoder_name / f"{dataset}.test.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_lines(rows), encoding="utf-8")

        transfer = {}
        for target in DATASETS:
            sources = [dataset for dataset in DATASETS if dataset != target]
            train_rows = [
                row for dataset in sources
                for row in encoder_features[dataset]["train"]
            ]
            train_labels = [
                record.label for dataset in sources
                for record in sampled[dataset]["train"]
            ]
            dev_rows = [
                row for dataset in sources
                for row in encoder_features[dataset]["dev"]
            ]
            dev_labels = [
                record.label for dataset in sources
                for record in sampled[dataset]["dev"]
            ]
            columns = feature_groups(list(train_rows[0]))["F0+F1+F3"]
            fit = fit_logistic(
                train_rows,
                train_labels,
                dev_rows,
                dev_labels,
                encoder_features[target]["test"],
                columns,
            )
            target_labels = [
                record.label for record in sampled[target]["test"]
            ]
            transfer[target] = {
                "trained_on": sources,
                "metrics": binary_metrics(
                    target_labels, fit.test, fit.threshold
                ),
                "scores": [float(value) for value in fit.test],
                "example_ids": [
                    record.example_id for record in sampled[target]["test"]
                ],
            }
        all_metrics[encoder_name]["cross_dataset_transfer"] = {
            target: {
                "trained_on": value["trained_on"],
                "metrics": value["metrics"],
            }
            for target, value in transfer.items()
        }
        _write_json(
            output / "raw" / encoder_name / "cross-dataset-transfer.json",
            transfer,
        )
        timings[f"encoder:{encoder_name}"] = perf_counter() - encoder_started

    elapsed = perf_counter() - started
    timings["total_seconds"] = elapsed
    if args.device.startswith("cuda"):
        import torch

        peak_memory = int(torch.cuda.max_memory_allocated())
    else:
        peak_memory = None
    configuration = {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": protocol_commit,
        "sample_caps": SAMPLE_CAPS,
        "datasets": list(DATASETS),
        "encoders": ENCODERS,
        "nli": NLI,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "max_nli_fraction": MAX_NLI_FRACTION,
        "device": args.device,
        "encoder_batch_size": args.encoder_batch_size,
        "nli_batch_size": args.nli_batch_size,
    }
    _write_json(output / "configuration.json", configuration)
    _write_json(output / "environment.json", _environment())
    _write_json(output / "metrics.json", all_metrics)
    _write_json(output / "timing-and-calls.json", {
        "timings": timings,
        "nli": nli_metadata,
        "encoders": encoder_metadata,
        "peak_cuda_memory_bytes": peak_memory,
    })
    file_manifest = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and "cache" not in path.parts:
            file_manifest[str(path.relative_to(output)).replace("\\", "/")] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    _write_json(output / "manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_commit": protocol_commit,
        "raw_dataset_files": verify_frozen_files(raw_root),
        "sample_counts": {
            dataset: {
                split: len(rows) for split, rows in by_split.items()
            }
            for dataset, by_split in sampled.items()
        },
        "artifacts": file_manifest,
    })
    print(f"wrote frozen Stage B results to {output}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit or run the frozen existing-human-evidence protocol."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="structural audit; no models")
    audit.add_argument("--raw-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.set_defaults(function=_audit)

    run = subparsers.add_parser("run", help="execute frozen Stage B")
    run.add_argument("--raw-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--protocol-commit", required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--encoder-batch-size", type=int, default=64)
    run.add_argument("--nli-batch-size", type=int, default=32)
    run.set_defaults(function=_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
