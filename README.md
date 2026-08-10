# Spectra-NSA

Spectra-NSA is an experimental embedding research project studying whether a
representation should capture not only **where a text lies in embedding space**, but also
**how that representation responds to controlled semantic changes**.

The repository combines two related research lines:

1. a compact dual-path encoder with semantic self-attention and real Fourier token mixing;
2. semantic response spectra built from meaning-changing and meaning-preserving interventions.

The project is research software. It is not a production embedding service, anomaly detector,
or validated confidence system.

The first protocol-frozen evaluation on existing human public evidence is now complete. Its
result is mostly negative: the current semantic/frame variables do not add robust incremental
information beyond cheap diagnostics, the spectrum has only a narrow CONDAQA high-similarity
signal, and the selective NLI cascade misses its complete gate. **A custom Spectra student is not
currently justified.**

## Current status

The public human-evidence experiment answered the current question for the implemented
measurements. It did not support the broad semantic-variable, central-spectrum, or selective-
cascade claims. Architecture scaling and custom-student work are therefore paused.

| Layer | Current evidence | Status |
| --- | --- | --- |
| Legacy v0.2 prototype | Architecture and training experiments with unsupported claims | Preserved for history |
| `spectra_v2` encoder core | Masking, gradients, ablations, output contracts, and mixer modes | Implemented and unit-tested |
| Phase 0 response measurement | Deterministic intervention and measurement infrastructure | Implemented |
| Synthetic Pilots v1-v4 | Strong development results, including a structural frame observer and selective NLI fallback | Development-only |
| Existing human public evidence | Protocol-first evaluation on CONDAQA, PAWS-Wiki, and ANLI with pinned MiniLM/E5/NLI revisions | Stage B complete: primary variable/spectrum/cascade gates not supported |
| Natural-source mining | Real Stack Exchange text, provenance controls, deterministic filtering, and balanced seed selection | Operational pilot |
| Human-locked evaluation | Human transformations, independent blind review, freeze, then one-shot evaluation | Frozen and untouched; not yet run |

The earlier synthetic Pilot v4 result remains development-only: one gate was corrected after
development metrics were observed, and synthetic text cannot establish natural-language
robustness. The later public-data experiment was protocol-first and auditable, but its negative
result is also development evidence because pretrained models may know these public
distributions. It does not alter or consume the separate human-locked lane.

## Research question and answer

> Given frozen public encoders and human semantic labels, do task-relative response measurements,
> response spectra, or semantic/frame coordinates predict failures beyond cosine and cheap text
> diagnostics?

For the current implementation:

- cheap F1 diagnostics are useful on PAWS-Wiki and moderately useful on CONDAQA, but they combine
  response norm with strong lexical/edit signals and do not establish a distinct Spectra effect;
- F3 semantic/frame variables add only about `0.003` to `0.011` AUROC across the six primary
  encoder/dataset comparisons, with every grouped confidence interval crossing zero;
- F2 response spectrum shows a repeated high-similarity hint on CONDAQA, but no robust general or
  cross-dataset result;
- the frozen cascade never satisfies quality, call-budget, and routing-comparator gates together;
- leave-one-dataset-out transfer is weak, indicating mostly task-specific signal.

The appropriate interpretation is not that every idea in the repository is useless. The
measurement and audit infrastructure works; the present semantic representation is not yet a
general reliability signal worth distilling into a new encoder.

## Research decision

What remains active:

- protocol-first evaluation, immutable manifests, dataset hashing, and grouped leakage controls;
- cheap lexical/edit diagnostics as task-specific monitoring baselines;
- one narrow research lead: response-spectrum behavior inside balanced, high-similarity human
  intervention orbits.

What is paused or demoted:

- training or scaling a custom Spectra student;
- treating the current deterministic F3 frame extractor as validated incremental signal;
- treating response spectrum as a general central claim;
- the current learned-ambiguity NLI routing policy;
- new human collection until a smaller follow-up question is precise enough to justify it.

Any follow-up must be a new frozen protocol. It should isolate response magnitude from
lexical/edit features and use balanced human orbit slices with enough incompatible examples. It
must not reinterpret or tune against the completed v1 results.

## What is implemented

### Compact v2 encoder

- semantic-only, spectral-only, and dual mixer modes;
- standard multi-head self-attention;
- real FFT token mixing over each sample's valid sequence length;
- learned dual-path fusion;
- branch-disagreement and spectral-concentration diagnostics;
- Matryoshka prefix embeddings;
- explicit masking, gradient, and ablation tests.

### Semantic response measurement

- matched intervention orbits;
- task-relative critical, irrelevant-control, and relevant-invariant roles;
- relation, direction, scope, and modality axes;
- selected-span, variable-state, lexical, canonical-span, and structural-frame observers;
- deterministic manifests, hashes, frozen thresholds, and partition reports;
- selective NLI fallback for structurally ambiguous relation cases.

### Source-grounded human collection

- official Stack Exchange API adapter without HTML-page scraping;
- record-level license, attribution, retrieval time, canonical URL, and content hashes;
- PII minimization and removal of code, preformatted blocks, and quotations;
- strict structured inference proposals for query, frame, and semantic axis;
- exact and near-duplicate rejection;
- diversity selection across axes, sites, domains, predicates, syntax, and length;
- separate `seed-only` and development-only `auto-proposal` modes;
- pseudonymous, fail-closed author bundles with private provenance kept outside them;
- human draft validation, blind review packets, compilation, and locked-set intake.

Inference may discover a useful source, query, and candidate axis. It does not determine the final
label. In the collection path, humans independently write the intervention triplets and blind
reviewers must reconstruct the intended axis and relation without seeing model output.

## Evidence boundary

The repository keeps two independent evidentiary lanes:

```text
PUBLIC DEVELOPMENT LANE                 FUTURE HUMAN-LOCKED LANE
CONDAQA / PAWS-Wiki / ANLI              source-seeded author bundles
          ↓                                          ↓
frozen public encoders                  independent human authoring/review
          ↓                                          ↓
protocol-first Stage A → Stage B         immutable intake → one-shot test
          ↓                                          ↓
completed; negative primary gates        frozen, untouched, not yet run
```

Spectra, MiniLM, the frame observer, and NLI must not run on source seeds, author drafts, rejected
items, or review packets. Machine-written triplets are marked development-only and are rejected by
the human collection loader.

## Install and test

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
pytest
```

Install the optional Phase 0 model dependencies with:

```bash
python -m pip install -e ".[dev,phase0]"
```

Install the public human-evidence evaluation dependencies with:

```bash
python -m pip install -e ".[dev,human-evidence]"
```

The protocol and structural dataset audit are frozen before model evaluation. The Stage B runner
requires a clean checkout and an explicit immutable protocol commit:

```bash
spectra-existing-human-evidence audit --help
spectra-existing-human-evidence run --help
```

See the [existing human evidence protocol](docs/EXISTING_HUMAN_EVIDENCE_PROTOCOL.md) and
[findings](docs/EXISTING_HUMAN_EVIDENCE_FINDINGS.md). Its dataset hashes, grouping rules, features,
metrics, confidence intervals, and continuation/falsification gates were predeclared. The Stage B
result did not support building a custom student. The CLI fails closed on paths resembling the
human-locked/source-seeded lane.

The exact audit trail is split into two commits:

- `13221c6`: protocol, gates, adapters, and runner frozen before metric inspection;
- `5ca0a47`: raw per-example output, metrics, manifest, findings, and execution metadata.

Minimal encoder use:

```python
import torch

from spectra_v2 import SpectraConfig, SpectraEmbeddingModel

config = SpectraConfig(
    vocab_size=30_522,
    mixer_mode="dual",  # "semantic", "spectral", or "dual"
)
model = SpectraEmbeddingModel(config)

input_ids = torch.randint(0, config.vocab_size, (2, 32))
attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
output = model(input_ids, attention_mask)

embedding = output["embedding"]
diagnostics = output["diagnostics"]
```

A tokenizer, pretrained checkpoint, production training pipeline, and benchmark claim are not part
of the current v2 contract.

## Source-seeded rehearsal

The source-mining command exposes the complete acquisition and packaging workflow:

```bash
spectra-phase0-source-mine --help
```

The main stages are:

```bash
spectra-phase0-source-mine fetch ...
spectra-phase0-source-mine mine ...
spectra-phase0-source-mine pack ...
spectra-phase0-source-mine prepare-distribution ...
```

The real rehearsal fetched 80 public questions from four Stack Exchange sites. Deterministic
filtering retained 69 usable records, and diversity selection produced 12 source-grounded seeds:
three for each semantic axis and four for each of three source domains. They were packaged into
three isolated author bundles, each containing one case per axis.

These numbers describe workflow validation, not model quality. No model evaluation was run on the
rehearsal material.

See the
[source-seeded collection protocol](docs/PILOT_V4_SOURCE_SEEDED_COLLECTION_PROTOCOL.md) and
[source-mining findings](docs/PILOT_V4_SOURCE_MINING_FINDINGS.md) for commands, provenance rules,
limitations, and the human handoff.

## Repository map

| Path | Purpose |
| --- | --- |
| [`spectra_v2/`](spectra_v2/) | Compact heterogeneous-mixer encoder |
| [`spectra_v3/`](spectra_v3/) | Semantic intervention and response data structures |
| [`experiments/`](experiments/) | Corpora, observers, evaluators, source mining, and human collection CLIs |
| [`tests/`](tests/) | Unit and workflow tests |
| [`docs/`](docs/) | Research specifications, frozen protocols, and pilot findings |
| [`examples/`](examples/) | Human collection and review schema examples |

## Read in order

1. [Research reset and verified status](docs/RESEARCH_RESET.md)
2. [Existing human evidence protocol](docs/EXISTING_HUMAN_EVIDENCE_PROTOCOL.md)
3. [Existing human evidence findings](docs/EXISTING_HUMAN_EVIDENCE_FINDINGS.md)
4. [v2 architecture](docs/V2_ARCHITECTURE.md)
5. [v3 semantic response spectra specification](docs/V3_SEMANTIC_RESPONSE_SPECTRA.md)
6. [Phase 0 response measurement protocol](docs/PHASE0_RESPONSE_MEASUREMENT.md)
7. [Pilot v4 frame findings](docs/PILOT_V4_FRAME_FINDINGS.md)
8. [Source-seeded collection protocol](docs/PILOT_V4_SOURCE_SEEDED_COLLECTION_PROTOCOL.md)
9. [Source-mining findings](docs/PILOT_V4_SOURCE_MINING_FINDINGS.md)
10. [Human-locked collection protocol](docs/PILOT_V4_HUMAN_COLLECTION_PROTOCOL.md)

Earlier pilot reports remain available in [`docs/`](docs/) as an audit trail. Later reports do not
silently rewrite their results or frozen configurations.

## Legacy prototype

The root-level files `anomalous_embedding_ultimate.py`, `training_monitor.py`,
`anomalous_eval_suite.py`, and the older guides belong to the v0.2 prototype. They are retained for
idea recovery and historical inspection, not presented as the validated v2 implementation.

## License

Apache License 2.0. Public source material collected through external APIs retains its own
record-level license and attribution requirements; it is not relicensed by this repository.
