# Spectra-NSA

Spectra-NSA is an experimental embedding research project studying whether a
representation should capture not only **where a text lies in embedding space**, but also
**how that representation responds to controlled semantic changes**.

The repository combines two related research lines:

1. a compact dual-path encoder with semantic self-attention and real Fourier token mixing;
2. semantic response spectra built from meaning-changing and meaning-preserving interventions.

The project is research software. It is not a production embedding service, anomaly detector,
or validated confidence system.

## Current status

The code has moved beyond the original research reset, but the central hypothesis is **not yet
validated on locked human data**.

| Layer | Current evidence | Status |
| --- | --- | --- |
| Legacy v0.2 prototype | Architecture and training experiments with unsupported claims | Preserved for history |
| `spectra_v2` encoder core | Masking, gradients, ablations, output contracts, and mixer modes | Implemented and unit-tested |
| Phase 0 response measurement | Deterministic intervention and measurement infrastructure | Implemented |
| Synthetic Pilots v1-v4 | Strong development results, including a structural frame observer and selective NLI fallback | Development-only |
| Natural-source mining | Real Stack Exchange text, provenance controls, deterministic filtering, and balanced seed selection | Operational pilot |
| Human-locked evaluation | Human transformations, independent blind review, freeze, then one-shot evaluation | Not yet run |

The latest synthetic Pilot v4 result is promising, but it is not claim-eligible: one gate was
corrected after development metrics were observed, and synthetic text cannot establish natural
language robustness. The next evidentiary boundary is therefore a genuinely human-authored,
source-grounded locked set.

## The research question

> At matched parameter, data, and update budgets, do heterogeneous semantic and Fourier token
> mixers produce complementary embedding views, and can task-relative response to controlled
> semantic interventions expose failures that a single embedding distance misses?

This is a hypothesis, not a result.

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

The repository deliberately separates development evidence from evaluation evidence:

```text
public natural text
        ↓
deterministic filtering + inference-assisted seed discovery
        ↓
human critical / control / invariant authoring
        ↓
two independent blind reviews per intervention
        ↓
balanced intake + immutable manifest
        ↓
one-shot evaluation on the frozen implementation
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
2. [v2 architecture](docs/V2_ARCHITECTURE.md)
3. [v3 semantic response spectra specification](docs/V3_SEMANTIC_RESPONSE_SPECTRA.md)
4. [Phase 0 response measurement protocol](docs/PHASE0_RESPONSE_MEASUREMENT.md)
5. [Pilot v4 frame findings](docs/PILOT_V4_FRAME_FINDINGS.md)
6. [Source-seeded collection protocol](docs/PILOT_V4_SOURCE_SEEDED_COLLECTION_PROTOCOL.md)
7. [Source-mining findings](docs/PILOT_V4_SOURCE_MINING_FINDINGS.md)
8. [Human-locked collection protocol](docs/PILOT_V4_HUMAN_COLLECTION_PROTOCOL.md)

Earlier pilot reports remain available in [`docs/`](docs/) as an audit trail. Later reports do not
silently rewrite their results or frozen configurations.

## Legacy prototype

The root-level files `anomalous_embedding_ultimate.py`, `training_monitor.py`,
`anomalous_eval_suite.py`, and the older guides belong to the v0.2 prototype. They are retained for
idea recovery and historical inspection, not presented as the validated v2 implementation.

## License

Apache License 2.0. Public source material collected through external APIs retains its own
record-level license and attribution requirements; it is not relicensed by this repository.
