# Contributing to Spectra-NSA

Spectra-NSA welcomes contributions that make the research question easier to test, falsify, compare, or reproduce.

The project is not currently seeking architecture scaling or optimization of the legacy v0.2 model. The most useful contributions are often methodological: alternative baselines, stronger controls, independent replications, critiques of the current measurement object, or small experiments that can distinguish competing explanations.

Before contributing, read:

1. [`README.md`](../README.md)
2. [`docs/RESEARCH_RESET.md`](../docs/RESEARCH_RESET.md)
3. [`docs/EXISTING_HUMAN_EVIDENCE_FINDINGS.md`](../docs/EXISTING_HUMAN_EVIDENCE_FINDINGS.md)
4. [`docs/V3_SEMANTIC_RESPONSE_SPECTRA.md`](../docs/V3_SEMANTIC_RESPONSE_SPECTRA.md)

If your contribution touches human-authored evidence, also read the relevant frozen collection and evaluation protocols under [`docs/`](../docs/).

## Current research state

The first protocol-frozen evaluation on existing public human evidence is complete. Its primary result is negative:

- the current semantic/frame variables do not show robust incremental value beyond cheap diagnostics;
- response spectrum retains only a narrow high-similarity lead that has not generalized;
- the current selective NLI cascade does not satisfy its complete quality/cost/comparator gate;
- a custom Spectra student is not currently justified.

Those results are evidence, not a target to optimize away.

The repository therefore welcomes work that helps answer questions such as:

- Can response geometry be isolated from lexical/edit confounds?
- Is the high-similarity response-spectrum observation reproducible under a different dataset or experimental design?
- Which selective-prediction, defer, uncertainty, or retrieval-risk baselines should be compared before another router is designed?
- Is a response spectrum the right object at all, or would gradients, Jacobians, local subspaces, contrast sets, counterfactual geometry, calibrated nearest-neighbor evidence, or another representation capture the relevant phenomenon better?

A contribution does not need to support the Spectra hypothesis. A well-designed negative result is useful.

## Evidence boundaries

### Frozen public evidence

Completed protocol-frozen results must remain unchanged.

Do not:

- edit historical raw results to reflect a later interpretation;
- tune thresholds against completed test results and present the rerun as the same experiment;
- modify a frozen protocol after observing its outcome;
- reinterpret a failed gate as passed by changing metrics or subsets post hoc;
- encode dataset-specific exceptions into a supposedly general method.

A revised experiment requires a new protocol version, explicit rationale, and separate result identity.

### Human-locked lane

The human-locked lane exists to preserve a one-shot evaluation boundary.

Do not run Spectra models, baselines, observers, NLI models, or exploratory scoring over:

- source seeds reserved for the locked lane;
- author drafts;
- review packets;
- rejected items;
- compiled pre-freeze material.

Do not use human-locked material to choose features, thresholds, architectures, prompts, routing policies, or stopping criteria.

If you are unsure whether a proposed contribution would consume or contaminate this lane, open an issue before implementing it.

## Good contribution types

### Research-method contributions

Examples:

- propose a stronger baseline with a clear reason it addresses an observed failure mode;
- implement an alternative local-sensitivity representation behind a separate experimental path;
- design a confound-controlled comparison between response magnitude and lexical/edit features;
- reproduce a narrow finding on independently selected public data;
- add grouped or leakage-aware evaluation where a simpler split would be misleading;
- identify relevant prior art and explain how its assumptions differ from the current Spectra setup;
- challenge a metric, aggregation rule, intervention design, or claim boundary with a concrete test.

### Reproducibility and audit contributions

Examples:

- deterministic tests;
- manifest or provenance validation;
- environment capture;
- model-free result replay;
- dataset-integrity checks;
- documentation that distinguishes historical, development, held-out, and locked evidence.

### Engineering contributions

Engineering work is welcome when it supports a current experiment or makes the evidence path safer and clearer.

Please do not optimize legacy training throughput, distributed training, mixed precision, or checkpointing merely to scale the old architecture. The current research decision does not justify that work.

Legacy root-level code is preserved primarily for history and idea recovery. New architectural research should use the modular `spectra_v2`, `spectra_v3`, and `experiments` paths unless an issue explicitly says otherwise.

## Proposing a new experiment

Before writing a large experimental PR, open an issue describing:

- the research question;
- the competing explanations or hypotheses;
- why the current evidence cannot answer it;
- the proposed baseline(s);
- the dataset/evidence class you intend to use;
- which data, if any, will be used for development or tuning;
- the metrics and grouping strategy;
- what outcome would count against your preferred hypothesis;
- whether the experiment must be frozen before execution.

Prefer small experiments that distinguish explanations over broad architecture changes.

## Development setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
pytest
```

For Phase 0 model-backed work:

```bash
python -m pip install -e ".[dev,phase0]"
```

For the existing public human-evidence tooling:

```bash
python -m pip install -e ".[dev,human-evidence]"
```

Do not install or run model-backed extras merely to work on model-free tests, documentation, schemas, or replay tooling.

## Pull request expectations

Keep one conceptual change per PR where practical.

A research or evaluation PR should state:

- research question or failure mode addressed;
- affected package/experiment;
- evidence class used (`synthetic`, `public development`, `held-out`, etc.);
- whether any reported data was visible during method development;
- baselines and controls;
- commands executed;
- raw-result location when applicable;
- known limitations;
- whether any protocol, metric definition, grouping rule, threshold, or claim boundary changed.

If results are negative or inconclusive, preserve them as such.

## Tests

Run the repository tests relevant to your change:

```bash
pytest
```

Add deterministic tests for new pure logic where possible. Model-free replay, schema, grouping, hashing, and protocol tests are strongly preferred over tests that require downloading large models.

## Claims and language

Do not describe a new result as `SOTA`, `production-ready`, `confidence`, `uncertainty`, `anomaly detection`, or a general semantic reliability signal unless the experiment actually establishes the corresponding standard.

Distinguish clearly between:

- hypothesis;
- development observation;
- protocol-frozen result;
- held-out result;
- human-locked result;
- interpretation.

## Questions and research discussion

Open a GitHub issue with the `question` or `help wanted` label when you want to:

- compare another method with Spectra;
- point to prior work that may solve the same problem differently;
- propose a falsification or replication;
- question an experimental assumption;
- discuss a new protocol before implementation.

Code is not required for a useful contribution. A precise methodological critique can be more valuable than another model branch.

## License

Spectra-NSA is licensed under the Apache License 2.0. Unless explicitly stated otherwise, contributions intentionally submitted for inclusion in the project are provided under the same license terms. External datasets and public-source records retain their own license and attribution requirements.