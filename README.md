# Spectra-NSA

Spectra-NSA is an experimental embedding research project exploring **heterogeneous token mixing**:
a conventional semantic self-attention path, a real Fourier path, and the possibility that their
disagreement can become a useful diagnostic signal.

## Current status: research reset

The original v0.2 implementation is preserved in this repository as a legacy prototype. It contains
real architecture and training work, but several claims, benchmarks, and monitoring features were
not validated strongly enough. New development starts from the small `spectra_v2` core and follows a
strict experiment-first protocol.

Read first:

- [Research reset and verified status](docs/RESEARCH_RESET.md)
- [v2 architecture](docs/V2_ARCHITECTURE.md)
- [experiment protocol](docs/EXPERIMENT_PROTOCOL.md)

## The narrow research question

> At matched parameter, data, and update budgets, do heterogeneous semantic and Fourier token
> mixers produce complementary embedding views, and does their disagreement predict difficult or
> unreliable inputs?

This is a hypothesis, not a result.

## What the v2 core contains

- semantic-only, spectral-only, and dual architectural modes;
- standard multi-head self-attention;
- real FFT-based token mixing over each sample's valid sequence length;
- learned dual-path fusion;
- branch-disagreement and spectral-concentration diagnostics;
- Matryoshka prefix embeddings;
- unit tests for masking, ablation boundaries, gradients, and output contracts.

The current v2 core is deliberately small. It does **not** claim state-of-the-art performance,
calibrated confidence, anomaly detection, or production readiness.

## Install and test

```bash
python -m pip install -e ".[dev]"
pytest
```

Minimal use:

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

A tokenizer, training pipeline, benchmark harness, and model checkpoints are intentionally not yet
part of the v2 contract. They will be added only alongside reproducible experiments.

## Legacy prototype

The root-level files `anomalous_embedding_ultimate.py`, `training_monitor.py`,
`anomalous_eval_suite.py`, and the older guides belong to the v0.2 prototype. They are retained for
history and idea recovery, not presented as the validated v2 implementation.

## License

Apache License 2.0.
