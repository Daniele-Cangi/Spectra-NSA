# Spectra-NSA Research Reset

## Why this reset exists

Spectra-NSA v0.2 contains a real experimental encoder, training code, checkpointing, monitoring,
and evaluation utilities. It also accumulated names and claims faster than the underlying evidence.
This document separates the historical prototype from the next research phase.

## Status of the legacy prototype

The legacy root-level implementation is preserved for idea recovery and historical reference. It is
not the validated v2 core.

Known issues include:

- the old `spectral` path is an attention-like token mixer, not a spectral transform;
- its correlation enrichment has cubic sequence-length cost;
- padding is not consistently excluded from the secondary mixing path;
- the old `--no-spectral` switch does not remove the architectural branch;
- some evaluation utilities use stale output keys;
- the Matryoshka consistency check is tautological because it compares slices created from the same
  vector;
- the simplified MS MARCO evaluation is not comparable to standard corpus-level retrieval results;
- the old anomaly score has no dedicated OOD training signal;
- several monitoring fields are placeholders rather than measurements from live embeddings;
- parameter, storage, and training-time estimates require revalidation.

These are engineering and experimental limitations, not a judgment that the project has no value.

## Verified in the v2 foundation

The `spectra_v2` package currently verifies only structural properties:

- semantic-only, spectral-only, and dual modes instantiate different module graphs;
- the spectral path uses a real FFT and learned complex frequency filter;
- padding does not change the embedding of the valid prefix in evaluation mode;
- gradients reach both paths in dual mode;
- Matryoshka outputs have stable shapes and unit norm;
- invalid head geometry is rejected early.

The local validation used for this reset completed with 7 passing tests.

## Not yet verified

The repository does not yet establish that:

- the Fourier path improves embedding quality;
- dual-path fusion beats a matched Transformer baseline;
- branch disagreement predicts retrieval errors or OOD inputs;
- any diagnostic is calibrated confidence;
- adaptive dimension selection reduces cost at fixed quality;
- the model is competitive with current embedding systems;
- the v2 core is ready for production.

## Claim policy

A claim may move from **hypothesis** to **observed** only when the repository contains:

1. an executable experiment;
2. a matched baseline;
3. saved configuration and commit SHA;
4. raw per-seed results;
5. a reproducible summary generated from those results.

Words such as `SOTA`, `production-ready`, `anomaly detection`, and `confidence` must not be used as
results until their standard evaluation criteria are met.

## Preservation policy

Do not delete the legacy prototype while v2 is exploratory. New code should live in the modular v2
package. Legacy fixes should be limited to security, reproducibility, or documentation corrections;
architectural research belongs in v2.
