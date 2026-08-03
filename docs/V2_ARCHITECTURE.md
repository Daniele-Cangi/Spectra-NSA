# Spectra v2 Architecture

## Research thesis

Spectra v2 studies whether two heterogeneous token mixers produce complementary semantic views:

1. **Semantic path** — conventional multi-head self-attention.
2. **Spectral path** — FFT along the valid token sequence, followed by a learned complex filter and
   inverse FFT.

The dual model measures branch disagreement before learned fusion. Disagreement is currently a raw
diagnostic only; it is not calibrated confidence or an anomaly score.

## Modes

- `semantic`: semantic mixer only;
- `spectral`: Fourier mixer only;
- `dual`: both mixers plus a learned token-wise gate.

These are true architectural ablations: disabled branches are not instantiated.

## Padding contract

The semantic mixer uses a key-padding mask. The reference Fourier mixer transforms each sample's
valid prefix independently and writes zeros to padded positions. This implementation is slower than
a fully vectorized version, but makes the masking contract explicit and testable.

## Embedding contract

The model returns:

```python
{
    "embedding": Tensor[batch, largest_dimension],
    "matryoshka": {dimension: Tensor[batch, dimension]},
    "diagnostics": {"branch_disagreement": Tensor[batch]},  # dual mode only
    "last_hidden_state": Tensor[batch, sequence, hidden],
}
```

Matryoshka outputs are normalized prefixes of one learned representation. Their usefulness must be
measured through task quality and neighbour preservation at each dimension, not by comparing a
prefix with itself.

## Deliberate omissions

The foundation does not yet include:

- an anomaly head;
- an adaptive compute policy;
- an early-exit mechanism;
- a training objective;
- a tokenizer contract;
- benchmark-specific code;
- model-size marketing presets.

Those components should be added only after the mixer comparison is reproducible.

## Possible next layer

If branch disagreement predicts errors better than standard uncertainty baselines, it may later
control:

- embedding dimension;
- reranking escalation;
- additional layers or model routing;
- abstention.

That is the long-term direction, not a current capability.
