# Existing Human Evidence Findings v1

Protocol: `existing-human-evidence-v1`

Frozen Stage A commit: `13221c6410d2fda5e346445570cb0d46e49b0892`

Execution status: completed once, unchanged, on the frozen commit

Verdict: **the primary semantic-variable gate fails; a custom Spectra student is
not justified. The spectral component has a localized CONDAQA hint, but not
robust enough to remain a central claim.**

## Integrity and scope

Stage A was committed and pushed before any model evaluation. Stage B then ran
from a clean checkout of that exact SHA. The committed result directory contains
configuration, environment, raw per-example outputs, aggregate metrics,
grouped-bootstrap intervals, timing/call counts, dataset hashes, and an artifact
manifest. All 12 artifacts listed by the manifest were re-hashed successfully.

No source-seeded/human-locked v4 material was opened by the runner. The public
dataset inputs came from the separate external raw root and matched every frozen
SHA-256 digest.

The deterministic samples were 3,000 train / 588 dev / 1,200 test for CONDAQA,
and 3,000 / 1,000 / 1,200 for each of PAWS-Wiki and ANLI. Cross-split group
overlap after the frozen purge was zero.

## Primary compatibility result

AUROC on the full test samples:

| Encoder | Dataset | F0 base | F0+F1 scalar/cheap | +F3 variables | +F2+F3 where valid | NLI all pairs |
|---|---|---:|---:|---:|---:|---:|
| MiniLM | CONDAQA | 0.569 | 0.606 | 0.609 | 0.612 | 0.758 |
| MiniLM | PAWS-Wiki | 0.592 | 0.810 | 0.813 | — | 0.778 |
| MiniLM | ANLI | 0.491 | 0.480 | 0.489 | — | 0.549 |
| E5 | CONDAQA | 0.564 | 0.625 | 0.628 | 0.641 | 0.758 |
| E5 | PAWS-Wiki | 0.577 | 0.805 | 0.809 | — | 0.778 |
| E5 | ANLI | 0.452 | 0.465 | 0.476 | — | 0.549 |

F1 clearly adds predictive information on PAWS-Wiki and moderately on CONDAQA,
but F1 deliberately bundles response norm with edit/overlap/negation and length
diagnostics. This result therefore supports a cheap diagnostic bundle, not a
claim that response geometry itself caused the gain. On PAWS the dominant signal
is consistent with task-specific lexical/edit structure. ANLI remains near
chance for the cheap probes.

The fixed monotone F4 score is not useful: AUROC is 0.493/0.495 on CONDAQA,
0.536/0.535 on PAWS, and 0.469/0.468 on ANLI for MiniLM/E5.

## Semantic variables: predeclared gate fails

The primary grouped-bootstrap comparison is F0+F1+F3 minus F0+F1:

| Encoder | Dataset | AUROC delta | 95% grouped CI |
|---|---|---:|---:|
| MiniLM | CONDAQA | +0.0035 | [-0.0042, +0.0113] |
| MiniLM | PAWS-Wiki | +0.0027 | [-0.0047, +0.0099] |
| MiniLM | ANLI | +0.0095 | [-0.0059, +0.0239] |
| E5 | CONDAQA | +0.0032 | [-0.0045, +0.0108] |
| E5 | PAWS-Wiki | +0.0045 | [-0.0012, +0.0110] |
| E5 | ANLI | +0.0108 | [-0.0028, +0.0243] |

Every delta is below the predeclared 0.03 threshold and every interval includes
zero. The direction is usually positive across encoders, but its magnitude is
too small to count as support.

Inside the high-similarity slice, F3 AUROC deltas were +0.0019/+0.0151 on
CONDAQA, +0.0044/+0.0043 on PAWS, and +0.0332/+0.0062 on ANLI for MiniLM/E5.
Only one isolated encoder/dataset slice reaches 0.03 and it does not reproduce.

Task 2 gives the same answer. Adding F3 to F0+F1 changes base-error-prediction
AUROC by +0.0027/-0.0007/+0.0062 for MiniLM and +0.0035/+0.0033/+0.0128 for E5
on CONDAQA/PAWS/ANLI. These are small and inconsistent in associated AUPRC.

Conclusion: the present deterministic frame/semantic-variable machinery adds
no demonstrated incremental information beyond the frozen cheap diagnostics.

## Spectrum: localized signal, central gate not credibly satisfied

F2 exists only for genuine CONDAQA orbits.

| Encoder | Full AUROC delta over F1+F3 | 95% grouped CI | Full AUPRC delta | High-sim AUROC delta | High-sim AUPRC delta |
|---|---:|---:|---:|---:|---:|
| MiniLM | +0.0030 | [-0.0114, +0.0183] | +0.0104 | +0.0294 | +0.0305 |
| E5 | +0.0126 | [-0.0005, +0.0256] | +0.0276 | +0.0181 | +0.0534 |

This is the most interesting non-null pattern: spectrum helps particularly in
the high-similarity region for both encoders, and E5's full AUPRC delta exceeds
0.02. However, neither full-set AUROC CI excludes zero.

The role-slice threshold is not credible evidence of replication. The apparent
large paraphrase AUROC deltas (+0.065 MiniLM, +0.158 E5) arise in a slice with
399 compatible examples and only one incompatible example. Scope and
affirmative slices do not reach the 0.02 criterion consistently. Thus the
literal E5 combination of full AUPRC plus degenerate paraphrase AUROC is not
treated as the "meaningful incremental value on two slices" required by the
protocol's central-claim wording.

Conclusion: retain the CONDAQA high-similarity spectrum observation as a narrow
follow-up lead, but demote response spectrum from a core general Spectra claim.
It has not reproduced beyond one orbit dataset and lacks a positive grouped CI.

## NLI and selective cascade

NLI-on-every-pair is strongest on CONDAQA (AUROC 0.758) and ANLI (0.549), while
the PAWS F0+F1 probe is stronger than this small NLI model (about 0.805–0.810
versus 0.778). The NLI baseline is therefore useful but not universally dominant.

| Encoder | Dataset | NLI AUROC | Cascade AUROC | Retained | Logical call fraction | Random route | Base-sim route |
|---|---|---:|---:|---:|---:|---:|---:|
| MiniLM | CONDAQA | 0.758 | 0.676 | 89.1% | 43.3% | 0.671 | 0.686 |
| MiniLM | PAWS-Wiki | 0.778 | 0.759 | 97.6% | 39.8% | 0.756 | 0.810 |
| MiniLM | ANLI | 0.549 | 0.509 | 92.7% | 40.9% | 0.485 | 0.487 |
| E5 | CONDAQA | 0.758 | 0.701 | 92.5% | 39.9% | 0.679 | 0.684 |
| E5 | PAWS-Wiki | 0.778 | 0.759 | 97.5% | 40.0% | 0.756 | 0.810 |
| E5 | ANLI | 0.549 | 0.497 | 90.5% | 38.8% | 0.483 | 0.487 |

Only PAWS retains at least 95% NLI AUROC, but its cascade is decisively worse
than the base-similarity comparator because the cheap PAWS probe was already
better than NLI. The other datasets miss the quality target. No row satisfies
the complete quality/cost/comparator gate.

There is also a documented Stage B implementation deviation: dev-quantile ties
were routed with `>=`, causing MiniLM CONDAQA and ANLI to exceed the nominal 40%
cap (43.3% and 40.9%). The outputs are not altered or rerun. Those rows are
automatically disqualified on cost. Any exact-count tie fix requires a versioned
Stage A revision before a rerun; it would not rescue the quality/comparator gate.

Risk-coverage agrees with the negative gate. Cascade AURC is worse than NLI on
all six comparisons. The base-similarity PAWS policy is also better than the
learned ambiguity route.

## Cross-dataset transfer

Leave-one-dataset-out F0+F1+F3 AUROC:

| Held dataset | MiniLM | E5 |
|---|---:|---:|
| CONDAQA | 0.539 | 0.538 |
| PAWS-Wiki | 0.624 | 0.613 |
| ANLI | 0.475 | 0.510 |

Transfer is much weaker than in-domain PAWS/CONDAQA performance and near chance
on ANLI. The learned cheap state is largely task-specific rather than a general
semantic reliability detector.

## Cost and reproducibility record

- Frozen embedding texts: 40,758 per encoder; 81,516 total encoder examples.
- NLI dev/test logical examples: 6,188; actual directional model pairs: 10,176.
- Runner wall time: 1,484.8 seconds (24.7 minutes).
- Per-encoder pipeline time: 537.0 seconds MiniLM, 691.8 seconds E5.
- Recorded NLI inference time across six runs: 169.9 seconds.
- Peak allocated CUDA memory: 2,471,651,840 bytes (2.30 GiB).
- Hardware: NVIDIA GeForce RTX 2060; CUDA 13.0.
- Runtime: Python 3.13.5, PyTorch 2.13.0+cu130,
  sentence-transformers 5.7.0, transformers 5.14.1, scikit-learn 1.9.0,
  NumPy 2.5.1, PyArrow 25.0.0.

## Direct answers

1. **Does anything beyond cosine/base diagnostics predict human semantic
   failure?** Yes: the F1 cheap bundle is strong on PAWS and modest on CONDAQA.
   This does not isolate a specifically Spectra-originated response effect.
2. **Does it survive high-similarity conditioning?** The cheap PAWS bundle does.
   F3 does not add a reproduced meaningful gain. CONDAQA spectrum shows a
   repeatable high-similarity hint, without full-set CI support.
3. **Does it reproduce across frozen encoders?** The task-specific F1 pattern
   does. The semantic-variable continuation gate does not. Spectrum's high-sim
   direction repeats, but its central evidence does not.
4. **Does response spectrum add beyond scalar response?** A small/localized
   amount on CONDAQA, strongest at high similarity; not robust enough for a
   general central claim.
5. **Do semantic variables/frame coordinates add anything?** Only tiny deltas;
   all primary CIs include zero. No supported incremental value.
6. **Does combining spectrum and variables help?** On CONDAQA it is the best
   cheap learned probe, especially for E5, but the improvement is driven by F2;
   F3 itself is negligible.
7. **How does NLI compare?** It is substantially better on CONDAQA and modestly
   better on ANLI; it is worse than the task-specific PAWS cheap probe.
8. **Can selective fallback preserve quality with fewer calls?** Not under the
   frozen complete gate. PAWS preserves quality but loses to its comparator;
   CONDAQA/ANLI lose too much quality, and two MiniLM rows exceed the call cap.
9. **What survives into a next architecture?** The protocol/audit machinery,
   grouped leakage controls, cheap lexical/edit diagnostics as task-specific
   monitoring, and at most a narrowly scoped CONDAQA spectrum follow-up. The
   current F3 extractor, fixed F4 aggregation, general cascade policy, and the
   claim that spectrum is broadly central do not survive.
10. **Is a custom student justified?** No. There is no sufficiently general,
    incremental target worth distilling. Architecture work should remain paused.

## Decision

The most accurate failure interpretation is **A plus D**, with a narrow spectral
lead: the current semantic variables are redundant/weak, and the useful cheap
signal is task-specific. The cascade also exhibits **F** outside PAWS, while PAWS
does not need the teacher enough to justify this router.

Do not build or scale a custom Spectra student from these results. If another
cheap protocol is funded, it should be explicitly smaller and adversarial: isolate
response norm from lexical/edit features, build balanced human orbit slices with
enough incompatible preserving edits, and test the CONDAQA high-similarity
spectrum hint with grouped uncertainty. That would be a new protocol, not a
reinterpretation of this one.

## Artifacts

- `results/existing-human-evidence-v1/manifest.json`
- `results/existing-human-evidence-v1/configuration.json`
- `results/existing-human-evidence-v1/environment.json`
- `results/existing-human-evidence-v1/timing-and-calls.json`
- `results/existing-human-evidence-v1/metrics.json`
- `results/existing-human-evidence-v1/raw/{minilm,e5}/*.test.jsonl`
- `results/existing-human-evidence-v1/raw/{minilm,e5}/cross-dataset-transfer.json`
