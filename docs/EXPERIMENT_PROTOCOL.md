# Spectra v2 Experiment Protocol

## Phase A — mixer comparison

Train small models first. Recommended initial scale: 10–30 million parameters.

Compare:

1. semantic-only baseline;
2. spectral-only baseline;
3. dual paths with static averaging;
4. dual paths with learned fusion.

Hold constant:

- tokenizer and dataset;
- effective parameter budget;
- optimizer updates and batch construction;
- sequence length;
- training objective;
- evaluation code;
- hardware class where practical.

Run at least three seeds. Report mean, standard deviation, raw runs, parameter count, measured latency,
and peak memory. Do not call a target result `SOTA`; compare only with reproduced baselines.

## Phase B — disagreement as a diagnostic

For each sample, save branch disagreement before fusion. Test whether it predicts:

- retrieval failure;
- low semantic-similarity accuracy;
- corrupted input;
- domain shift;
- independently labelled OOD datasets.

Compare against simple baselines:

- embedding norm;
- distance from the training centroid;
- energy score;
- nearest-neighbour distance;
- a small supervised confidence head.

Use AUROC, AUPRC, calibration error, and risk-coverage curves where appropriate. If disagreement does
not outperform useful baselines, it remains an interpretability signal rather than a control signal.

## Phase C — adaptive representation

Only after Phase B succeeds, test policies that choose:

- 32, 64, 128, or larger embedding prefixes;
- direct retrieval versus reranking;
- continued computation versus abstention.

Success means equal or better task quality with lower measured average cost, not merely a more complex
router.

## Reproducibility requirements

Every run directory must contain:

- resolved configuration;
- git commit SHA;
- random seed;
- environment and dependency versions;
- parameter count from instantiated modules;
- raw metrics;
- timing and memory measurements;
- checkpoint hash when a checkpoint is retained.

Failures and NaNs must stop the run and preserve diagnostics. Do not replace a non-finite loss with a
finite number and continue silently.

## Merge gates for future features

A new architectural component requires:

1. a true removal ablation;
2. a test proving the flag changes the module graph or forward path;
3. a matched baseline result;
4. documentation that distinguishes implementation from observed benefit.
