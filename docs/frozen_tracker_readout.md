# Frozen tracker momentum readout: local experiment

This diagnostic starts from the small scaling model at 50,000 steps. The backbone,
set decoder, shared hard grouping, presence, PID, direction and energy predictions
are frozen. Only three independent 77,825-parameter momentum heads are trained.
Each is a copy of the existing zero-initialized TrackerMomentumReadout and changes
only the final predicted charged-slot log-pT. The correction is bounded by log(1.5).

Oracle membership pooling is implemented entirely in the study helper
`scripts/studies/frozen_tracker_pooling.py`. Hard and soft learned associations
delegate to the production pooling function unchanged. The oracle path mirrors
its reductions using fixed memberships, with output and gradient parity tests;
the production model API does not accept truth-association overrides.

## Arms and controls

- **Baseline:** the frozen checkpoint, with no correction.
- **Hard:** tracker summary from the checkpoint's initial hard associations.
- **Soft:** the same summary operator with top-4/background truncated-softmax
  weights. This changes only the momentum branch; shared pooling stays hard.
- **Oracle:** truth hit membership mapped to the frozen baseline's Hungarian
  slot-to-target matches. This is a diagnostic upper bound for this head and
  representation, not a deployable model. Truth momentum is never an input.

All heads start identically and receive exactly the same batches. Extraction uses
inference mode and BF16; summaries, head computation and regression are FP32.
The three arms share no trainable parameters. Head optimization uses AdamW,
LR 0.001 cosine-decayed to 0.0001, weight decay 0.01, per-head gradient clipping 1,
5,000 updates, and batch 128. Validation and checkpoint intervals are 1,000.
This is a new head-only schedule, not continuation of the original LAMB optimizer.

The objective retains the baseline sqrt(target pT)-weighted log-pT MSE and its
matched-particle normalization. Training Hungarian assignments are calculated
once from the frozen baseline; corrections cannot alter supervision assignments.
The predicted charged-PID and tracker-support gates remain unchanged.

## Local data and budget

Only CLD dataset configuration `1:3.2.1` is installed on this machine. Each process
(ttbar, WW fullhad, qq) contributes 4,096 randomly selected **train-split** events,
for 12,288 cached training events. Sampling is without replacement within each
batch; batches repeat the finite cache over 640,000 event presentations. This is
not the full ten-configuration training campaign.

Validation uses test-split indices `[2048, 2304)` per process (768 events total).
Final evaluation uses `[2304, 2816)` (1,536 events total), disjoint from validation.
The first 2,048 test events are skipped to avoid earlier local diagnostics. These
are held out from head training and selection, not guaranteed untouched by every
historical backbone validation or study. Seed is 12345. Event IDs and ordering are
saved in each cache's `events.json`.

The frozen backbone is evaluated one event at a time on the local 16 GB RTX 5060 Ti.
FP32 summary caches require approximately 13 GiB for all three splits. They avoid
repeated backbone inference. The command checks free disk space and holds a run
lock. Extraction resumes at its last flushed progress record, and head training
resumes from `checkpoints/latest.pth`. Arguments, checkpoint hashes and source
hashes must agree on resume. Do not edit source files underneath a running job.
The self-contained helper is included in the source manifest. Earlier completed
runs retain their original source snapshots; this refactor intentionally changes
the source hashes, so a new extraction/training run requires a new output directory.
It does not rewrite the earlier caches or results.

## Evaluation and interpretation

Every validation point reports the baseline and all three heads on identical
events: process-mean particle F1, charged/charged-hadron/neutral efficiency,
rematched central-68% residual width and tails, mean event momentum error, and
charged-hadron residuals on the fixed baseline training-matcher angular cohort.
Physics rematching uses the prior studies' geometry/log-pT Hungarian cost.
Fractional tracker purity/completeness, effective counts, support and correction
saturation are also recorded. Undefined ownership fractions are excluded, not
counted as zeros. These diagnostics refer to baseline-matched truth particles.

Each head selects its minimum validation fixed-cohort charged-hadron width,
including the zero-correction step-0 candidate. This diagnostic selection does
**not** impose an F1/efficiency guard: all accompanying tradeoffs must be reviewed
before promoting any head. The selected heads are evaluated once on final held-out
events, including jets with the existing CLD configuration. Jet IQR here uses
response-qualified matches; energy remains frozen, so this is a pT-only
intervention rather than a complete physically consistent momentum/energy model.
Oracle metrics are always marked diagnostic-only. One seed is not a seed study.

## Launch and monitor

From the repository root, outside the sandbox for GPU access:

```bash
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 .venv/bin/python -m scripts.studies.train_frozen_tracker_readout \
  --run experiments/cld_pf_heptv2_model_scaling_50k/hit_heptv2_small_seed12345_20261016_121953_896470 \
  --output experiments/cld_pf_heptv2_frozen_readout_local_5k
```

The launched job writes `run.log` and `launcher.json` in its output directory.
Inspect `cache/train/progress.json` during extraction, then `history/valid_step_*.json`
and `checkpoints/` during training. `selected_heads.json` records selected steps;
`completed.json` is written only after final test evaluation succeeds. Checkpoints
store heads, optimizer, scheduler and minibatch RNG state; the large base checkpoint
is referenced by hash rather than duplicated.

The separate `cld_pf_heptv2_frozen_readout_local_batch128_smoke` directory is only
an end-to-end implementation check, not physics evidence.

## Calibration control follow-up

`scripts/studies/calibrate_frozen_tracker_readout.py` fits simple controls on the
same cached **training** events, with separate controls for hard and soft support
gates. The constant control has one coefficient. The pT/eta control has six
features: `[1, z, a, z², za, a²]`, where `z=clip(log(predicted pT), -4, 6)/3` and
`a=clip(abs(predicted eta), 0, 5)/2`. Neither sees hit embeddings or geometry.

Weighted least squares uses the original sqrt(target pT) weighting and frozen
slot matches. Soft support multiplies the design before fitting, matching its
continuous inference gate. Ridge values 0, 0.001, 0.01, 0.1 and 1 are selected
using validation fixed-cohort charged-hadron width; the intercept is unpenalized.
The constant uses unregularized fitting. A zero-correction candidate is included.
The fitted polynomial is clipped to the same +/-log(1.5) correction bound at
inference. The fit itself solves the unclipped least-squares problem.

```bash
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 .venv/bin/python -m scripts.studies.calibrate_frozen_tracker_readout
```

Outputs are under `experiments/cld_pf_heptv2_frozen_calibration_local`:
`calibration_selection.json`, `results.json`, per-arm event arrays, and
`paired_intervals.json`. The analysis verifies reproduction of the earlier
learned-head test metrics before computing 1,000 paired process-stratified event
bootstrap intervals. Width quantiles use exact integer event multiplicities,
including all particles within each resampled event. Intervals cover event
sampling, not training-seed or validation-selection uncertainty. Since these
are the already-inspected test events, this is an exploratory control comparison,
not a new independent confirmation sample.
