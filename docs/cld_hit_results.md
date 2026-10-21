# CLD raw-hit reconstruction: evidence and current recommendation

This is the evidence available in the local study archives reviewed on
**21 October 2026**, not a live report of running jobs. The figures and fuller
comparison are attached to [PR #508](https://github.com/jpata/particleflow/pull/508).
The dated `notebooks/studies` directories are scientific snapshots; many of their
scripts, inputs and outputs are local artifacts, not part of a fresh checkout.

## Recommended default and experimental options

Keep `pyg-cld-hits-heptv2-set-v1`: width 256, six backbone layers, four global
Flash decoder layers, 256 slots, grid-diverse aggregate proposals, supervised
hard grouping and tracker-neighborhood input features. The default scenario is
`cld_pf_heptv2_default_50k` (batch 64, 50k steps). Newer small-model readout/loss
campaigns use batch 128; their absolute results are not paired effects of that
batch-size change.

The dedicated tracker momentum head and particle-balanced grouping CE are
promising opt-in options, not changes to the recommended default. Sparse pooling,
extra momentum/doublet inputs, fitted readouts and auxiliary fit/overlap losses
remain experimental. The width-only, three-seed scenario combines the head and
balanced CE as an unvalidated candidate; it has no result in this evidence snapshot.

## Findings at matched checkpoints

Each row uses its own campaign control. Gains cannot be added across rows.
Widths below are charged-particle relative-pT central-68% half-widths unless
otherwise stated; F1 is test particle F1 except where marked validation.

| Intervention | Step | Result and interpretation |
|---|---:|---|
| Global HEPTv2 versus attention | 50k | F1 0.6297 → 0.7197; jet IQR 0.1646 → 0.1293; event p3 error 23.26 → 18.34 GeV. Strong observed improvement for the fixed single-seed pair. |
| Supervised hit grouping | 50k | Validation F1 0.7150 → 0.7213 and MET error 6.07 → 5.55 GeV; jet IQR improves in all three processes. |
| Tracker-neighborhood features | 50k | Width 16.53% → 14.89%, efficiency 77.34% → 78.40%, F1 0.7304 → 0.7370. Supports the current default. |
| Tracker momentum head with hard pooling | 50k | Width 12.78% → 12.40%, tails 3.69% → 3.38%, jet IQR 0.10316 → 0.09815. F1 +0.15 pp has paired 95% interval [-0.32, +0.62] pp. |
| Sparse pooling, alone / with head | 50k | F1 -0.82 / -1.23 pp, with paired intervals excluding zero; widths worsen to 13.69% / 13.21% from 12.78%. Do not promote. |
| Particle-balanced grouping CE | 32k, partial | Width 16.86% → 15.81%, tails 4.98% → 4.63%; F1 +0.45 pp has interval [-0.05, +0.96] pp. Jet IQR slightly worsens. |
| Overlap alone / with balanced CE | 32k, partial | Alone worsens F1 and width; combined width is 15.49%. Combined F1 improvement remains uncertain; synergy is not established. |
| Momentum moments / moments + doublets | 10k, partial | Width 19.97% → 23.96% / 21.27%, with lower charged efficiency. Initial gains did not persist; no completed-schedule conclusion. |
| Geometry supervision | 35k, partial | Width 17.91% → 17.29%, but tracker completeness drops; the 30k result was worse. Mixed evidence. |
| Curvature / geometry + curvature | 30k / 45k, partial | Worse efficiency, widths and tails. Not supported as defaults at the tested weights. |
| Bounded fit correction / tracker anchors | 50k | Width 16.05% → 15.87% / 15.48%, but efficiency 78.30% → 76.70% / 76.61% and higher tails. Narrower cores are not an overall win. |

The readout and grouping-loss campaigns are independent factorials. In
particular, the 32k loss result cannot be ranked against the 50k readout result.
The head's tail reduction has a paired interval excluding zero, but its F1 gain
does not. These intervals describe event sampling, not training-seed robustness.

## Scaling and the remaining PF gap

The joint width/depth study uses widths 256/384/512, backbone depths 6/9/12 and
decoder depths 4/6/8. Trainable parameter counts are 12.71M/42.06M/98.81M.

At the latest common **42k** prediction checkpoint:

| Model | F1 | Charged width | Jet IQR | Event p3 error, GeV |
|---|---:|---:|---:|---:|
| Small | 0.76273 | 13.46% | 0.10363 | 17.06 |
| Medium | 0.75749 | 14.48% | 0.10560 | 16.32 |
| Large | 0.75392 | 15.13% | 0.09479 | 15.73 |

Large improves jet response and closure but costs about 2.84 times the recorded
elapsed time and loses inclusive charged precision. At completed 50k, small also
beats medium (F1 0.7756/0.7547; charged width 12.43%/14.22%). Large has not completed
50k in this archive. Early 21k/31k advantages do not establish the final ranking.

Historical track/cluster references remain substantially more precise: the
September reference has jet IQR 0.0710 and event p3 error 13.21 GeV. The separate
October reference has charged width about 3.10%. These are different reference
models and training recipes, not a combined best-of baseline or a matched-budget
control. Higher particle-finding F1 alone does not establish equivalent momentum
reconstruction.

Resource logs show roughly 17 ms exposed batch wait versus 2.53/4.78/7.58 s
instrumented compute. This does not indicate an exposed I/O bottleneck. It does
not measure total asynchronous storage work, CPU saturation or memory bandwidth.

## Provenance and metric compatibility

- Backbone, grouping, tracker features, fits and input-feature evidence:
  `20261008_cld_pf_heptv2_comparison`; final fit corrections and held-out filtering
  are also retained in `20261013_cld_hit_progress/inputs`.
- Scaling: `20261019_cld_heptv2_model_scaling`, refreshed to matched 42k on
  21 October, with separate small/medium 50k results.
- Readout and grouping losses: `20261021_cld_tracker_losses_readout`, run revisions
  `45969feb` and `983b2bd3`, respectively.

These campaigns use seed 12345 and CLD TFDS 3.2.1. Recent paired comparisons use
512 events each from ttbar, WW fullhad and qq, with exact truth-event multisets
verified within a campaign. F1 and jet metrics average processes equally;
charged widths pool particles. Historical matching uses uncapped
`(deltaR/0.1)^2 + (log(pT ratio)/log(2))^2`; acceptance requires deltaR < 0.1 and
absolute relative pT error < 0.5, without PID agreement. Widths use angular
matches without the momentum acceptance cut, although assignment still uses pT.

The current repository validation matcher caps each cost at 2.0, and the current
model specification selects CLD hits TFDS 3.2.2. Re-running a scenario today is
therefore not automatically a reproduction of these historical results. Use the
archived revision/configuration and metric definitions, or rescore every model
with a common estimator and explicitly identify the changed dataset.

Most experiments have one seed. The latest width/slice estimates lack confidence
intervals, and repeated test inspection makes the findings exploratory. Frozen
instance probes use an oracle foreground mask; truth-owned/cleaned fits are
upper-bound diagnostics, not deployable reconstruction results. No reported
improvement removes the need to check efficiency, tails, jet response and closure
alongside momentum core width.
