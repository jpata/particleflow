# CLD hit-model training campaigns

This note records the design and outcomes of the 50k-step CLD hit-model scenarios that led to the current default, `pyg-cld-hits-heptv2-set-v1`. Launch commands for each scenario are in the [scenario README](https://github.com/jpata/particleflow/blob/main/configs/training/README.md); the overall evidence is summarized in [CLD hit-model evidence](cld_hit_results.md).

## Current default: `cld_pf_heptv2_default_50k`

The default promotes the 50k `hit_heptv2_grouped_tracker_features` result: six HEPTv2 layers, four global Flash decoder layers, 256 slots, supervised hit grouping, and tracker-neighborhood features. Circle-fit readout, bounded fit correction, detector-balanced grouping loss, and event-consistency loss are disabled. The recipe uses learning rate 0.001, global batch 64, and seed 12345.

On the study's 1,536 test events, the charged pT residual half-width improved from 16.53% to 14.89%, and charged matching efficiency from 77.34% to 78.40%, relative to the previous grouped reference. This is a single-seed result evaluated on 2026-10-13.

## Hit recipe in the backbone benchmark

The hit variants of `clic_cld_backbone_benchmark` use the small-model recipe with the tracker momentum head from the completed readout campaign (`notebooks/studies/20261021_cld_tracker_losses_readout`): four global Flash decoder layers, 256 slots, grid-diverse aggregate proposals, supervised hard hit grouping, and geometry plus calorimeter/tracker-neighborhood features. Decoder and inputs match `cld_pf_heptv2_default_50k` with `tracker_momentum_readout: true`.

In the matched 50k campaign, the head improved charged width from 12.78% to 12.40%, tails from 3.69% to 3.38%, and jet IQR from 0.10316 to 0.09815. Its +0.15 percentage-point F1 change has a paired 95% interval spanning zero; the tail reduction interval excludes zero. It is the best-supported benchmark candidate, not a multi-seed replacement for the conservative default. Sparse pooling, momentum/doublet inputs, fitted readouts, balanced CE, and auxiliary fit/overlap losses stay off. The loss campaign only has matched 32k results; combining balanced CE with the head is untested. CLIC and attention transfer of this recipe are new measurements, not previously validated wins.

## Model-size scaling: `cld_pf_heptv2_model_scaling_50k`

Width and depth are scaled together, keeping the 154 input channels of the previous baseline (geometry, tracker and calorimeter neighborhoods); tracker momentum and doublet features are disabled.

| Variant | Embedding / encoder width | Backbone layers | Decoder layers | Trainable parameters |
|---|---:|---:|---:|---:|
| `hit_heptv2_small` | 256 | 6 | 4 | 12.7M |
| `hit_heptv2_medium` | 384 | 9 | 6 | 42.1M |
| `hit_heptv2_large` | 512 | 12 | 8 | 98.8M |

Head counts (16 backbone, 8 decoder), 256 slots, proposals, grouping, and objective are unchanged; activation checkpointing is on for all sizes. All runs use global batch 128, 50k steps, LAMB at LR 0.001 with cosine decay, and validation every 1,000 steps (6.4M event presentations). This is a joint width/depth comparison, not an ablation of each.

Outcome (matched 42k measurements, 21 October archive): the small model remains the practical reference; the large model improves jets but worsens inclusive charged momentum. The later [width-only study](tracker_width_scaling.md) separates width from depth.

## Grouped tracking follow-up: `cld_pf_heptv2_grouped_tracking_50k`

Five runs (seed 12345, global batch 64, LR 0.001, 50k steps), each changing one mechanism from the grouped HEPTv2 baseline:

| Variant suffix | Change from `hit_heptv2_grouped_baseline` |
|---|---|
| `tracker_features` | Tracker-neighborhood input features |
| `balanced` | Tracker and calorimeter grouping losses averaged with equal detector weight |
| `tracker_readout` | Predicted-group geometric summaries and circle-fit momentum anchors |
| `event_consistency` | Visible event momentum/energy consistency loss, weight 0.1 |

The `tracker_features` variant became the current default.

**Detector balancing** keeps the background weight of 0.1 within each detector, normalizes over the local batch, and averages only detectors present in the batch. It does not rebalance individual particles.

**Tracker readout.** For each predicted group, a transverse circle is fit with three Huber-reweighted least-squares iterations in double precision. Nine summaries (hit and surface counts, radial and longitudinal spans, transverse scale, fitted log-pT, relative residual, conditioning, validity flag) feed a zero-initialized slot projection. A fit is valid with at least four hits on three distinct surfaces, 100 mm radial span, Gram-matrix eigenvalue ratio above 1e-7, radial RMS residual below 2% of the span, and pT in (0.1, 10,000) GeV. It assumes unit charge and the 2 T CLD field (`tracker_group_magnetic_field`), so it is a geometric prior, not a material-aware track fit. Valid fits supply log-pT and a massless energy estimate to the decoder's residual head; `metrics/set_slot/tracker_fit_fraction` and `tracker_fits_per_event` count them before the charged-PID gate. The later [charged fit correction](charged_fit_correction.md) bounds the learned correction to the fit.

**Event consistency** sums the final slots' visible px, py, pz, and energy weighted by presence probability, normalizes momentum residuals by the target scalar momentum sum and energy by total energy (both floored at 1 GeV), and adds smooth-L1 terms outside automatic task weighting. It is a weak auxiliary objective; judge it by jet widths, closure, efficiencies, and count bias rather than total loss.

**Validation probes.** The scenario enables `validation_linear_probe` (16 events per rank): fresh calorimeter-continuation rankers on raw features and embeddings, fit and evaluated on disjoint events (see the [continuation probe](calo_continuation_probe.md)), and tracker/calorimeter instance ARI with 8 fitting, 4 calibration, and 4 evaluation events per rank.

## Related ablations

| Scenario | Note |
|---|---|
| `cld_pf_heptv2_tracker_readout_50k` | [Tracker momentum readout and sparse association](tracker_momentum_readout.md) |
| `cld_pf_heptv2_fit_loss_50k` | [Tracker-fit supervision](tracker_fit_loss_experiments.md) |
| `cld_pf_heptv2_fit_correction_50k` | [Charged fit correction](charged_fit_correction.md) |
| `cld_pf_heptv2_tracker_grouping_losses_50k` | [Tracker grouping losses](tracker_grouping_loss_experiments.md) |
| `cld_pf_heptv2_tracker_width_scaling_50k` | [Tracker width scaling](tracker_width_scaling.md) |
| `cld_pf_heptv2_batch512_50k` | [HEPTv2 activation checkpointing](heptv2_checkpointing.md) |
