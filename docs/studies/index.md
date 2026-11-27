# Research notes

These pages record individual studies, mostly of the CLD raw-hit particle-set model: what was tried, how it was configured, and what was measured. Each note is dated by its evidence and tied to the code, data versions, and checkpoints it used. They are not maintained user guides; options described here stay off by default unless the [CLD hit-model evidence](cld_hit_results.md) recommends them.

Start with the [CLD hit-model evidence](cld_hit_results.md), which summarizes the current recommendation (`pyg-cld-hits-heptv2-set-v1`) and links the studies below.

## Training campaigns

| Note | Topic |
|---|---|
| [CLD hit-model campaigns](cld_hit_campaigns.md) | The 50k scenarios behind the current default: promotion, model-size scaling, grouped-tracking mechanisms |
| [CMS training exposure](cms_training_exposure.md) | Why the CMS scenarios use 96/768-event batches and longer training |

## Model options

| Note | Topic |
|---|---|
| [Tracker momentum features](tracker_momentum_features.md) | Optional curvature, longitudinal, and doublet features for tracker hits |
| [Tracker momentum readout](tracker_momentum_readout.md) | Tracker-only momentum head and sparse hit-to-slot pooling |
| [Charged fit correction](charged_fit_correction.md) | Circle-fit charged pT with a bounded learned correction |
| [HEPTv2 activation checkpointing](heptv2_checkpointing.md) | Recomputing backbone activations to reduce training memory |
| [HEPTv2 and Attention on ColliderML PU0](heptv2_colliderml_pu0.md) | L40S replay of coordinate reuse, padding compaction, and batch scaling |
| [Reconstruction diagnostics](reconstruction_diagnostics.md) | Optional hit-grouping and momentum diagnostics during validation |

## Ablations and scaling

| Note | Topic |
|---|---|
| [Tracker grouping losses](tracker_grouping_loss_experiments.md) | Particle-balanced and overlap supervision of tracker hit groups |
| [Tracker-fit supervision](tracker_fit_loss_experiments.md) | Training-only supervision of learned ownership from track fits |
| [Tracker width scaling](tracker_width_scaling.md) | Four model widths, three seeds each |
| [Frozen tracker readout](frozen_tracker_readout.md) | Head-only hard, soft, and oracle momentum pooling on a frozen backbone |

## Probes

| Note | Topic |
|---|---|
| [Hit instance probe](hit_instance_probe.md) | Linear probes of particle identity in raw features and HEPTv2 embeddings |
| [Calorimeter continuation probe](calo_continuation_probe.md) | Cross-layer shower continuation in ambiguous neighborhoods |
