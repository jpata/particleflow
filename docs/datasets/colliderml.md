# ColliderML tracks and clusters

[ColliderML](https://huggingface.co/datasets/CERN/ColliderML-Release-1) Release 1 is an open proton-proton simulation with ACTS tracks and raw calorimeter hits. The MLPF converter clusters the calorimeter hits (the datasets use CLUE, `--algorithm clue`; the converter default is a radius graph) and builds particle targets from the simulation truth. This is a research workflow at dataset version **1.5.0**; there is no rule-based PF reference to compare against.

## Physics samples

| Physics sample | Pileup | TFDS dataset | Model recipe |
|---|---|---|---|
| Top-quark pairs | None | `colliderml_ttbar_nopu_pf` | `pyg-colliderml-v1` |
| Top-quark pairs | Approximately 200 interactions | `colliderml_ttbar_pu200_pf` | `pyg-colliderml-pu200-v1` |

`pyg-colliderml-pu0pu200-groups-v1` combines both in separate groups with independent batch sizes and step weights; see [dataset mixing](../training/data-mixing.md).

Builders also exist for the other Release-1 samples, with the same schema: `dihiggs`, `ggf`, `zprime`, `hidden_valley`, `higgs_portal` (no pileup), `dihiggs_pu200`, `ggf_pu200`, `zee_pu200`, `zmumu_pu200`, and `pileup_only`. They are not in the production mapping or any training recipe.

The Hub has the two ttbar samples only at version 1.4.0. The recipes expect 1.5.0, which must be produced locally (see the [catalog](catalog.md#published-availability)).

## Event view

These views show the same simulated no-pileup `ttbar` event (index 0) in the transverse plane. Inputs are ACTS tracks and spatial calorimeter clusters. The particle panel shows visible primary-leaf truth; allocated training targets are defined by postprocessing.

| Tracks and clusters | Primary-leaf truth proxy |
|:---:|:---:|
| ![ColliderML: tracks and clusters for event 0.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/colliderml/event_0_tracks_clusters.png) | ![ColliderML: primary-leaf truth proxy for the same event.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/colliderml/event_0_particles.png) |

[All-component overlay](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/colliderml/event_0_overlay.png),
[vector input view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/colliderml/event_0_tracks_clusters.svg) and
[vector particle view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/colliderml/event_0_particles.svg) are available for reuse.
See [event-view conventions](event-displays.md) for particle sources, sampling and scale.

## Production

`productions.colliderml` in `particleflow_spec.yaml` defines the source revision, download and workspace roots, and TFDS mappings. The pipeline is: download Release-1 shards → convert to MLPF Parquet with `mlpf/data/colliderml/postprocessing.py` → build TFDS with the builders under `mlpf/heptfds/colliderml_pf/`. Validate converted files with `tests/validate_parquet.py --detector colliderml`.

At Flatiron, `scripts/flatiron/colliderml_process_samples.sh` submits the whole chain as dependent Slurm jobs per sample (download, conversion of all 1,000 shards, one TFDS build per configuration). Every step is resumable; run it with `--dry-run` first. Downloads are deliberately bandwidth-limited.

For a local check, `scripts/local_test_colliderml.sh` converts an offline synthetic fixture and trains two CPU steps, and `scripts/local_test_colliderml_real.sh` converts and trains on 1,000 real events; see the [detector-comparison runbook](../validation/detector-comparison.md). [ColliderML hit datasets](colliderml-hits.md) use separate builders and recipes.

## Target assignment

ColliderML assigns tracks and calorimeter clusters to particles by bipartite
matching: each particle gets at most one input, and each input at most one
particle. The converter default is `majority_energy`, a bipartite matching with
an energy score:

1. Tracks: a track is eligible only for its majority owner, the particle holding
   more than half of **all** its hits (hits without selected truth ownership count
   in the denominator). A particle with several eligible tracks takes the purest.
2. Clusters: the particles without a track share the clusters through one sparse
   maximum-weight bipartite matching with edge score `particle_energy + shared_deposit`
   (calibrated GeV). Each particle has a private unmatched choice with score zero.
   The score keeps energetic particles hosted, and a claim can be rerouted to
   another of the particle's clusters when that lets a second particle keep one.

`--assignment greedy` reproduces the former energy-ordered rule. The former
`bipartite` overlap-score option has been removed in favor of `majority_energy`.

After matching, hit assignment and merging use the shared allocator. An unmatched
particle's full truth four-vector is added to the target of its largest-deposit
cluster. The host keeps its particle type and charge; its `ispu` becomes the
pileup share of the merged energy, `sum(E_i * ispu_i) / sum(E_i)`. If that
cluster has no assigned target, the particle is dropped and its energy is
reported in the accounting check. Merging does not use an angular cut or search
other clusters. This can still produce large cluster-to-target direction tails.

ColliderML and MAIA use `majority_energy`. CLD, CLIC, IDEA, and CMS keep their
greedy or native assignment; the shared allocator defaults to greedy.

New converted files carry `mlpf.target_assignment=<rule>-v1` (for example
`majority_energy-v1`) in their Parquet metadata. Resume reconverts files written
by another rule or without the tag, including complete greedy outputs.
Files also carry `mlpf.pileup_truth=energy-fraction-v1` and
`mlpf.forward_truth=drop-eta3.5-v1`; resume reconverts older files lacking these
tags so that pileup fields and target semantics stay consistent.

## Out-of-acceptance particles

A particle beyond |eta| = 3.5 passes through the bore of the endcap
calorimeters and never enters them, but its Geant4 secondaries, born outside the
calorimeter, can still deposit in the endcap. Such a particle is not
a target, and the deposits of it and its secondaries stay unattributed. The cut
is `OUT_OF_ACCEPTANCE_ABS_ETA` in `mlpf/data/colliderml/truth.py`;
`drop_out_of_acceptance` switches it off.

## Pileup truth

Clustered TFDS version **1.4.0** adds the release `event_id`, `pufrac` in input
order (tracks, then clusters), and `genmet_withpu` / `genjets_withpu`. Track
pileup fractions count hits; cluster fractions weight calibrated deposited energy.
These fractions describe the input even when it has no assigned target.
`genmet` / `genjets` use only measurable hard-scatter truth; the `*_withpu`
fields retain all collisions. Both clustered and hits targets carry fractional
`ispu` after mixed-origin merges; training binarizes it at `>= 0.5`.

## Version history

| Clustered | Hits | Change |
|---|---|---|
| 1.5.0 | 1.3.0 | Particles beyond \|eta\| = 3.5 are no longer targets |
| 1.4.0 | 1.2.0 | Pileup truth: fractional `ispu`; clustered adds `pufrac`, `genjets_withpu`, `genmet_withpu` |
| 1.3.0 | 1.1.0 | `majority_energy` target assignment |
| 1.2.0 | | Surrogate ancestors in the truth, as for Key4HEP |
| 1.1.0 | | CLUE clustering |

Versions change the targets, not just the metadata: regenerate rather than relabel.

## Comparing assignment rules

`scripts/studies/compare_colliderml_matching.py` rebuilds targets for the same events with the former greedy rule, majority-owned tracks with greedy clusters, and `majority_energy`, and reports track and cluster direction tails, cluster purity, merged and dropped energy, and target-jet recall against the truth jets. `scripts/studies/plot_colliderml_matching.py` makes the plots and paired event-bootstrap intervals. On the first 100 events of shard 0, `majority_energy` reduced the PU200 high-pT cluster direction tail (ΔR > 0.3) from 60% to 16% and the merged hard-scatter energy by half relative to greedy, and raised PU200 truth-jet recall from 27% to 33%. These numbers describe target construction, not a trained model.
