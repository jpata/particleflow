# ColliderML tracks and clusters

ColliderML Release 1 supplies tracks and spatial calorimeter clusters for top-pair events. This is a research workflow with dataset version **1.5.0**; it has no PF baseline in the local detector comparison.

## Physics samples

| Physics sample | Pileup | TFDS dataset | Model recipe |
|---|---|---|---|
| Top-quark pairs | None | `colliderml_ttbar_nopu_pf` | `pyg-colliderml-v1` |
| Top-quark pairs | Approximately 200 interactions | `colliderml_ttbar_pu200_pf` | `pyg-colliderml-pu200-v1` |

`pyg-colliderml-pu0pu200-groups-v1` combines these samples in separate groups with independent batch sizes and optional step weights. See the runnable [dataset-mixing example](../training/data-mixing.md).

Both samples have earlier configuration-1 data in the [catalog](catalog.md#published-availability). Current recipes configure partitions 1--10 and require regenerated targets at the versions described below.

## Event view

These views show the same simulated no-pileup `ttbar` event (index 0) in the transverse plane. Inputs are ACTS tracks and spatial calorimeter clusters. The particle panel shows visible primary-leaf truth; allocated training targets are defined by postprocessing.

| Tracks and clusters | Primary-leaf truth proxy |
|:---:|:---:|
| ![ColliderML: tracks and clusters for event 0.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/colliderml/event_0_tracks_clusters.png) | ![ColliderML: primary-leaf truth proxy for the same event.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/colliderml/event_0_particles.png) |

[All-component overlay](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/colliderml/event_0_overlay.png),
[vector input view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/colliderml/event_0_tracks_clusters.svg) and
[vector particle view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/colliderml/event_0_particles.svg) are available for reuse.
See [event-view conventions](event-displays.md) for particle sources, sampling and scale.

## Production and validation

`productions.colliderml` in `particleflow_spec.yaml` defines the source revision, download and workspace roots, postprocessing, and TFDS mappings. The converter supports radius-graph and CLUE clustering and surrogate-ancestor truth targets. Follow [dataset generation](generate.md) with the reviewed source and workspace paths, and use the shared Parquet validator with `--detector colliderml`.

The [detector-comparison runbook](../validation/detector-comparison.md) records the real-data training check and feature-validation limits. [ColliderML hit datasets](colliderml-hits.md) use separate names and recipes.

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

CLD, CLIC, IDEA, MAIA, and CMS keep their existing greedy or native assignment.
The shared allocator defaults to greedy; only the ColliderML converter selects
`majority_energy`. The other assignments in the study scripts are
comparisons, not production settings.

New converted files carry `mlpf.target_assignment=<rule>-v1` (for example
`majority_energy-v1`) in their Parquet metadata. Resume reconverts files written
by another rule or without the tag, including complete greedy outputs.
Files also carry `mlpf.pileup_truth=energy-fraction-v1` and
`mlpf.forward_truth=replace-eta3.5-v1`; resume reconverts older files lacking these
tags so that pileup fields and target semantics stay consistent.
Rebuild ColliderML TFDS as clustered version **1.5.0** or hits version **1.3.0**;
the old versions in the published catalog retain their original targets.
The recipe and builder defaults select the new versions. They must be generated
before training; their availability is not implied by the published catalog.

## Out-of-acceptance particles

A particle beyond |eta| ≈ 3.4 passes through the bore of the endcap
calorimeters and never enters them, but its Geant4 secondaries (mostly from
beam-pipe interactions) can still deposit in the endcap. Such a particle is
replaced as a target by its direct Geant4 children, repeated while a child is
itself beyond the cut, so the targets are the particles that entered the
detector and the deposits stay attributed. The cut is
`OUT_OF_ACCEPTANCE_ABS_ETA` in `mlpf/data/colliderml/truth.py`;
`replace_out_of_acceptance` switches it off.

## Pileup truth

Clustered TFDS version **1.4.0** adds the release `event_id`, `pufrac` in input
order (tracks, then clusters), and `genmet_withpu` / `genjets_withpu`. Track
pileup fractions count hits; cluster fractions weight calibrated deposited energy.
These fractions describe the input even when it has no assigned target.
`genmet` / `genjets` use only measurable hard-scatter truth; the `*_withpu`
fields retain all collisions. Both clustered and hits targets carry fractional
`ispu` after mixed-origin merges; training binarizes it at `>= 0.5`.

## Greedy and joint matching on PU0 and PU200

`scripts/studies/compare_colliderml_matching.py` compares three assignments on the same original events and published calorimeter clustering: the former energy-ordered greedy rule, majority tracks with energy-ordered greedy clusters, and `majority_energy`. All three use the shared allocator for subsequent hit assignment, merging, and drop accounting. An explicit `input_assignment` lets the allocator accept the chosen matches without repeating its greedy assignment stage.

The results below are historical measurements, including the removed overlap-score matcher (`bipartite`, also called the joint solver). The current study and plotting scripts compare `majority_energy` against the two greedy baselines; rerunning them does not reproduce the removed matcher columns or the historical overlap-score bootstrap intervals.

The comparison used the first 100 events of shard 00000 for each sample. The former greedy rule reproduced the published track and cluster targets on every event. Eight CPU workers ran outside the sandbox with one numerical thread per worker; the complete comparison took 147 seconds. Example command using the locally downloaded sources:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
PYTHONPATH=. uv run python3 scripts/studies/compare_colliderml_matching.py \
  --pu0-source local_test_data/detector_comparison/colliderml_source \
  --pu200-source /tmp/colliderml-original-pu200 \
  --pu0-processed /tmp/colliderml-validation/pu0/train-00000-of-01000.parquet \
  --pu200-processed /tmp/colliderml-validation/pu200/train-00000-of-01000.parquet \
  --output experiments/colliderml_matching_comparison --events 100 --workers 8
```

Each worker writes per-event metrics and final target directions. `summary.csv` aggregates counts and ownership measurements, and `inputs.json` records the input paths and execution settings. Generate the plots and paired event-bootstrap intervals with:

```bash
uv run python3 scripts/studies/plot_colliderml_matching.py \
  --input experiments/colliderml_matching_comparison
```

The following results compare the former greedy rule with the joint solver. Direction measurements include merges; high-pT cluster measurements require target pT >= 5 GeV. Jet recall compares target jets with the published truth reference: anti-kt R=0.4, pT > 3 GeV, one-to-one matching within delta-R < 0.1, including pileup.

| Metric | PU0 greedy | PU0 solver | PU200 greedy | PU200 solver |
|---|---:|---:|---:|---:|
| Track delta-R > 0.1 | 0.66% | 0.51% | 16.99% | 0.63% |
| Cluster delta-R > 0.3 | 36.40% | 36.74% | 48.75% | 48.77% |
| High-pT cluster delta-R > 0.3 | 5.22% | 6.10% | 60.41% | 40.69% |
| Mean cluster deposit purity | 75.33% | 77.62% | 50.57% | 56.08% |
| Truth-jet recall | 95.72% | 96.75% | 26.52% | 29.97% |
| Merged fraction of selected particles | 7.05% | 5.59% | 24.52% | 22.79% |
| Dropped truth energy across the sample | 0 GeV | 0 GeV | 110.49 GeV | 91.30 GeV |

### Overlap score (`bipartite`) and energy score (`majority_energy`)

The same 100 events per sample, with the former greedy rule, removed overlap-score matcher, and current default side by side. Merged energy sums the truth energy of particles merged into another target; jet response uses matched reference jets with pT > 20 GeV.

| Metric | PU0 greedy | PU0 bipartite | PU0 majority_energy | PU200 greedy | PU200 bipartite | PU200 majority_energy |
|---|---:|---:|---:|---:|---:|---:|
| Track targets | 6,086 | 6,086 | 6,086 | 176.7k | 151.2k | 151.2k |
| Track ΔR > 0.1 | 0.66% | **0.51%** | **0.51%** | 16.99% | **0.63%** | **0.63%** |
| Hard-scatter track ΔR > 0.1 | 0.66% | **0.51%** | **0.51%** | 8.58% | **0.48%** | 0.50% |
| Cluster targets | 15.85k | 16.20k | 16.38k | 928.2k | 979.1k | 1,036.5k |
| Mean cluster purity | 75.3% | **77.6%** | 75.4% | 50.6% | **56.1%** | 50.9% |
| High-pT cluster ΔR > 0.3 | **5.2%** | 6.1% | 6.5% | 60.4% | 40.7% | **16.0%** |
| Merged fraction of selected particles | 7.05% | 5.59% | **4.82%** | 24.52% | 22.79% | **18.87%** |
| Merged energy | 1.51 TeV | 1.79 TeV | **0.65 TeV** | 456 TeV | 1,026 TeV | **284 TeV** |
| Merged hard-scatter energy | 1.51 TeV | 1.79 TeV | **0.65 TeV** | 4.3 TeV | 7.4 TeV | **2.2 TeV** |
| Dropped energy | **0 GeV** | **0 GeV** | **0 GeV** | 110.5 GeV | **91.3 GeV** | **91.3 GeV** |
| Truth-jet recall | 95.7% | 96.8% | **97.0%** | 26.5% | 30.0% | **32.5%** |
| Truth-jet precision | 96.8% | 97.6% | **97.8%** | 31.7% | 34.8% | **37.2%** |
| Matched jets, reference pT > 20 GeV | **671** | **671** | **671** | 3,635 | 3,820 | **4,081** |
| Jet response median / IQR (pT > 20 GeV) | **1.000** / 0.002 | **1.000** / **0.000** | **1.000** / **0.000** | 0.936 / **0.110** | **0.944** / 0.111 | 0.943 / **0.110** |

- **Rules:** greedy = former energy-ordered first-come rule; bipartite = joint bipartite matching, tracks require > 50% hit ownership, clusters scored by overlap 2·shared / (E_cluster + E_particle_deposit); majority_energy = the same majority-owned tracks, then a bipartite cluster matching scored E_particle + shared deposit.
- **ΔR metrics** compare final targets (after merging) with their track or cluster; "high-pT" means target pT ≥ 5 GeV.
- **Mean cluster purity** = mean over cluster targets of the host's shared deposit / cluster energy.
- **Jets** are anti-kt R=0.4, pT > 3 GeV, built from the targets and matched one-to-one within ΔR < 0.1 to the published `genjet` (pileup included).
- **Bold** marks the best value per pileup sample, with ties all bold; track and cluster target counts have no "better" direction and are not marked. For jet response, a median closer to 1 and a smaller IQR are better.

The track improvement comes from majority eligibility. To isolate the solver, compare it with greedy matching on identical ownership edges and scores. On PU200, the solver increased the objective by 0.274% and truth-jet recall by 1.38 percentage points (95% paired event-bootstrap interval 0.67 to 2.06 points). On PU0, the corresponding recall difference was 0.21 points, with an interval of -0.21 to 0.69 points. Mean cluster purity is slightly lower for the solver than for same-score greedy matching because the objective sums overlap across all matches rather than maximizing average purity.

Median exclusive assignment times with eight workers active were 0.117 s for former greedy, 0.629 s for same-score greedy, and 0.587 s for the solver on PU200. PU0 times were 0.00134, 0.00597 and 0.00087 s respectively. These exclude common truth/matrix preparation and subsequent merging and describe these implementations on this machine.

The large PU200 high-pT cluster improvement comes primarily from changing the cluster ownership score; the solver gives an additional improvement. It does not remove the overall calorimeter direction tail. Target identities and merging change between rules, so object-level rates have different target populations. The bootstrap resamples events within this shard and does not measure variation between production shards. These results evaluate target construction, not retrained model performance.
