# ColliderML tracks and clusters

ColliderML Release 1 supplies tracks and spatial calorimeter clusters for top-pair events. This is a research workflow with dataset version **1.3.0**; it has no PF baseline in the local detector comparison.

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

ColliderML uses one sparse bipartite match for tracks and calorimeter clusters.
Each particle gets at most one input; each input gets at most one particle.
Track edges require more than half of **all** track hits, including hits without
selected truth ownership in the denominator, and score `1 + hit_fraction`.
Cluster edges score `2 * shared_deposit / (cluster_energy + particle_deposit)`
using calibrated deposited energies. Each particle has a private unmatched
choice with score zero. The solver maximizes the total score, with reliable
tracks preferred over clusters. Equal-score duplicate tracks may be selected
in a different order.

After matching, hit assignment and merging use the shared allocator. An unmatched
particle's full truth four-vector is added to the target of its largest-deposit
cluster. The host keeps its particle type, charge, and pileup label. If that
cluster has no assigned target, the particle is dropped and its energy is
reported in the accounting check. Merging does not use an angular cut or search
other clusters. This can still produce large cluster-to-target direction tails.

CLD, CLIC, IDEA, MAIA, and CMS keep their existing greedy or native assignment.
The shared allocator defaults to greedy; only the ColliderML converter selects
bipartite matching. Alternative assignments in the study scripts are comparisons,
not production settings.

New converted files carry `mlpf.target_assignment=bipartite-v1` in their Parquet
metadata. Resume reconverts older files, including complete greedy outputs.
Rebuild ColliderML TFDS as clustered version **1.3.0** or hits version **1.1.0**;
the old versions in the published catalog retain their original targets.
The recipe and builder defaults select the new versions. They must be generated
before training; their availability is not implied by the published catalog.

## Greedy and joint matching on PU0 and PU200

`scripts/studies/compare_colliderml_matching.py` compares four assignments on the same original events and published calorimeter clustering: the former energy-ordered greedy rule, majority tracks with energy-ordered greedy clusters, greedy matching on the joint ownership scores, and the exact joint solver. All four use the shared allocator for subsequent hit assignment, merging, and drop accounting. An explicit `input_assignment` lets the allocator accept the chosen matches without repeating its greedy assignment stage.

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

Each worker writes per-event metrics and final target directions. `summary.csv` aggregates counts and ownership measurements, and `source_manifest.json` records the sources and execution settings. Generate the plots and paired event-bootstrap intervals with:

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

The track improvement comes from majority eligibility. To isolate the solver, compare it with greedy matching on identical ownership edges and scores. On PU200, the solver increased the objective by 0.274% and truth-jet recall by 1.38 percentage points (95% paired event-bootstrap interval 0.67 to 2.06 points). On PU0, the corresponding recall difference was 0.21 points, with an interval of -0.21 to 0.69 points. Mean cluster purity is slightly lower for the solver than for same-score greedy matching because the objective sums overlap across all matches rather than maximizing average purity.

Median exclusive assignment times with eight workers active were 0.117 s for former greedy, 0.629 s for same-score greedy, and 0.587 s for the solver on PU200. PU0 times were 0.00134, 0.00597 and 0.00087 s respectively. These exclude common truth/matrix preparation and subsequent merging and describe these implementations on this machine.

The large PU200 high-pT cluster improvement comes primarily from changing the cluster ownership score; the solver gives an additional improvement. It does not remove the overall calorimeter direction tail. Target identities and merging change between rules, so object-level rates have different target populations. The bootstrap resamples events within this shard and does not measure variation between production shards. These results evaluate target construction, not retrained model performance.
