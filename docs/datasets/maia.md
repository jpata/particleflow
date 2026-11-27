# MAIA tracks and clusters

MAIA provides a track/cluster workflow for a muon-collider detector study. Conversion, TFDS building, and local CPU training are implemented, but data validation retains failed gates and incomplete detector metadata.

## Physics sample

| Physics sample | TFDS dataset | Schema version | Model recipe |
|---|---|---|---|
| Top-quark pairs | `maia_edm_ttbar_pf` | 1.2.0 | `pyg-maia-v1` |

The builder defines ten configuration partitions; the recipe trains, validates, and tests on configuration 10 of the local smoke sample. MAIA data are not [published](catalog.md#published-availability).

## Event view

These views show the same simulated `ttbar` event (index 0) in the transverse plane. The particle panel shows the Pandora reference reconstruction from the local MAIA sample.

| Tracks and clusters | Pandora reconstructed particles |
|:---:|:---:|
| ![MAIA: tracks and clusters for event 0.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/maia/event_0_tracks_clusters.png) | ![MAIA: pandora reconstructed particles for the same event.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/maia/event_0_particles.png) |

[All-component overlay](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/maia/event_0_overlay.png),
[vector input view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/maia/event_0_tracks_clusters.svg) and
[vector particle view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/maia/event_0_particles.svg) are available for reuse.
See [event-view conventions](event-displays.md) for particle sources, sampling and scale.

## Detector-specific processing

MAIA differs from CLD and CLIC in four postprocessing choices, all enabled only for MAIA:

- **Muon-system truth.** MAIA's calorimeter truth links cover only ECAL and HCAL, so muon-system hits are linked to particles through `RelationMuonHit` and the yoke simulation hits.
- **Track calorimeter entry.** Tracks carry `eta_ecal`/`phi_ecal`, the track state at the calorimeter seen from the origin, in otherwise unused columns; the input width stays 19.
- **Cluster direction.** Cluster eta, phi, and ET are taken from the centroid position rather than Pandora's shower axis.
- **Targets.** Tracks and clusters are assigned to particles with the `majority_energy` bipartite matching described for [ColliderML](colliderml.md#target-assignment).

All four were introduced in version 1.2.0.

## Production and validation

Review `productions.maia` in `particleflow_spec.yaml` before following [dataset generation](generate.md). MAIA has no generator here: the production converts existing EDM4hep files placed as `<workspace_dir>/gen/ttbar/root/reco_ttbar_<seed>.root`, with `seed_range` set to the seeds present, and `PROD=maia pixi run tfds_full` builds the general dataset from at least two files. The builder is under `mlpf/heptfds/maia_pf_edm4hep/`. The [general dataset](full.md) `maia_edm_ttbar_full` (builder `mlpf/heptfds/maia_full/`) adds the raw tracker and calorimeter hits with truth labels for the `pyg-maia-hits-heptv2-set-v1` recipe. `scripts/local_test_maia.sh` converts the local sample, builds TFDS, and trains for two CPU steps. The script continues past failed Parquet checks; the sample still fails some gates (tracker-hit momentum, calorimeter-hit energy, baseline particle energy); the supplied ROOT file contains only 10 events. A passing smoke run does not resolve these data limitations; see the [detector-comparison runbook](../validation/detector-comparison.md).
