# IDEA tracks and clusters

IDEA is a pipeline-validation workflow using truth-seeded proxy tracks and calorimeter clusters. Its oracle reference does not establish reconstructed-physics performance.

## Physics samples

| Physics sample | TFDS dataset | Recipe version |
|---|---|---|
| Top-quark pairs | `idea_edm_ttbar_pf` | 0.3.0 |
| Fully hadronic WW | `idea_edm_ww_fullhad_pf` | 0.3.0 |
| Inclusive quark pairs | `idea_edm_qq_pf` | 0.3.0 |

The `pyg-idea-pipeline-v1` recipe (HEPTv2) uses all three samples with configuration **1**. Version 0.3.0 is [published](catalog.md#published-availability).

## Event view

These views show the same simulated `ttbar` event (index 0) at 365 GeV in the transverse plane. Tracks are truth-seeded, and the particle panel shows visible stable truth particles.

| Truth-seeded tracks and clusters | Visible stable truth proxy |
|:---:|:---:|
| ![IDEA: truth-seeded tracks and clusters for event 0.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_tracks_clusters.png) | ![IDEA: visible stable truth proxy for the same event.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_particles.png) |

[All-component overlay](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_overlay.png),
[vector input view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_tracks_clusters.svg) and
[vector particle view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_particles.svg) are available for reuse.
See [event-view conventions](event-displays.md) for particle sources, sampling and scale.

## Version history

| Version | Change |
|---|---|
| 0.3.0 | Track phi, D0, and Z0 stored at the perigee relative to the origin, as for CLD and CLIC (previously at the production vertex for secondaries); `radiusOfInnermostHit` from the AtFirstHit track state |
| 0.2.1 | Cluster shower widths from the associated calorimeter cells |
| 0.2.0 | Muon-system cluster features; input width 17 → 19 |

Versions are not interchangeable: rebuild rather than relabel an older dataset directory. Input-normalization statistics exist for 0.1.0, 0.2.1, and 0.3.0.

## Production

Review `productions.idea` in `particleflow_spec.yaml` and the builders under `mlpf/heptfds/idea_pf_edm4hep/`, then follow [dataset generation](generate.md).
