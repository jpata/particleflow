# IDEA tracks and clusters

IDEA is a pipeline-validation workflow using truth-seeded proxy tracks and calorimeter clusters. Its oracle reference does not establish reconstructed-physics performance.

## Physics samples

| Physics sample | TFDS dataset | Recipe version |
|---|---|---|
| Top-quark pairs | `idea_edm_ttbar_pf` | 0.3.0 |
| Fully hadronic WW | `idea_edm_ww_fullhad_pf` | 0.3.0 |
| Inclusive quark pairs | `idea_edm_qq_pf` | 0.3.0 |

The `pyg-idea-pipeline-v1` recipe uses all three samples with configuration **1**. The [catalog](catalog.md#published-availability) records their public availability.

## Event view

These published views are from the previous campaign; the upstream-update
branch has not regenerated them. They show the same simulated `ttbar` event (index 0) at 365 GeV in the transverse plane. Tracks are truth-seeded, and the particle panel shows visible stable truth particles.

| Truth-seeded tracks and clusters | Visible stable truth proxy |
|:---:|:---:|
| ![IDEA: truth-seeded tracks and clusters for event 0.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_tracks_clusters.png) | ![IDEA: visible stable truth proxy for the same event.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_particles.png) |

[All-component overlay](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_overlay.png),
[vector input view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_tracks_clusters.svg) and
[vector particle view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_particles.svg) are available for reuse.
See [event-view conventions](event-displays.md) for particle sources, sampling and scale.

## Builder and recipe compatibility

The builders, production mapping, and `pyg-idea-pipeline-v1` training, validation,
and test selections use **0.3.0**. Use data built with this version, which
uses the November 2026 upstream IDEA simulation/reconstruction with local
dual-readout truth patches. Regenerate simulation and downstream data for this
version; geometry, tracking selection, clustering and truth-link behavior changed. Do not relabel
an older dataset directory to satisfy the recipe.

For production, review `productions.idea` in `particleflow_spec.yaml` and the builders under `mlpf/heptfds/idea_pf_edm4hep/`, then follow [dataset generation](generate.md). The [detector-comparison runbook](../validation/detector-comparison.md) preserves the separate historical sample inventory and its limitations.

## Local upstream-update branch

The `idea-upstream-2026-11` branch updates the IDEA component pins to upstream
HEADs checked on 16 November 2026 and ports the MLPF Cherenkov truth and
per-channel cluster-energy patches. Its new production workspace is
`IDEA_o1_v03_upstream_20261116`; it does not reuse the previous simulation campaign.
See the [generator instructions](../../mlpf/data/key4hep/gen/idea/README.md)
for builds, stack selection and smoke checks.

The generator runs GGTF/GenFit reconstruction by default, together with the
truth-seeded tracks still consumed by MLPF postprocessing. The **0.3.0 training
representation remains a proxy-track workflow**. Fitted-track truth matching and
reader support are separate work; there is still no reconstructed-PFO baseline.
The published 0.2.1 dataset remains the older campaign. Version 0.3.0 requires a
new production and is not established as published by this branch.

The local electron smoke test completed simulation, reconstruction and loading
into MLPF tensors, but returned zero GGTF track candidates at the upstream
thresholds. It therefore did not exercise a successful GenFit fit. See the
generator validation notes before drawing conclusions about tracking performance.
