# IDEA tracks and clusters

IDEA uses measured GGTF/GenFit tracks and dual-readout calorimeter clusters. Its oracle candidate reference does not establish reconstructed-physics performance.

## Physics samples

| Physics sample | TFDS dataset | Recipe version |
|---|---|---|
| Top-quark pairs | `idea_edm_ttbar_pf` | 0.4.0 |
| Fully hadronic WW | `idea_edm_ww_fullhad_pf` | 0.4.0 |
| Inclusive quark pairs | `idea_edm_qq_pf` | 0.4.0 |

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
and test selections use **0.4.0**. This version selects `FittedTracks` at the IP,
rejects invalid fits, and derives training truth links from tracker-hit
associations. Tracks retain their measured kinematics; truth is used only for
labels. At least 75% of the distinct measurements must vote for one direct,
charged status-1 particle to give a track a representative label. Secondary,
fake and low-purity tracks remain inputs without an ancestor-momentum label.

Regenerate **Parquet and TFDS**. Updated ROOT files that already contain
`FittedTracks` and tracker-hit simulation links can be reused. Published 0.2.1
ROOT files and proxy-only reconstruction cannot provide these inputs. There is
no automatic fallback to proxy tracks, and the fitted-track builder rejects
unmarked or proxy Parquet files. Do not relabel old dataset directories.

Quality selection requires finite parameters, one IP state, nonnegative chi2,
positive NDF and associated hits. The innermost radius comes from measured hits.
The dEdx fields are missing-value zeros; the truth-parametrized dN/dx estimator
is deliberately excluded. These cuts remove broken fit records; they do not
eliminate the large momentum tails observed in the small tracking study.

For production, review `productions.idea` in `particleflow_spec.yaml` and the builders under `mlpf/heptfds/idea_pf_edm4hep/`, then follow [dataset generation](generate.md). The [detector-comparison runbook](../validation/detector-comparison.md) preserves the separate historical sample inventory and its limitations.

## Local production and validation

The `idea-upstream-2026-11` branch pins upstream components checked on
16 November 2026 and preserves the local dual-readout truth patches. The
fitted-track production uses workspace `IDEA_o1_v03_upstream_20261116_fitted` to
avoid reusing cached proxy Parquet. See the [generator instructions](../../mlpf/data/key4hep/gen/idea/README.md).

The generator retains proxy collections for diagnostics, but the default MLPF
reader does not load them. For explicit legacy diagnostics only, use
`--idea-track-source truth_seeded` and read the resulting Parquet with
`expected_track_source="truth_seeded"`. TFDS versions below 0.4.0 retain legacy
proxy metadata; 0.4.0 requires `track_source="ggtf_genfit"` and matching quality
and truth-policy metadata. The pipeline-preparation helper preserves provenance
and requires `--legacy-proxy` for old unmarked proxy inputs.

On three ttbar events, the fitted path kept **21, 11, 15** tracks, rejecting
three invalid fits. Hit-majority matching labelled **18, 10, 12** tracks as direct
primary representatives. The local 0.4.0 TFDS smoke dataset built and read both
splits successfully. This validates integration, not tracking efficiency or PF
performance. `ycand` remains a target oracle, and `suitable_for_physics` remains
false. The published 0.2.1 data and the upstream-update 0.3.0 recipe are proxy
campaigns, not substitutes for the new fitted-track dataset.
