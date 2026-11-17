# IDEA tracks and clusters

IDEA uses measured GGTF/GenFit tracks and dual-readout calorimeter clusters. Its oracle candidate reference does not establish reconstructed-physics performance.

## Physics samples

| Physics sample | TFDS dataset | Recipe version |
|---|---|---|
| Top-quark pairs | `idea_edm_ttbar_pf` | 0.4.1 |
| Fully hadronic WW | `idea_edm_ww_fullhad_pf` | 0.4.1 |
| Inclusive quark pairs | `idea_edm_qq_pf` | 0.4.1 |

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
and test selections use **0.4.1**. Versions 0.4.x select `FittedTracks` at the IP,
reject invalid fits, and derive training truth links from tracker-hit
associations. Tracks retain their measured kinematics; truth is used only for
labels. At least 75% of the distinct measurements must vote for one direct,
charged status-1 particle to give a track a representative label. Secondary,
fake and low-purity tracks remain inputs without an ancestor-momentum label.

For **0.4.1**, regenerate reconstruction, Parquet and TFDS. This release fixes
circle-seed orientation and writes GENFIT's fitted charge instead of the seed
hypothesis. Stored simulation or GGTF candidates can be reused for reconstruction;
unpatched 0.4.0 `FittedTracks` must be refitted. Published 0.2.1
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
fitted-track production uses workspace
`IDEA_o1_v03_upstream_20261116_fitted_trackingfix_20261117` to avoid reusing
cached proxy or unpatched fitted data. See the [generator instructions](../../mlpf/data/key4hep/gen/idea/README.md).

The generator retains proxy collections for diagnostics, but the default MLPF
reader does not load them. For explicit legacy diagnostics only, use
`--idea-track-source truth_seeded` and read the resulting Parquet with
`expected_track_source="truth_seeded"`. TFDS versions below 0.4.0 retain legacy
proxy metadata; 0.4.x requires `track_source="ggtf_genfit"` and matching quality
and truth-policy metadata. The pipeline-preparation helper preserves provenance
and requires `--legacy-proxy` for old unmarked proxy inputs.

Before the tracking fix (0.4.0), the 16 November validation generated
**100 ttbar events at 365 GeV** in four disjoint 25-event batches. The updated
ROOT files were postprocessed with both
proxy and fitted tracks. The reference was freshly postprocessed from the
published 100-event ROOT file in explicit legacy mode. These are not matched
generator events across campaigns. The unmodified `tests/validate_parquet.py`
checked all 100 events in each view:

| View | PASS | FAIL | Track direction within ΔR < 0.1 | Cluster direction within ΔR < 0.3 |
|---|---:|---:|---:|---:|
| Published ROOT, proxy tracks | 20 | 0 | 97.33% | 97.49% |
| Updated ROOT, proxy tracks | 19 | 1 | 90.48% | 93.90% |
| Updated ROOT, fitted tracks | 18 | 2 | 69.51% | 73.26% |

All three views also have seven warnings and one skipped baseline check. The
direction gates require at least 95% for tracks and 80% for clusters. The
updated proxy view already fails track alignment; the fitted view fails both
direction gates. The fitted path retained **1,543 tracks** from 1,628 stored
fits, rejecting 85 invalid fits, and assigned **1,335 primary representative
labels**. Its median input/truth pT ratio is 0.9984, but the range is
0.000069–99.18; 10.26% of labelled tracks point within ΔR < 0.1 of the opposite
truth direction. No event reached the configured tracking hit limit. These
results require further reconstruction and association studies before physics
use. Local reports, plots, ROOT manifests and Parquet files are under
`reports/idea-100-ttbar-20261116/`.

The 17 November follow-up traced the failures to native ROOT identities and
states. All 475 proxy direction failures are secondary tracks relabelled with
status-1 ancestor momenta; all proxies agree with their directly linked MC
particles. The fitted-track tail is already present in ROOT: 137 labelled
tracks are nearly reversed, 128 also have the wrong charge sign, and the
failure persists at high pT and high hit purity. Relaxing the track gate would
hide reconstruction failures.

The aggregate cluster gate has a different limitation. Original neutral-source
particles pass at 95.50%, while original charged-source particles represented
by clusters pass at 35.62% against their production directions. For the same
1,994 charged-source clusters with simulated endpoint radius above 1,800 mm,
using the transported endpoint direction instead raises alignment from 46.04%
to 94.58%. Cluster checks should distinguish original source charge and use a
transport reference for charged particles, rather than relax a universal cone.
The mapped cluster target charge is always zero and cannot supply that
distinction. Changing the representative to the largest attributed-energy
cluster only raises aggregate alignment from 73.26% to 73.85%. The detailed
study and figure are in `reports/idea-alignment-20261117/`; the original
validation thresholds and data remain unchanged.

One parallel reconstruction worker exited with code 137. Its complete 25-event
simulation was reconstructed again in isolation with the same settings,
successfully retaining all events. The isolated reconstruction peaked at
**28.59 GiB RSS**. The recipe now requests **64,000 MB** for generation and
reconstruction; avoid concurrent GGTF jobs on a 64 GB host.

The earlier three-event 0.4.0 TFDS smoke dataset built and read both splits
successfully. This validates integration, not tracking efficiency or PF
performance. `ycand` remains a target oracle, and `suitable_for_physics` remains
false. The published 0.2.1 data and the upstream-update 0.3.0 recipe are proxy
campaigns, not substitutes for the new fitted-track dataset.

## Tracking fix in 0.4.1

The local k4RecTracker patch orients the circle seed using the signed angular
progression of all ordered measurements. The old PCA-to-second-hit direction
could reverse the transverse seed in the presence of misassigned inner hits.
The output helix, covariance and calorimeter extrapolation now also use the
actual fitted charge; GENFIT can change q/p sign relative to the seed.

Refitting the same 3,046 GGTF candidates in the 100 simulated events gives:

| Quantity | Before | Patched |
|---|---:|---:|
| Accepted fitted inputs | 1,543 | 1,567 |
| Primary representative labels | 1,335 | 1,349 |
| Nearly reversed representatives | 137 (10.26%) | 3 (0.22%) |
| Direction within ΔR < 0.1 | 69.51% | 80.58% |
| Wrong charge sign | 239 | 108 |

The three development events went from five reversals to zero; the remaining
97 events went from 132/1,295 reversals to 3/1,308. Of the original 137 reversed
candidates, 113 now align, 13 no longer pass fit/truth selection, eight remain
other direction outliers and three remain reversed. All three residual
reversals have truth pT below 0.55 GeV. No simulation, digitization, GGTF
inference, truth-based fit corrections or threshold changes were used.

Numerical circle-seed regressions include measured hits from a reversed
candidate, both flight directions, and arcs exceeding a full turn. A native
charge audit found six serialized/fitted-charge mismatches in 23 baseline
outputs and zero in 23 patched outputs. The original three-event refit was
bit-identical before the patch, establishing the comparison baseline. Detailed
outputs are under `reports/idea-tracking-fix-20261117/`.

This fixes the dominant reversal mechanism. Overall track alignment still
fails the 95% quality gate, and the cluster and legacy proxy target-definition
limitations above remain. Do not relabel existing 0.4.0 files as 0.4.1.
