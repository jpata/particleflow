# IDEA tracks and clusters

IDEA uses measured GGTF/GenFit tracks and dual-readout calorimeter clusters.
Its oracle candidate reference (`ycand`) is a pipeline check, not a
reconstruction baseline, so the datasets do not establish physics performance.

## Physics samples

| Physics sample | Process name | TFDS dataset | Energy |
|---|---|---|---|
| Top-quark pairs | `p8_ee_ttbar_ecm365` | `idea_edm_ttbar_pf` | 365 GeV |
| Fully hadronic WW | `p8_ee_WW_fullhad_ecm365` | `idea_edm_ww_fullhad_pf` | 365 GeV |
| Inclusive quark pairs | `p8_ee_qq_ecm365` | `idea_edm_qq_pf` | 365 GeV |
| Z→qq | `p8_ee_Z_qq_ecm91` | `idea_edm_z_qq_pf` | 91 GeV |
| Z→tautau | `p8_ee_Z_tautau_ecm91` | `idea_edm_z_tautau_pf` | 91 GeV |

All datasets use version **0.5.0** and configuration **1**. The
`pyg-idea-pipeline-v1` recipe trains on ttbar, WW and qq. The
[catalog](catalog.md#published-availability) records public availability; the
published 0.2.1 data use proxy tracks and are not interchangeable with 0.5.0.

## Event view

These views are from the previous (proxy-track) campaign. They show simulated
`ttbar` event 0 at 365 GeV in the transverse plane.

| Truth-seeded tracks and clusters | Visible stable truth proxy |
|:---:|:---:|
| ![IDEA: truth-seeded tracks and clusters for event 0.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_tracks_clusters.png) | ![IDEA: visible stable truth proxy for the same event.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_particles.png) |

[All-component overlay](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_overlay.png),
[vector input view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_tracks_clusters.svg) and
[vector particle view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_particles.svg) are available for reuse.
See [event-view conventions](event-displays.md) for particle sources, sampling and scale.

## Software and generation

The generator lives in `mlpf/data/key4hep/gen/idea` (key4hep-sim v1.4.0) and
pins the `idea-upstream-2026-11` forks of FCC-config, k4geo, k4RecTracker and
k4RecCalorimeter with the local dual-readout truth patches and the tracking
fixes (circle-seed orientation from the ordered hit path, GENFIT's fitted
charge in the output). It runs on stable Key4hep 2026-04-08 with the shared
k4FWCore and MarlinReco overlays. Build everything once per host following the
[build instructions](../../mlpf/data/key4hep/gen/README.md); the build also
downloads the GGTF model `SimpleGatrIDEAv3o1.onnx` so that jobs never fetch it.

The runner writes fitted `FittedTracks` (used by MLPF), the proxy
`TracksFromGenParticles` for diagnostics, dual-readout `TopoClusterAll`
clusters with cell and truth links, and a `reco_PROCESS_SEED.software.txt`
manifest. Generation uses the shared FCC-ee cards, whose 2025 beam-spot widths
are the only vertex smearing. See the
[generator README](../../mlpf/data/key4hep/gen/idea/README.md) for options.

The [joint v1.4.0 campaign](key4hep-50k.md) generates **10,000 events per
sample** in 1,000 jobs of 10 events under `/local/joosep/mlpf/idea/v1.4.0/50k`.
IDEA jobs request 48,000 MiB and 4 CPUs for generation and reconstruction;
the 10-event jobs peak near 23 GiB. TFDS 0.5.0 uses 8,000 training and 2,000
test events per sample with a deterministic event split. Old `.done` markers
under `snakemake_jobs/idea/` do not establish completion of this campaign.

## Inputs and targets

Postprocess with:

```bash
PYTHONPATH=. python mlpf/data/key4hep/postprocessing.py \
  --detector idea --idea-track-source fitted \
  --input /path/to/reco.root --outpath /path/to/post/p8_ee_ttbar_ecm365
```

Fitted tracks are read from `FittedTracks` at their unique interaction-point
state. Quality selection requires finite parameters, nonnegative chi2,
positive NDF and associated hits; the innermost radius comes from measured
hits, and the dEdx fields are zero because the truth-parametrized dN/dx
estimator is excluded. Tracks keep their measured kinematics. Truth is used
only for labels: at least **75%** of a track's distinct measurements must
vote for one direct, charged status-1 particle, and secondary, fake and
low-purity tracks stay inputs without a label. Duplicate primary matches keep
one representative by purity, hit count and fit quality.

The Parquet files record the track source, collection, quality and truth
policies, and raw, rejected and matched track counts. The 0.5.0 builders and
the loader validate this provenance and reject proxy or unmarked inputs; there
is no implicit fallback to proxy tracks. Proxy-only diagnostics require
`--idea-track-source truth_seeded` and `expected_track_source="truth_seeded"`
in the loader.

## Calorimeter clustering

`CaloTopoClusterFCCee` builds calorimeter-only topological clusters from the
digitized Cherenkov (C) and scintillation (S) fibre signals with a constant
1 MeV noise estimate (seed significance 4, growth 2, one-cell halo). Cluster
energy is raw C+S, with the channel sums kept separately; the position is the
signal-weighted centroid projected onto the fibre front face, so there is no
longitudinal depth information. There is no shower splitting, track
association or charged-shower subtraction, and truth links do not influence
clustering. Events contain thousands of mostly sub-100 MeV clusters, and MLPF
receives one input per fragment. Two native defects remain in the upstream
code: a merged cluster can abandon part of its growth frontier (a small
energy loss), and the `dR_over_E` shape parameter uses a wrong azimuth
reference (MLPF recomputes widths from cells and ignores it).

## Limitations

- Track direction alignment does not yet meet the 95% validation gate;
  residual outliers are concentrated at low pT, and charged-source clusters
  align poorly with production directions. Treat `ycand` as an oracle and
  `suitable_for_physics` as false.
- Datasets produced with proxy tracks (0.2.1, 0.3.0) or before the tracking
  fix cannot be relabelled as 0.5.0; regenerate reconstruction, Parquet and
  TFDS in a fresh workspace.
