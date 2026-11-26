# IDEA tracks and clusters

IDEA datasets use fitted GGTF/GenFit tracks and dual-readout calorimeter
clusters. IDEA has no reconstructed particle-flow baseline: the candidate
reference (`ycand`) is an oracle built from truth, useful for checking the
pipeline but not for judging physics performance.

## Physics samples

| Physics sample | Process name | TFDS dataset | Energy |
|---|---|---|---|
| Top-quark pairs | `p8_ee_ttbar_ecm365` | `idea_edm_ttbar_pf` | 365 GeV |
| Fully hadronic WW | `p8_ee_WW_fullhad_ecm365` | `idea_edm_ww_fullhad_pf` | 365 GeV |
| Inclusive quark pairs | `p8_ee_qq_ecm365` | `idea_edm_qq_pf` | 365 GeV |
| Z→qq | `p8_ee_Z_qq_ecm91` | `idea_edm_z_qq_pf` | 91 GeV |
| Z→tautau | `p8_ee_Z_tautau_ecm91` | `idea_edm_z_tautau_pf` | 91 GeV |

All datasets use version **0.5.0** and configuration **1**. The
`pyg-idea-pipeline-v1` recipe trains on ttbar, WW and qq;
`pyg-idea-key4hep-v140` uses all five samples. Version 0.5.0 is
[published](catalog.md#published-availability) on the Hub. The Hub also keeps
the earlier 0.3.0 datasets, which use truth-seeded proxy tracks and cannot
replace 0.5.0.

## Event view

These views show simulated `ttbar` event 0 at 365 GeV in the transverse
plane. They were made from the earlier 0.3.0 data, with truth-seeded tracks.

| Truth-seeded tracks and clusters | Visible stable truth proxy |
|:---:|:---:|
| ![IDEA: truth-seeded tracks and clusters for event 0.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_tracks_clusters.png) | ![IDEA: visible stable truth proxy for the same event.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_particles.png) |

[All-component overlay](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_overlay.png),
[vector input view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_tracks_clusters.svg) and
[vector particle view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/idea/event_0_particles.svg) are available for reuse.
See [event-view conventions](event-displays.md) for particle sources, sampling and scale.

## Software and generation

The generator is in `mlpf/data/key4hep/gen/idea`. It pins forks of
FCC-config, k4geo, k4RecTracker and k4RecCalorimeter (branch
`idea-upstream-2026-11`) that add dual-readout truth information and two
tracking fixes: the circle-seed orientation follows the ordered hit path, and
the output stores the charge fitted by GENFIT. It runs on the stable Key4hep
2026-04-08 release with the k4FWCore and MarlinReco builds shared by all
detectors. Build everything once per host following the
[build instructions](../../mlpf/data/key4hep/gen/README.md); the build also
downloads the GGTF model `SimpleGatrIDEAv3o1.onnx`, so jobs do not fetch it.

Each job writes the fitted tracks (`FittedTracks`, used by MLPF), the
truth-seeded proxy tracks (`TracksFromGenParticles`, for diagnostics only),
dual-readout clusters (`TopoClusterAll`) with cell and truth links, and a
`reco_PROCESS_SEED.software.txt` manifest of the software used. Events are
generated with the shared FCC-ee cards; their 2025 beam-spot widths are the
only vertex smearing. See the
[generator README](../../mlpf/data/key4hep/gen/idea/README.md) for options.

The datasets come from the [Key4hep v1.4.0 production](key4hep-50k.md):
10,000 events per sample, split into 8,000 training and 2,000 test events.
Reconstruction needs about 23 GiB of memory per 10-event job.

## Inputs and targets

Postprocess with:

```bash
PYTHONPATH=. python mlpf/data/key4hep/postprocessing.py \
  --detector idea --idea-track-source fitted \
  --input /path/to/reco.root --outpath /path/to/post/p8_ee_ttbar_ecm365
```

Track parameters are taken from the interaction-point state of each
`FittedTracks` entry. A track is kept if its parameters are finite, its chi2 is
nonnegative, its NDF is positive and it has associated hits. The innermost
radius comes from the measured hits. The dE/dx features are zero, because the
available dN/dx estimator is parametrized from truth.

Input tracks keep their measured kinematics; truth only assigns labels. A track
is labelled with a particle when at least **75%** of its distinct measurements
belong to one charged, status-1 particle from the primary interaction.
Secondary, fake and low-purity tracks remain inputs without a label. If
several tracks match the same particle, one is chosen by purity, then hit
count, then fit quality.

The Parquet files record the track source, collection, selection and
labelling policies, and the numbers of raw, rejected and matched tracks. The
0.5.0 builders and the loader check this record and reject proxy-track or
unrecorded inputs. To study proxy tracks instead, postprocess with
`--idea-track-source truth_seeded` and load with
`expected_track_source="truth_seeded"`.

## Calorimeter clustering

`CaloTopoClusterFCCee` builds topological clusters from the digitized
Cherenkov (C) and scintillation (S) fibre signals, using a constant 1 MeV noise
estimate, a seed significance of 4, a growth significance of 2 and a one-cell
halo. The cluster energy is the raw C+S sum, and the two channel sums are also
stored. The position is the signal-weighted centroid projected onto the front
face of the fibres, so clusters carry no information on shower depth.

Clustering uses calorimeter information only: there is no shower splitting,
track association or charged-shower subtraction, and truth links do not affect
it. A typical event therefore has thousands of clusters, most below 100 MeV,
and each becomes one MLPF input.

Two issues in the upstream clustering code remain. When clusters merge, part of
the growth frontier can be dropped, losing a small amount of energy. The
`dR_over_E` shape parameter uses the wrong azimuthal reference; MLPF ignores it
and computes shower widths from the cells instead.

## Version history

| Version | Change |
|---|---|
| 0.5.0 | Fitted GGTF/GenFit tracks with truth used only for labels; Z→qq and Z→tautau samples; Key4hep v1.4.0 production |
| 0.3.0 | Track phi, D0, and Z0 stored at the perigee relative to the origin, as for CLD and CLIC (previously at the production vertex for secondaries); `radiusOfInnermostHit` from the AtFirstHit track state |
| 0.2.1 | Cluster shower widths from the associated calorimeter cells |
| 0.2.0 | Muon-system cluster features; input width 17 → 19 |

Versions are not interchangeable: rebuild rather than relabel an older dataset directory. Input-normalization statistics exist for 0.1.0, 0.2.1, 0.3.0, and 0.5.0.

## Limitations

- Track directions do not yet pass the 95% alignment check in the dataset
  validation. The outliers are mostly low-pT tracks. Clusters from charged
  particles also align poorly with the particles' production directions.
- `ycand` is an oracle, and every event has `metadata_suitable_for_physics`
  set to false.
- Data from older versions cannot be relabelled as 0.5.0. Rerun
  reconstruction, postprocessing and TFDS in a new workspace.
