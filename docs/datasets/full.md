# General datasets: tracks, clusters and hits

CLD, CLIC, MAIA, IDEA and ColliderML have two dataset kinds:

| Kind | TFDS name | Inputs | Use |
|---|---|---|---|
| Track/cluster (PF) | `<det>_edm_<process>_pf` (ColliderML: `colliderml_<process>_<pileup>_pf`) | Reconstructed tracks and calorimeter clusters | Standard MLPF ([track/cluster](track-cluster.md)) |
| General (full) | `<det>_edm_<process>_full` (ColliderML: `colliderml_ttbar_full`, `colliderml_ttbar_pu200_full`) | Tracks, clusters, tracker hits and calorimeter hits | Reconstruction from raw hits, or from any combination of element kinds |

The general dataset replaces the former hit (`*_hits`) and track/calorimeter-hit
(`*_track_calo_*`) views. Training selects which element kinds to use. The target
particles are the same for every selection.

## Targets

The target particles are the visible status-1 (or surrogate) generator particles,
with the same visibility rule as the track/cluster view. Unlike that view, no
reconstructed object is needed to represent a particle, and particles are not
merged. A visible particle is a target if at least one element of any kind is
labelled with it. `target_energy_accounting` records the source, invisible,
unlabelled and target energy (GeV) and closes per event.

`ytarget_particles` stores one row per target in the
[ParticleFeatures](../reference/configuration.md) layout. Column 0 is the class
index (`[0, 211, 130, 22, 11, 13]`), using the truth charge. When the training
selection contains no tracking element of a charged particle, the loader maps it
to its neutral counterpart. A tracking element is a track, or a tracker hit the
particle produced itself.

## Per-element labels

Each element kind has a label matrix `ylabel_<kind>` aligned with `X_<kind>`:

| Column | Meaning |
|---|---|
| `particle_number` | Owning target row (1-based, 0 = background). The target ancestor with the largest truth weight; descendant hits count for their ancestor. |
| `instance_number` | Row in `target_instances` of the simulated particle that produced the dominant part of the element |
| `origin` | How the element relates to its target (table below) |
| `purity` | Fraction of the element's truth weight contributed by its owning target |
| `true_px`, `true_py`, `true_pz` | Simulated momentum (GeV) of the producing particle at a tracker hit; zero for other elements |

| `origin` | Name | Meaning |
|---:|---|---|
| 0 | `none` | Background or no truth link |
| 1 | `primary` | Produced by the target particle itself (tracker hits: outgoing leg) |
| 2 | `primary_looping` | Tracker hit of the target itself on an inward-moving leg (p·r < 0) |
| 3 | `primary_unsaved_secondary` | Tracker hit attributed to the target but produced by a secondary not stored as an MC particle (SimTrackerHit quality bit 30, for example a delta ray) |
| 4 | `decay_product` | Produced by a descendant created in a decay of its parent: a long-lived species (π±, K, K0S/K0L, Λ, Σ, Ξ, Ω, μ, π0) whose products at that vertex are all allowed decay products (for example π→μν, K0S→ππ, Λ→pπ) |
| 5 | `secondary` | Produced by another descendant: interaction products, conversions, bremsstrahlung, delta rays |

The simulator's "decayed in tracker/calorimeter" status is set whenever a track ends
there, including inelastic interactions, so it is not used to identify decays.

Producers come from the direct truth links, before descendants are folded into
their status-1 ancestor (key4hep; IDEA and ColliderML are listed below):
- tracker hits: SimTrackerHit → MCParticle;
- calorimeter hits: the largest `CalohitMCTruthLink` contribution;
- tracks: the largest `SiTracksMCTruthLink` weight;
- clusters: their hits' truth energy.

`target_instances` has one row per producing particle:
`instance_number, particle_number, origin, PDG, charge, pt, eta, sin_phi, cos_phi, energy, vertex_x, vertex_y, vertex_z`.
The vertex is the production vertex in mm.

### Why the origin labels matter

`particle_number` alone is inclusive. It assigns the hits of decay products and
other secondaries to the status-1 ancestor. On 20 converted CLD ttbar events
(1,012 tracker hits per event on average), charged-particle circle fits from
truth-labelled tracker hits give these relative-pT 68% half-widths:

| Tracker hits used | Half-width | Within 5% |
|---|---:|---:|
| all hits with the particle's `particle_number` | 17.6% | 66% |
| hits with `origin == primary` | 0.9% | 92% |

Origin shares among labelled CLD tracker hits are:

| Origin | Share |
|---|---:|
| primary | 62% |
| primary looping | 3% |
| primary, unsaved secondary | 3.5% |
| decay product | 5% |
| other secondary | 20% |

Training selection:
- Supervise primary trajectories with the `primary` origin.
- Use `particle_number` for energy flow and particle-level targets.
- Use `instance_number` to supervise secondaries as separate instances.

## Training

The set recipes (`pyg-cld-hits-set-v1`, `pyg-cld-hits-heptv2-set-v1`,
`pyg-cld-hits-heptv2-sector-set-v1`, `pyg-clic-hits-set-v1`,
`pyg-colliderml-hits-set-v1`, and the new `pyg-idea-hits-heptv2-set-v1` with the
`idea_hits` model family) read the full datasets. The full datasets support only the set output mode, so elementwise
recipes keep reading previously built `*_hits` 3.2.2 datasets. Their comparison
scenario uses `pyg-cld-hits-set-legacy-v1`, the same set model on that data.

| Option | Default | Meaning |
|---|---|---|
| `full_elements` | `[hit_tracker, hit_calo]` | Element kinds, in element-type order (type = position + 1). Kinds: `track`, `cluster`, `hit_tracker`, `hit_calo`. |
| `model.set_decoder.hit_grouping_origins` | `null` (inclusive) | Origin codes that count as members of a particle in the hit-grouping loss, e.g. `[1]` |
| `model.set_decoder.hit_grouping_excluded_as` | `ignore` | Excluded members get zero weight (`ignore`) or are supervised as background (`background`) |

The default selection reproduces the legacy hit input layout: tracker hits are
type 1, calorimeter hits type 2, 15 hit features, input type `hits`. The current
hit recipes therefore train unchanged. Other selections pad each kind to the
widest kind's feature count (`input_dim` and `elemtypes_nonzero` follow the
selection), and the batch's `input_type_id` is `full`. In mixed selections the
hit feature engineering runs only on hit rows; tracks and clusters keep their raw
features. Batches expose the labels as `batch.ylabel`. For example, train the
HEPTv2 set recipe on tracks and calorimeter hits with `--full_elements [track,hit_calo]`.

```bash
python mlpf/pipeline.py --spec-file particleflow_spec.yaml \
  --model-name pyg-cld-hits-heptv2-set-v1 --production-name cld \
  --data-dir /path/to/tensorflow_datasets train --gpus 1 \
  --model.set_decoder.hit_grouping_origins [1]
```

## Produce

The key4hep (CLD, CLIC, MAIA, IDEA) and ColliderML postprocessors write the full view
into every Parquet file next to the track/cluster targets: `ylabel_{track,cluster,hit_tracker,hit_calo}`,
`target_particles`, `target_instances`, `target_energy_accounting` and
`targetjet_full`. It no longer writes the legacy `ytarget_hit_*` or
`*_track_calo_*` fields. Rerunning it on a file without the full view regenerates
that file.

```bash
PROD=cld pixi run snakefile
PROD=cld pixi run tfds_full
```

Each `tfds_full` job reads one sample's `post/<process_name>/` directory. It
uses the same deterministic file split as the PF datasets: a sorted 90/10
train/test split, then ten configurations per split. Event-disjoint `train/` and
`test/` subdirectories override the split. For a manual build:

```bash
PYTHONPATH=. tfds build mlpf/heptfds/cld_full/ttbar.py --config 10 \
  --manual_dir /path/to/manual --data_dir /path/to/tensorflow_datasets
```

The builder validates:
- input/label alignment;
- particle and instance references;
- that every element's instance belongs to its owning particle;
- origin codes;
- that each target labels at least one element;
- energy accounting.

## Scope

- **Covered:** CLD (all mapped samples, including the granularity scans), CLIC (ttbar, WW, qq), MAIA (ttbar builder), IDEA (ttbar, WW, qq) and ColliderML (ttbar without pileup and with pileup 200).
- **Validated on converted events:** CLD, CLIC, IDEA and ColliderML (no pileup). The strict Parquet validator runs the TFDS contract on every event (`full_view_labels` gate).

## Detector specifics

**IDEA.**
- Tracker hits come from the vertex detector (`VTXB/VTXD`), the silicon wrapper (`SiWrB/SiWrD`) and the drift chamber (`DCH_DigiCollection`), each linked digi → SimTrackerHit → MCParticle.
- `system` codes are 1/2 for the vertex barrel/endcap, 3 for the drift chamber and 5/6 for the silicon wrapper barrel/endcap.
- `side` and `layer` are zero, because the files carry no CellIDEncoding metadata.
- Drift-chamber hits (`type` = 1) store the position on the sense wire without the drift distance, so they are not suitable for circle fits. An event has 1–4 × 10⁴ of them.
- Calorimeter hits are the separate Cherenkov (`type` 1) and scintillation (`type` 2) channels. Their signals are not calibrated particle energy.
- Tracks are truth-seeded (`TracksFromGenParticles`). Clusters use `ClusterMCParticleLinks`.
- On converted ttbar events, fits to primary silicon hits give a 0.6% relative-pT half-width (17% with inclusive `particle_number`).

**ColliderML.**
- Producers come from `tracker_hits.particle_id` and `calo_hits.contrib_particle_ids`. Target ownership uses the converter's leaf walk, so the owning target matches the existing attribution.
- ColliderML stores Geant4 shower particles. Deposits of particles produced beyond the tracker envelope (r > 1150 mm or |z| > 3100 mm) are credited to the ancestor that entered the calorimeter, as in key4hep.
- Tracker hits have no simulated momentum. `true_px/py/pz` stay zero, and inward legs are found from the time-ordered hit radius of each producer.
- There is no unsaved-secondary flag.
- Secondary electrons (delta rays, conversions) are stored as separate particles, so about half of the labelled tracker hits are `secondary`.
- Tracks and clusters use the 17-column union layout, and the pileup reference fields `genmet_withpu`/`genjets_withpu` are copied through.
- On 20 converted events, fits to primary tracker hits give a 5% relative-pT half-width (68% within 5%), at 3 T.
