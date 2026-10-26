# Reconstructed tracks plus calorimeter-hit targets

The opt-in `--track-calo-targets` flag on the Key4hep and ColliderML postprocessors
exports an additional target view. CLD, CLIC, MAIA and IDEA use Key4hep adapters;
ColliderML uses its native release tables. Existing track/cluster and all-hit
outputs keep their existing definitions. The new view is built before the legacy
cluster allocator changes the particle population.

## Assignment policy

`mlpf/data/track_calo_targets.py` implements the shared assignment:

1. Start from the source reader's PF ancestry and visibility-selected particles.
   Recheck visibility using calibrated **calorimeter** deposits and reconstructed
   track support, excluding raw tracker-hit energy. The existing 10% fractional,
   0.5 GeV charged-MIP and 20% track-hit-fraction criteria apply.
2. Construct sparse particle-to-track and particle-to-calorimeter-hit associations.
   Sum duplicate associations. Eligible representative tracks require at least
   20% hit fraction; calorimeter associations require positive deposited energy.
3. Visit particles in descending energy order, breaking ties by original index.
   Prefer available tracks over calorimeter hits, with associations ordered by
   weight and then element index. Iterative augmenting paths resolve conflicts
   through multiple reassignments. Matching has maximum cardinality, but does not
   optimize a global physics score or guarantee maximum track occupancy.
4. A visible particle without a representative can merge only into a retained
   particle owning a calorimeter hit to which it contributed. Select the host
   with the largest summed shared-hit contribution. Sum four-momenta. If no host
   exists, drop the particle and account for its energy explicitly. Reconstructed
   clusters do not determine retention or merge hosts.
5. Remap original associations to final targets and sum merged contributions
   **before** selecting inclusive owners. Each target has one full representative
   row; all other associated elements carry only `particle_number`. Representative
   ownership takes precedence over dominant-contributor ownership, preserving the
   existing row-target convention.
6. Apply the existing charged/neutral fallback class mappings using track/calo
   representation; calo-only targets have zero charge. ColliderML retains its
   pi0-as-photon convention. Build target jets from the unique final target table
   and copy jet indices to representatives.

Unmatched and duplicate reconstructed tracks remain inputs. An unclaimed track
can have a membership label without becoming another particle target. The helper
neither changes the input `EventData` nor modifies legacy outputs.

## Additional Parquet fields

| Field | Shape | Meaning |
|---|---|---|
| `ytarget_track_calo_track` | `[T,14]` | Targets aligned with `X_track` |
| `ytarget_track_calo_hit` | `[Hcalo,14]` | Targets aligned with `X_hit_calo` (IDEA: `X_track_calo_hit`) |
| `track_calo_target_particles` | `[P,14]` | Unique final particle table, standard `ParticleFeatures` order |
| `track_calo_original_to_target` | `[Pinput]` | Zero-based final row index or -1; input index refers to the reader's filtered PF truth table, **not** a raw MCParticle ID |
| `track_calo_energy_accounting` | `[4]` | Input, newly invisible, dropped, final target energy in GeV |
| `targetjet_track_calo` | `[J,4]` | Jets of final targets: pt, eta, phi, energy |

`particle_number` is one-based with zero for background. IDs belong to this view
and must not be joined to legacy targets by numerical ID. Jets and representative
rows must likewise come from the same view. Energy accounting begins after the
source reader's ancestry/visibility selection, not from all simulated particles.
Auxiliary source diagnostics such as `gp_to_cluster` retain the source values;
they do not control the new assignment.

Inputs are `X_track` and `X_hit_calo` (IDEA uses `X_track_calo_hit` to preserve its
legacy empty raw-hit view). Key4hep tracks have the 16-column EDM4hep
layout; ColliderML tracks retain their existing 17-column union layout. Do not
assume they are identical schemas. The new TFDS builders map both into the common
layout below; existing datasets keep their original layouts.

## Run and validate

Add `--track-calo-targets` to either converter command. Use a distinct output
directory for comparison. When requested, the flag forces regeneration rather
than accepting an existing legacy-only output as complete.

The local smoke script uses the CLD/CLIC CI ROOT samples, the MAIA sample used by
`scripts/local_test_maia.sh`, the local IDEA ttbar ROOT sample, and the locally downloaded real ColliderML tables:

```bash
NUM_EVENTS=3 bash scripts/local_test_track_calo_targets.sh
```

Override `CLD_ROOT_FILE`, `CLIC_ROOT_FILE`, `MAIA_ROOT_FILE`, `IDEA_ROOT_FILE`, `COLLIDERML_SOURCE`,
`TRACK_CALO_OUTPUT`, or `TRACK_CALO_PYTHON` as needed. ColliderML source files are
Parquet release tables, not ROOT. The script explicitly selects Uproot's local
file reader to avoid the known FSSpec/PODIO metadata stall.

Validation checks array alignment, finite values, one unique representative per
target, equality with the explicit particle table, valid target/jet references,
and energy accounting. It can be rerun without converting:

```bash
PYTHONPATH=. .venv/bin/python scripts/studies/validate_track_calo_targets.py \
  local_test_data/track_calo_targets/{cld,clic,maia,colliderml}/*.parquet \
  --output local_test_data/track_calo_targets/validation.json
```

## Local validation, 2026-10-23

Three events per source, using real CLD, CLIC, MAIA and ColliderML data:

| Source | Legacy targets | New targets | Track representatives | Calo representatives |
|---|---:|---:|---:|---:|
| CLD | 368 | 409 | 190 | 219 |
| CLIC | 280 | 306 | 137 | 169 |
| MAIA | 481 | 707 | 32 | 675 |
| ColliderML | 385 | 386 | 89 | 297 |

All output contract checks passed. Newly invisible and dropped energy were zero
in these samples. More retained targets reflect removing cluster-capacity limits;
this is not evidence of improved reconstruction accuracy. The MAIA fixture lacks
CellIDEncoding metadata, so some detector identifiers remain unknown.

Focused tests cover duplicate tracks, tracker-only visibility, empty events,
track-only particles, shared-cluster separation, multi-step matching conflicts,
merged-label propagation, explicit drops, jet indices, unchanged legacy outputs,
and both Parquet layouts. Matching cardinality is also checked against SciPy on
random sparse graphs. Full production-scale performance and downstream training
are not established by this smoke test.


## TFDS builders (2026-10-26)

Five version-1.0.0 ArrayRecord builders share `mlpf/heptfds/track_calo_utils`:

| Builder module | TFDS dataset name |
|---|---|
| `mlpf/heptfds/cld_track_calo/ttbar.py` | `cld_track_calo_ttbar` |
| `mlpf/heptfds/clic_track_calo/ttbar.py` | `clic_track_calo_ttbar` |
| `mlpf/heptfds/idea_track_calo/ttbar.py` | `idea_track_calo_ttbar` |
| `mlpf/heptfds/maia_track_calo/ttbar.py` | `maia_track_calo_ttbar` |
| `mlpf/heptfds/colliderml_track_calo/ttbar.py` | `colliderml_track_calo_ttbar` |

The common `X[N,25]` schema keeps the 15 hit columns first, renaming column 1 to
`pt_or_et` and column 5 to `p_or_energy`. Columns 15–24 are `chi2`, `ndf`, `dEdx`,
`dEdxError`, `radiusOfInnermostHit`, `tanLambda`, `D0`, `omega`, `Z0`, and `n_meas`.
Track rows copy their first six shared features, populate time where available,
and map named track fields to those additional columns. Track hit-position and
surface fields are zero; they must not be interpreted as physical hit geometry.
Calo rows preserve their 15 columns and have zero track-specific columns.

ColliderML's `n_meas` is not assigned to `ndf`. Redundant source `theta` and `qop`
are not stored separately: the layout retains direction, momentum and signed
curvature. Builder metadata records feature order, element types, class labels,
units, target/schema versions and source feature availability. Availability flags
are schema-level information, not per-measurement validity or quality masks.
ColliderML's placeholder calo time/type/surface fields and the current MAIA
production's missing surface metadata are marked unavailable.

The TFDS output interface remains `X`, `ytarget`, `ycand`, `genmet`, `genjets`,
`targetjets`. `ytarget` is the concatenated new track/calo representative arrays,
with PDGs mapped to `[0, 211, 130, 22, 11, 13]` class indices and membership IDs
unchanged. `targetjets` comes exclusively from `targetjet_track_calo`; `ycand` is
zero and metadata declares that no reconstructed PF-candidate baseline exists.
The explicit Parquet particle table is checked against the representatives and
is not stored a second time in TFDS. Input-type ID 3 identifies this new view;
existing input-type IDs are unchanged.

Conversion fails on missing new fields, wrong nonempty feature widths, nonfinite
values, row misalignment, duplicate/missing representatives, invalid IDs/classes,
and inconsistent energy accounting. Only required Parquet columns are loaded.
Empty and background-only events are retained, including original per-file event
indices in example keys.

For production builds, prepare event-disjoint `train/` and `test/` directories
under `manual_dir`. Alternatively, supply at least two Parquet files directly:
sorted files are split 90/10, matching the existing file-based split policy.
Configs 1–10 partition each split as in the legacy builders; use config 10 for
small samples. Same-file train/test symlinks are rejected, but the caller remains
responsible for ensuring different files do not duplicate physical events.

```bash
PYTHONPATH=. .venv/bin/tfds build mlpf/heptfds/cld_track_calo/ttbar.py \
  --config 10 --manual_dir /path/to/manual \
  --data_dir /path/to/tensorflow_datasets
```

For one local Parquet sample, the smoke helper writes disjoint event slices and
builds/reads back every TFDS example:

```bash
PYTHONPATH=. .venv/bin/python scripts/build_track_calo_tfds.py \
  --detector cld \
  --input local_test_data/track_calo_targets/cld/reco_p8_ee_ttbar_ecm365_300000.parquet \
  --output-dir local_test_data/track_calo_tfds
```

Use a fresh output directory when changing source samples: TFDS caches completed
dataset versions. The helper compares all seven output fields after serialization,
accounting for TFDS shuffling.

Local real-data validation produced two training events and one test event for
each of the four detectors under
`local_test_data/track_calo_tfds/tensorflow_datasets/<dataset>/10/1.0.0`.
Unit tests also exercise feature mappings, malformed inputs, both Parquet layouts,
ArrayRecord metadata persistence, and `TFDSDataSource` sorting/padding/set-target
extraction. The combined focused suite passes 63 tests.

These builders do not yet enable model training with the new view: dataset/model
configuration registration and a track/calo input encoder are separate work.
The existing tracker-hit geometry encoder must not be used on reconstructed-track
rows simply because their `elemtype` is also 1.


## IDEA extension

IDEA postprocessing now accepts `--track-calo-targets`, and its production spec
turns the option on by default. Existing track/cluster outputs and the empty
legacy raw-hit columns remain unchanged. A separate `X_track_calo_hit[N,15]`
column holds positive-energy digitized signals from
`DRcaloSiPMreadoutDigiHit` (Cherenkov) and
`DRcaloSiPMreadoutDigiHit_scint` (scintillation). The IDEA TFDS builder explicitly
requires that column. Zero-signal cells are omitted without using truth; positive
unlinked signals remain as background inputs. Raw tracker hits are not included.

Hit `type` is 1 for Cherenkov and 2 for scintillation; `subdetector` is 2 for the
combined dual-readout calorimeter. System/side/layer are unknown and zero-filled.
The channels remain separate measured signals, even when their positions or cell
IDs coincide. Their energy sum is a response quantity, not a claim of calibrated
particle-energy reconstruction. The shared visibility rule uses the combined
attributed channel response; physical calibration/performance needs validation
beyond these schema tests.

`CaloHitMCParticleLinks` identifies hits by **collection ID and local index**.
The adapter validates both endpoints and multiplies stored link fractions by the
measured channel energy before ancestry propagation. It does not copy cluster
labels to hits or renormalize inherited truth fractions. The target truth set is
non-neutrino status-1 particles, then the common visibility/assignment rules apply.

The input tracks are `TracksFromGenParticles`: **truth-seeded proxies**, not a
measured track-finding baseline. TFDS metadata records `track_source=truth_seeded`,
channel labels, and missing dE/dx fields. Only direct status-1 track associations
are eligible representatives. Descendant tracks can retain PF membership in an
ancestor, but cannot supply that ancestor's full kinematic target. An optional
`representative_track_edges` argument in the shared allocator implements this
separation; the other four detectors retain their previous default behavior.

Three real IDEA ttbar events were converted and validated locally:

- 339 new targets (165 track representatives, 174 calo representatives), versus
  340 legacy targets.
- 21,147 / 14,666 / 20,258 positive-energy input signals per event.
- 0.550276 GeV removed by visibility; zero energy dropped by assignment.
- Exact equality for all legacy fields with an unflagged conversion.
- ArrayRecord build/readback passed with two train events and one test event at
  `local_test_data/idea_track_calo_tfds/tensorflow_datasets/idea_track_calo_ttbar/10/1.0.0`.

The numerical comparison is a small-sample consistency check, not tracking or PF
performance evidence. IDEA does not gain an oracle PF candidate: `ycand` remains
zero in the new builder, as for the other track/calo datasets.
