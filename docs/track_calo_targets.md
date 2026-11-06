# Track and calorimeter-hit target datasets

The Key4hep and ColliderML postprocessors produce particle targets on reconstructed
tracks and individual calorimeter hits for CLD, CLIC, MAIA, ColliderML, and IDEA.
These targets form a separate view alongside the existing track/cluster targets.
The converters regenerate Parquet files that lack the complete track/calo view.

## Parquet output

The inputs are `X_track` and `X_hit_calo`. Each target row aligns with one input
row. Key4hep tracks use a 16-column source schema; ColliderML tracks use 17
columns. The TFDS builders map both to a common schema.

| Field | Shape | Meaning |
|---|---|---|
| `ytarget_track_calo_track` | `[T,14]` | Targets aligned with `X_track` |
| `ytarget_track_calo_hit` | `[H,14]` | Targets aligned with `X_hit_calo` |
| `track_calo_target_particles` | `[P,14]` | One row per final target particle |
| `track_calo_original_to_target` | `[Pinput]` | Final target row index, or `-1` if removed |
| `track_calo_energy_accounting` | `[4]` | Input, newly invisible, dropped, and retained target energy in GeV |
| `targetjet_track_calo` | `[J,4]` | Target jet `pt`, `eta`, `phi`, and energy |

`particle_number` is one-based, with zero for background. Its IDs are local to
this view; do not join them numerically to the existing track/cluster targets.
Each target has one full representative row. Other associated inputs carry its
`particle_number` without repeating its kinematics.

The assignment uses visible particles and their track and calorimeter-hit
associations. It prefers a representative track when one is available. A visible
particle without its own representative can merge into a retained particle
through a shared calorimeter hit; otherwise it is recorded as dropped energy.
The energy accounting starts from the source reader's selected particle set.
Tracker-hit energy and reconstructed clusters do not determine this assignment.

## Build TFDS datasets

The five builders produce the same `X`, `ytarget`, `ycand`, `genmet`, `genjets`,
and `targetjets` fields:

| Detector | Dataset name |
|---|---|
| CLD | `cld_track_calo_ttbar` |
| CLIC | `clic_track_calo_ttbar` |
| MAIA | `maia_track_calo_ttbar` |
| ColliderML | `colliderml_track_calo_ttbar` |
| IDEA | `idea_track_calo_ttbar` |

Place event-disjoint Parquet files in `train/` and `test/` under the TFDS
`manual_dir`, then build the matching detector module. For example:

```bash
PYTHONPATH=. .venv/bin/tfds build mlpf/heptfds/cld_track_calo/ttbar.py \
  --config 10 --manual_dir /path/to/manual \
  --data_dir /path/to/tensorflow_datasets
```

Alternatively, place at least two Parquet files directly in `manual_dir` for a
deterministic file-based 90/10 split. Different files must contain different
physical events. Configs 1–10 partition each split; config 10 takes the final
partition and works for small samples with fewer than ten files per split.

The common `X` matrix has 25 columns: 15 hit features followed by track-specific
features. The builders record feature availability and source information in
TFDS metadata. `ytarget` concatenates the track and calorimeter-hit target rows
and maps PDG labels to class indices for `[0, 211, 130, 22, 11, 13]`.
`targetjets` comes from `targetjet_track_calo`. `ycand` is zero because these
datasets do not provide reconstructed PF candidates.

The builders validate input widths, target alignment, representative uniqueness,
particle and jet references, finite values, and energy accounting. To inspect a
converted Parquet file directly:

```bash
PYTHONPATH=. .venv/bin/python scripts/studies/validate_track_calo_targets.py \
  /path/to/parquet/*.parquet --output /path/to/validation.json
```

## IDEA readout and tracks

IDEA's `X_hit_calo` contains positive-energy digitized Cherenkov and
scintillation signals. The two channels remain separate hits, even at the same
position. Positive unlinked signals remain as background inputs; zero-signal
cells are omitted. Hit `type` distinguishes Cherenkov (`1`) from scintillation
(`2`). The channel sum is a detector response, not calibrated particle energy.

IDEA currently uses truth-seeded `TracksFromGenParticles` as track inputs. Only
direct status-1 track associations can represent a target particle; descendant
tracks can carry membership without representing the ancestor's kinematics.
IDEA's existing `ytarget_hit_calo` remains empty because the new hit labels are
stored in `ytarget_track_calo_hit`.

## Current scope

These builders prepare datasets but do not register a track/calo training
configuration or input encoder. The tracker-hit geometry encoder is not suitable
for reconstructed-track rows, even though both use `elemtype=1`.
