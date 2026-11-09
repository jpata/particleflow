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
`particle_number` without repeating its kinematics. The representative is the
track or calorimeter hit chosen to anchor that target. A membership-only track
can be a different track associated with the same target; its own direction
need not equal the target direction.

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

The production specification also maps CLD `ww_fullhad`, `zz`,
`zz_tautau_365`, `zz_tautau_240`, `zh_tautau_365`, `zh_tautau_240`, `qq`,
`z_qq`, and `z_tautau`; CLIC `ww_fullhad` and `qq`; IDEA `ww_fullhad` and `qq`;
and ColliderML `ttbar_pu200`. The CLD and IDEA single-particle guns remain
outside the production TFDS mappings. Each additional builder uses the same
track/calo schema and a distinct TFDS dataset name.

For CLD, CLIC, or IDEA production, regenerate Snakemake scripts after updating
the production workspace, then run the separate track/calo stage after
postprocessing:

```bash
PROD=cld pixi run snakefile
PROD=cld pixi run tfds_track_calo
```

The generator update in PR #527 changes the CLD, CLIC, and IDEA workspace
paths. Regenerate the Snakemake scripts after that change. The `.done` markers
still live under `snakemake_jobs/<production>/`; inspect or force the generation
and postprocessing tasks for the new campaign so old markers do not make the
new TFDS stage start before the new Parquet files exist.

The new builder's `manual_dir` is the corresponding `post/<process_name>/`
directory. The generated stage waits for that sample's postprocessing `.done`
markers. ColliderML uses its separate Flatiron launcher; the third argument
selects this view, for example
`sbatch --array=1-10 scripts/flatiron/colliderml_tfds.sh pu0 /path/to/parquet track_calo`.

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

To check the angular alignment of target-bearing tracks, place converted files
under `cld/`, `clic/`, `idea/`, and `colliderml/` in one input directory and run:

```bash
PYTHONPATH=. .venv/bin/python scripts/studies/plot_track_calo_track_resolution.py \
  --input-dir /path/to/converted --output-dir /path/to/track_resolution
```

The plots show signed track-minus-target residuals at ±0.01, ±0.1, and full
angular range, individual signed residuals versus target pT, absolute-residual
cumulative distributions, and 68%/95% absolute residuals versus target pT.
Phi differences are wrapped to [-π, π]. Full target rows on tracks are representatives;
`particle_number`-only rows are membership links and are plotted separately.
In particular, IDEA's descendant-track membership does not imply that the
descendant direction should equal the ancestor target direction. The JSON report
also checks whether large representative-track residuals occur on the same track
in the existing track/cluster target view.

To compare tracks with calorimeter-hit groups for the same target particles:

```bash
PYTHONPATH=. .venv/bin/python scripts/studies/plot_track_calo_centroid_alignment.py \
  --input-dir /path/to/converted --output-dir /path/to/centroid_alignment
```

This groups `X_hit_calo` rows by `ytarget_track_calo_hit.particle_number`
and uses their hit-energy-weighted XYZ centroid. The centroid direction from
the origin is compared with the target direction using signed eta and wrapped
phi residuals. Paired track and hit-centroid curves use exactly the same
track-represented targets with at least one labelled calorimeter hit. A third
curve shows hit centroids for calorimeter-represented targets. The JSON report
records group coverage and residual quantiles. The figures' third column uses
the track momentum magnitude `X_track[:,5]` as an energy proxy, or the sum of
`X_hit_calo[:,5]` for a hit group, against target energy. Residual plots show
`log10(input / target energy)`; the point plot shows the two energies directly.

These particle-number groups are labels based mainly on the dominant hit
contributor, with representative-hit labels taking precedence; they are not
exclusive per-particle energy deposits. Charged-particle bending can separate a
calorimeter centroid from the initial target direction. IDEA's hit weights are
digitized readout signals rather than calibrated particle energies, so its
hit-signal-to-GeV ratio is only a numerical comparison, not a calibrated
calorimeter energy response.

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

## Source-link and retention audit

The local source-link audit reruns the source readers for the converted events and
compares each stored calorimeter-hit label with the generator-to-hit association
weights passed to the target builder:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/particleflow_mpl .venv/bin/python \
  scripts/studies/validate_track_calo_truth.py --events 20
```

`--cld-root`, `--clic-root`, `--idea-root`, and `--colliderml-source-dir` can
override the local source paths. The report and plot are written to
`plots/track_calo_targets/truth_validation/`. Hit purity is the fraction of a
labelled hit's source association weight contributed by source particles mapped
to its target. Target completeness is the fraction of a target's source
association weight on hits carrying its label. Targets with no source calorimeter
link are excluded from the completeness denominator. These are checks against
the links used by assignment; they are not independent shower simulation truth
or calibrated energy-response measurements. In IDEA, weights are raw dual-readout
signals. The source particle set is already selected by each source reader, so
“invisible” here means newly invisible to the track/calo view.

On the local 20-event ttbar samples per detector, the audit found:

| Detector | Source particles | Represented | Newly invisible | Merged / dropped | Association-weighted hit purity and target completeness |
|---|---:|---:|---:|---:|---:|
| CLD | 2074 | 2074 | 0 | 0 / 0 | 99.79% |
| CLIC | 2027 | 2025 | 2 | 0 / 0 | 99.78% |
| IDEA | 2406 | 2318 | 88 | 0 / 0 | 99.74% |
| ColliderML | 4540 | 4540 | 0 | 0 / 0 | 97.15% |

IDEA's newly invisible particles account for 55.13 GeV, or 0.865% of its
selected source energy. Of the 88, 78 are photons, 81 have pT below 0.3 GeV,
and 56 have |η| above 3; these categories overlap. CLIC's two newly invisible
photons carry 0.17 GeV and both have pT below 0.3 GeV. All retained source
particles obtained their own representative in this sample, so no merge or drop
case was exercised. ColliderML has the largest shared-hit contamination: 7
calorimeter representative hits override the largest contributor, but those
hits explain only about 2.2% of its total foreign association weight. Most of
the difference is minority contributions on otherwise correctly labelled hits.
The ColliderML sample is one pileup-free shard; these numbers do not establish
performance under pileup or across production files.

## Current scope

These builders prepare datasets but do not register a track/calo training
configuration or input encoder. The tracker-hit geometry encoder is not suitable
for reconstructed-track rows, even though both use `elemtype=1`.
