# ILD at FCC-ee

The `ild` production uses **ILD_FCCee_v02**, a 2 T field, full Geant4 simulation,
Clupatra pattern recognition and DDKalTest refits (`MarlinTrkTracks`), followed
by Pandora calorimeter clustering and particle flow. TFDS version **0.1.0**
is an experimental qualification dataset. Passing conversion or building TFDS
does not establish reconstruction performance.

## Software and generation

The generator is under `mlpf/data/key4hep/gen/ild`. Its ILDConfig submodule pins
`fc1c9f58f9912b4d1fb10da5defbaca2133d60a0`. It reuses the pinned k4geo and
k4RecTracker installations and development container from the IDEA recipe.
The qualified environment is the LCG `devkey-head/Mon`, GCC 16 view. The
Tuesday view tested on 17 November 2026 did not expose `TrackMerger`, which
the ILD steering imports even with optional track merging disabled. Builds
must match the view checksum; see the [IDEA build instructions](../../mlpf/data/key4hep/gen/idea/README.md).

Initialize submodules recursively. In a clean AlmaLinux shell, build the shared
geometry and tracking installations in the same view if they are absent:

```bash
git submodule update --init --recursive
apptainer exec --bind /cvmfs,"$PWD" \
  --env IDEA_STACK_RELEASE=Mon mlpf/data/key4hep/gen/idea/idea-dev.sif \
  bash mlpf/data/key4hep/gen/idea/build_k4geo.sh
apptainer exec --bind /cvmfs,"$PWD" \
  --env IDEA_STACK_RELEASE=Mon mlpf/data/key4hep/gen/idea/idea-dev.sif \
  bash mlpf/data/key4hep/gen/idea/build_k4rectracker.sh
```

For example, generate ten 10 GeV muons:

```bash
apptainer exec --bind /cvmfs,"$PWD",/tmp \
  --env LD_PRELOAD=/usr/lib64/libffi.so.8 \
  mlpf/data/key4hep/gen/idea/idea-dev.sif \
  env NEV=10 OUTDIR="$PWD/reports/ild/gen" WORKDIR="$PWD/reports/ild/mu-work" \
  bash mlpf/data/key4hep/gen/ild/run_sim.sh gun_mu_10gev 600000
```

Use a fresh `WORKDIR` for every job. The runner retains simulation, generation
cards, logs, an immutable recipe snapshot and reconstruction output. It copies
the final ROOT file to `OUTDIR/PROCESS/root/reco_PROCESS_SEED.root` only after
successful reconstruction, together with a `.software.txt` manifest recording
source commits, source diffs, stack checksum, compact XML/field-map/CAD checksums
and command arguments. There is no automatic scratch deletion.

The geometry staging step copies pinned compact XML and field maps and supplies
the six installed beam-pipe STL files. It changes CAD references to relative
paths because the tested GeoSvc reader incorrectly prepends the XML directory
to expanded absolute filenames. It changes no material, dimension or field.
The staged geometry is shared by simulation and reconstruction. A job uses a
private steering directory so concurrent reconstructions cannot overwrite
upstream diagnostics. Simulation retains the upstream TPC sensitive action.

Supported guns are `gun_e_*`, `gun_mu_*`, `gun_pi_*`, `gun_gamma_*` and
`gun_kaon0L_*`; `GUN_ENERGY` and `GUN_PARTICLE` can override their defaults. Optional
`GUN_THETA_MIN` and `GUN_THETA_MAX` accept DD4hep units such as `45*degree`
and `135*degree` for an acceptance-restricted control sample.
`--gun.energy` is the DD4hep gun momentum magnitude, so a massive particle's
total energy is slightly above its nominal gun setting. Physics cards are
`p8_ee_qq_ecm365`, `p8_ee_ttbar_ecm365` and `p8_ee_WW_fullhad_ecm365`.
The steering selects the FCC-ee model explicitly, without perfect tracking or
perfect PFA. The default final track collection refits Clupatra's silicon-hit
pickup; optional separate silicon/TPC track merging is not enabled.

The production, TFDS mappings and `pyg-ild-v1` recipe are in
`particleflow_spec.yaml`. Guns remain available through the runner for manual
calibration checks; the batch campaign below contains only collision samples.

## Tallinn 10k-event campaign

The `ild` production requests **10,000 events per physics sample** at 365 GeV:

| Sample | Seeds (end exclusive) | Jobs | Events per job |
|---|---|---:|---:|
| ttbar | 300000–300100 | 100 | 100 |
| WW full-hadronic | 400000–400100 | 100 | 100 |
| qq | 500000–500100 | 100 | 100 |

Its workspace is
`/local/joosep/mlpf/ild/ILD_FCCee_v02_10k_20261117`, with `gen/`, `post/` and
`tfds/` subdirectories. With 100 complete, independent input shards, each
dataset has **9,000 train / 1,000 test events**, configuration `1`, version
`0.1.0`, in ArrayRecord format. Train and test use disjoint ROOT jobs.

On Tallinn, update the checkout and select and initialize the SLURM profile:

```bash
git pull --ff-only
git submodule update --init --recursive
ln -sfn configs/tallinn/pixi.toml pixi.toml
pixi run init
bash scripts/check_ild_runtime.sh
```

The runtime check verifies the development image, Monday-view checksum against
both local installations, pinned ILDConfig, DDG4 and TrackMerger imports, and
the staged geometry/CAD assets. The image and local installations are ignored
by Git: a fresh checkout needs the image built as described in the IDEA README
and geometry/tracker builds using the commands above. Reuse existing matching
builds; rebuilding shared IDEA installations changes the runtime used by other
jobs. Each generated ILD job explicitly exports the qualified Monday view and
`LD_PRELOAD=/usr/lib64/libffi.so.8`.

Preview the complete production and then submit it:

```bash
PROD=ild pixi run datasets --executor slurm --jobs 20 --dry-run --printshellcmds
PROD=ild pixi run datasets --executor slurm --jobs 20
```

`datasets` targets `tfds` in the combined generated `Snakefile`. It schedules
simulation/reconstruction, postprocessing and dataset building as dependencies;
it does not request training. `--jobs 20` bounds simultaneous batch jobs and can
be adjusted to the site allocation. The resource requests are 16 GB per
generation/reconstruction job, 8 GB per postprocessing or TFDS job, and a
48-hour runtime limit. `datasets_ild` is the ILD convenience alias. For
generation only, use `PROD=ild pixi run gen`.

ILD explicitly forbids partial production, including when the shared Pixi
workflow generator receives `--allow-partial`. An existing subset of successful
jobs cannot shrink the required input set. Failed jobs block their dataset
instead of silently producing fewer than the requested events. Rerun the same
command to resume completed work.

Scaling the measured 600-event campaign suggests approximately **300 GB of
retained work directories** on scratch, **150 GB of published ROOT files**, and
**22 GB of Parquet files**, plus TFDS and temporary builds. Reserve headroom;
the native runner retains simulation and reconstruction intermediates.

The complete graph was dry-run with 300 generation jobs, 300 postprocessing
jobs and three TFDS builds. A local Snakemake execution reused the six qualified
ROOT/Parquet shards and built all three datasets; all 600 decoded records had
finite, aligned features and targets. That two-shard check has 100 train / 100
test events per sample; the full campaign uses the 90/10 file split above.
Validation artifacts are in `reports/ild-snakemake-10k-20261117/`. Submission on
Tallinn remains an operator action; the physics limitations below still apply.

## Inputs, target policy and baseline

Postprocess ROOT with:

```bash
mkdir -p reports/ild/post/p8_ee_qq_ecm365
PYTHONPATH=. python -m mlpf.data.key4hep.postprocessing \
  --detector ild --input /path/to/reco_p8_ee_qq_ecm365_500000.root \
  --outpath reports/ild/post/p8_ee_qq_ecm365
```

Track features come from the unique fitted state at the interaction point;
the first-hit radius comes from `AtFirstHit`. Invalid states or fit quality
fail conversion. Fully invisible events also fail explicitly; they are not
silently removed. A full-angle gun can produce such events outside detector
acceptance, so use an explicit angular acceptance for calibration samples. No truth momentum replaces fitted inputs. The common feature
width is 19 and the target width is 14. New EDM4hep cluster `iPhi` and legacy
`phi` are both supported.

The target policy is **`ild-direct-hit-majority-v1`**:

- Each distinct attached tracker measurement supplies at most one vote through
  digitization-to-simulation and simulation-to-MC identities. Duplicate links
  cannot inflate purity. A measurement shared by several distinct MC particles
  splits its vote; an unlinked measurement remains in the denominator.
- A track can label only its direct charged status-1 majority owner, at **75%**
  purity or higher. Secondary, conversion and low-purity tracks remain inputs
  without a primary track label. Tracker ancestry never substitutes a parent's
  production momentum for a measured secondary track.
- Calorimeter links use each particle's share of a reconstructed hit's truth
  weight, multiplied by the calibrated hit energy. Descendant deposits are
  propagated to status-1 ancestors. The common visibility rule requires a
  deposit above 10% of particle energy, or above 0.5 GeV for charged particles,
  or a qualifying track. The existing propagated-parent surrogate policy is
  applied to both targets and generator reference jets/MET.
- Visible particles compete for one exclusive track or cluster in descending
  truth energy. An unmatched particle merges its full four-vector into the
  representative of its largest-deposit cluster, if that cluster has a host;
  otherwise it is dropped and its lost energy is accounted explicitly. This
  retains the common Key4hep calorimeter assignment and merge convention.
- A cluster representative has neutral charge and the corresponding neutral
  class, including when its original source was charged. Track representatives
  have charged classes. Nonrepresentative inputs keep only their inclusive
  particle-number association. Generator jets/MET are a separate pre-visibility
  reference, not the sum of target rows.

Native `MarlinTrkTracksMCTruthLink` weights are retained for audit but are not
used for training labels: the ten-event qq check found values above one.
`metadata_premerge_energy`, particle counts and target energy preserve the
visibility/merge boundary for qualification.

`ycand` is **real Pandora output**, not an oracle. Multi-track PFOs receive one
representative: charged PFOs prefer their highest-momentum track; neutrals prefer
their highest-energy cluster, with the other input type as fallback. The complete
PFO momentum is stored once. Native ROOT retains every original association.
The audit checks baseline count and energy against the native PFO collection.

Upstream drops EDM4hep dQ/dx and tracker collection-encoding metadata. dEdx fields
therefore contain missing-value zeros. Tracker surface identifiers are unknown;
there is no ILD hit-dataset builder or hit-training mapping in this recipe.

## Qualification and TFDS

Run the independent native audit and the existing validation gates:

```bash
PYTHONPATH=. python -m scripts.studies.qualify_ild \
  /path/to/reco.root /path/to/reco.parquet --output reports/ild/native_audit.json
PYTHONPATH=. python tests/validate_parquet.py \
  --detector ild --input reports/ild/post/p8_ee_qq_ecm365 \
  --plots-dir reports/ild/validation
```

The native audit checks event preservation, collection/index identities, fitted
IP momenta, direct-primary tracking, angular acceptance, PFO count/energy
preservation, target majority provenance, unique representatives, merge energy
loss and target-jet constituent four-vector closure. Native truth-weight
warnings are reported separately from structural errors. Review both the JSON
and the unchanged validator thresholds. A structurally valid file may still
fail reconstruction quality gates.

Builders are `ild_edm_qq_pf`, `ild_edm_ttbar_pf` and `ild_edm_ww_fullhad_pf`,
configuration **1**, version **0.1.0**. They require compatible ILD provenance,
finite arrays and at least two independent generation jobs per process. Sorted
files are divided into disjoint train/test sets; events from the same ROOT job
cannot cross the split. Duplicate source ROOT checksums are rejected even if
Parquet filenames differ. Conversion that silently discards events is rejected.

```bash
tfds build mlpf/heptfds/ild_pf_edm4hep/qq.py --config 1 \
  --manual_dir reports/ild/post --data_dir reports/ild/tfds
```

## Qualification results, 17 November 2026

**Execution and structural integration pass; physics qualification fails.**
The artifacts are under `reports/ild-fccee-qualification-20261117/` in the
qualification workspace. They include ROOT and Parquet files, manifests,
native audits, validator reports and plots, and the TFDS smoke dataset.

The positive campaign converted **186 events** without skips: 110 qq events
at 365 GeV in independent 100- and 10-event jobs, three ttbar events, three
fully hadronic WW events, ten each of 10 GeV electrons, muons, charged pions,
photons and neutral kaons, and twenty 100 GeV muons restricted to
45–135 degrees. A separate fresh one-event runner test also passed. Native
audits found no broken references, invalid fitted-track records, baseline
double counting or target-jet four-vector closure errors in the positive
campaign. This tests pipeline execution, not detector resolution or efficiency
at production statistics.

The unchanged validator reports the following for the **100-event qq job**:

| Check | Result |
|---|---|
| Overall | 23 PASS, 3 FAIL, 4 WARN, no skipped checks |
| Assigned track direction, ΔR < 0.1 | **92.65%**, below the 95% gate; 1,484 representatives |
| Assigned cluster position, ΔR < 0.3 | **80.52%**, above the 80% gate; 2,664 representatives |
| Pandora/target total-energy disagreement | Median **3.21%**, p90 **39.52%**; p90 exceeds the 15% gate |
| Muon deposit check | Fails on ten low-energy qq muons; see the separate MIP control below |

Track alignment is 95.65% above 1 GeV truth pT and 98.33% above 5 GeV; the
all-track gate remains failed. Native direct-primary matching also has charge
mistakes and large momentum tails. No truth correction, removal of direction
outliers, or relaxed threshold was used. The native match counts for charged
status-1 particles above 1 GeV pT are 718/746 at |cosθ| < 0.8, 296/357 at
0.8–0.95, and 0/192 above 0.95. These counts use generator particles without
a hit requirement, and identify a forward-coverage limitation of the selected
final track collection rather than a complete tracking-efficiency measurement.

The target direction audit uses the retained charge in hit representatives to
distinguish original sources, rather than the forced-zero cluster target
charge. Neutral-source clusters align at **91.98%**, while charged-source
clusters align at **39.41%**. Charged transport and merged showers therefore
need separate interpretation from neutral calorimeter directions; the
aggregate gate has not been changed.

Native track-truth weights exceed one in 38 events, reaching **1.7143**.
Training labels use the independent distinct-measurement vote described above.
The 100-event file also contains **72 multi-track PFOs**; their complete count
and energy survive the one-representative baseline conversion.

The median retained/premerge target energy is **99.79%**. Loss is not uniformly
small: event 9 drops 163.59 GeV, including a **163.35 GeV forward photon** at
|cosθ| = 0.99639. Its linked hits include 146.89 GeV in `LCAL`; none of its
linked shower hits belong to `PandoraClusters`. This exposes the difference
between full-detector hit visibility and representability by the chosen PF
inputs. It is not repaired by relabelling another input. The complete
generator reference is retained separately, and losses remain explicit.

The twenty central 100 GeV muons pass the MIP-deposit check: **17/20** have
deposit/energy below 10% while depositing above 0.5 GeV. Median Pandora energy
response is 0.9915. The generic validator's total-energy reference is designed
for balanced collision events and fails for this one-particle gun because it
adds generator jet energy and generator missing transverse momentum; that
failure is not used as a gun-calibration criterion.

A separate twenty-event full-angle 100 GeV muon probe is retained under
`acceptance_probe/`. Event 10 has no detector hits, tracks, clusters or PFOs
at |cosθ| = 0.99851. Postprocessing fails explicitly on that event. It does
not silently turn a twenty-event job into a smaller dataset. Handling or
selecting fully invisible events must be decided before a large campaign.

TFDS built **100 train / 10 test** qq events from independent ROOT jobs. All
ten test examples read back with finite matrices; the MLPF loader assigned
source ID 7 and a two-event CPU attention forward pass produced finite outputs.
This is an untrained integration test. The focused and regression suites pass
**135 tests**, with one unavailable historical IDEA fixture skipped and seven
subtests passed. First-party lint and complexity checks and generator shell
syntax checks pass.

Keep version 0.1.0 experimental. The remaining work is reconstruction and
acceptance/target-definition qualification, not a TFDS or serialization fix.

## Comparing CLD and ILD distributions

Convert both detectors' ROOT files with the current postprocessor, placing
Parquet shards in `POST/{cld,ild}/{ttbar,qq,ww_fullhad}/`. Then run:

```bash
bash scripts/run_cld_ild_comparison.sh POST OUTPUT 200
```

This reuses `scripts/compare_detector_features.py` to generate a six-sample
comparison and one report per process. Reports include track and cluster
features on shared bins, target/truth jet spectra normalized per event,
one-to-one angular matching efficiency and purity, and matched jet pT and
energy responses. JSON summaries retain counts, ranges, quantiles and
structural checks. Feature histograms are normalized per object.

Use the generator spectra to assess sample differences. Equal generator-card
and seed labels across software versions do not establish identical events;
check generator identities before interpreting detector differences as an
event-paired measurement. CLD's existing track associations and ILD's direct
hit-majority target policy also differ, so this compares both reconstruction
and the resulting dataset definitions.

The 17 November 2026 comparison used two 100-event ROOT shards per process
and detector, **1,200 reconstructed events** in total. All twelve conversions
completed. Stable-particle PDG lists differ in every corresponding event,
confirming that shared seed names do not give paired generator events.

Each entry below is **CLD / ILD**; multiplicities are means per event:

| Process | Tracks | Labelled tracks | Clusters | Truth jets matched to target jets |
|---|---:|---:|---:|---:|
| ttbar | 52.53 / 71.16 | 46.92 / 41.35 | 95.19 / 101.07 | 97.93% / 98.61% |
| qq | 25.25 / 29.98 | 21.44 / 14.66 | 47.11 / 47.86 | 91.06% / 95.80% |
| WW full-hadronic | 40.03 / 50.73 | 34.58 / 26.60 | 75.55 / 79.03 | 97.62% / 97.47% |

Matching uses one-to-one ΔR < 0.1 with no response cut. Both detectors use
ee_genkt jets with R=0.4, p=-1 and pT > 5 GeV. Median matched target/truth pT
and energy responses are essentially one for all six samples. Targets carry
retained truth four-vectors, so this is a target-construction/retention check,
not a detector energy-resolution measurement.

ILD has more low-pT and unlabelled tracks and broader D0/Z0 tails. Tracks below
1 GeV pT comprise 57–64% of ILD inputs versus 42–47% for CLD. The fraction of
track inputs at |cosθ| > 0.95 is 0.4–0.7% for ILD versus 4.7–17.9% for CLD;
these are input distributions, not tracking efficiencies. Cluster spectra are
closer: ILD's mean multiplicities are 2–6% higher and median energies modestly
lower in these samples. Reconstruction and label-policy effects both enter.

All samples pass finite-value, matrix-alignment, phi-normalization, track
kinematics, target-class and target-jet-index checks. ET = E/cosh(eta) fails
for **three clusters per detector**: CLD has two in qq and one in WW; ILD has
one per process. One inspected ILD cluster has nonzero energy and position but
native iTheta=0, producing ET=eta=0 in the current converter. These rows remain
in the comparison and are recorded individually in its audit report.

Local artifacts are under `reports/cld-ild-comparison-20261117/`: `index.html`
links the combined and per-process plots; `input_audit.json` records target
coverage and the inconsistent cluster rows; `generator_comparison.json`
records the generator checks. The report does not supersede the failed
reconstruction/target-direction gates above.
