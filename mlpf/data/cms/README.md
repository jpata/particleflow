https://github.com/jpata/particleflow/wiki/CMS

## Truth-graph postprocessing prototype

`postprocessing_truth.py` builds a versioned, row-aligned representation from
HGCal NanoAOD and reconstructed-energy product-overlap weights exported from
CMSSW. It uses ordinary tracks, GSF tracks, barrel ECAL/HCAL PFClusters, and one
HGCal trackster collection (linked tracksters by default).

The assignment is intentionally small and comparable to `postprocessing2.py`:

1. Keep positive associations; require track purity > 0.75 and reserve GSF
   anchors for electrons.
2. Split a non-electron target across multiple matched ordinary tracks within
   dR < 0.4, in proportion to purity times track momentum. Electrons are not split.
3. Visit GSF, ordinary tracks, ECAL, HCAL and HGCal in that order. Within a
   collection, visit descending input pT and pick the strongest unassigned target.
4. Merge remaining targets onto their strongest supported input. Sum four-vectors;
   take PID and charge from the same largest-energy contributor.

Original truth is never overwritten. The sparse `provenance` array contains
`(output_row, original_truth_row, fraction)` and is checked for per-truth fraction
and total four-momentum conservation. `unmatched_truth` retains targets without
accepted input links. Splits have separate derived `target_id` values, while their
original graph IDs remain available. Mixed-label merges are flagged. Fractional
splitting copies categorical PID/charge, like the old code; it does not enforce
physical charge conservation.

The current HGCal NanoAOD `sharedEnergy` uses simulation-energy weighting. It
cannot be substituted for the old reconstructed-energy product overlap. Export
those weights and the complete, uncut status-1 generator reference first:

```bash
APPTAINER_NV=true ./env.sh python3 "$PWD/scripts/cmssw/extract_association_metrics.py" \
  /absolute/path/to/step3.root --skip-legacy \
  --output /absolute/path/to/run/truth_postprocessing_weights.json

.venv/bin/python -m mlpf.data.cms.postprocessing_truth \
  /absolute/path/to/run/step3_inNANOAODSIM.root \
  --associations /absolute/path/to/run/truth_postprocessing_weights.json \
  --output /absolute/path/to/run/truth_postprocessing.pkl
```

The output schema `cms_truth_graph_v1` contains minimal named input features
(`Xelem`), the familiar named particle target fields (`ytarget`), authoritative
float64 target four-vectors, original truth, generator truth, input identities,
provenance, and split/merge/unmatched diagnostics. `particle_number` identifies
**derived outputs**, not original truth particles. HGCal has input type 12; this
prototype is not directly wired into the existing Phase-1 CMS TFDS loader and does
not fabricate an aligned full-PF `ycand` baseline. The current sidecar reader
requires unique event numbers within a file and checks run/lumi, root identities,
root four-vectors and input indices. Collision/pileup use requires an inclusive
truth-root selection; the gun setup uses signal-seeded PFReconstructable roots.

### Validation

Current physics validation compares visible fiducial GEN status-1 particles
directly with the new assigned targets. The default validator produces jet and
MET median response and resolution plots without a normalizer compatibility
baseline. See [the current workflow and cuts](../../../scripts/cmssw/README.md#current-physics-validation-fiducial-gen-versus-new-targets).
The 1,000-event ttbar sample gives matched-jet median pT response 0.980 and
relative sigma68 resolution 3.56%; fiducial MET magnitude response is 0.981
with 9.07% relative resolution for GEN MET >= 20 GeV. These measure target
closure, rather than performance of independent reconstructed PF candidates.

The comparisons below describe historical implementation checks and earlier
reports. They are not the physics baseline used by the current validator.
For those checks, `truth_postprocessing_validation.py` calls the actual
`postprocessing2.prepare_normalized_table` on identical reconstructed inputs,
truth particles and accepted calibrated weights. HGCal uses the final legacy
calorimeter dispatch slot only in this test adapter; other missing old feature
fields are zero because assignment does not use them. This isolates assignment
from the previous ntuplizer's different truth definitions and detector coverage.
It is not an end-to-end legacy PFAnalysis ntuple comparison: that analyzer crashes
in its Phase-2 feature extractor on this release. The prototype also corrects the
old mixed-target convention that could take PID and charge from different parents.

```bash
.venv/bin/pytest -q tests/test_cms_truth_postprocessing.py

scripts/cmssw/run_mixed_gun_test.sh
# Export the resulting run's graph weights as above, then:
.venv/bin/python scripts/cmssw/validate_truth_postprocessing.py \
  --sample pion logs/cmssw/20261114T184048Z-10events.V08TyD \
  --sample mixed logs/cmssw/20261114T211857Z-mixedguns.pfpvT4 \
  --output reports/cmssw_truth_postprocessing
```

The mixed gun generates electron, muon, photon, K0L and pion particle/antiparticle
pairs with pT 0.7–10 GeV and |eta| < 3, ten events total. Validation compares
individual targets to visible status-1 generator particles using PID, charge,
dR < 0.1 and stored ancestry. It also groups descendants back to their generator
ancestor, so conversions and simulation interactions can be distinguished from
assignment errors. Input energy/momentum response and angular distances are
reported for every assigned track/cluster. The GEN reference comes from the RECO
file; the NanoAOD generator table has a pT cut and is not used as an uncut reference.

Results are in `reports/cmssw_truth_postprocessing/summary.json` and the per-sample
`validation.json`, `events.pkl`, `targets_vs_gen.png`, `targets_vs_inputs.png`, and
`particle_content.png`. On both ten-event samples, every event agrees with the
old normalizer's aligned four-vectors, PID and charge; maximum component difference
is below 4e-6 GeV. The pion sample yields 20 targets matching 20 generated pions.
The mixed sample yields 109 targets, including three merged rows, from 156
PFReconstructable roots and 100 generated particles. It retains 41 unmatched roots
with no input links. Total target energy is 96.9% of generator energy; 93/100 GEN
particles have ancestor-grouped assigned energy within 1%. Both samples require
zero track splits; deterministic tests exercise splitting and combined split/merge
cases. These are small gun validations, not dense-jet or pileup performance studies.

### Larger gun suite

Run `.venv/bin/python scripts/cmssw/run_gun_validation.py` in an environment where
`env.sh` can launch Apptainer. Defaults are 100 mixed events plus 50 events each of:

- Forward electron pairs: pT 5–50 GeV, 1.6 < |eta| < 2.8.
- Barrel photon, K0L and neutron pairs: pT 0.7–10 GeV, |eta| < 1.3.
- Muon, pion, kaon and proton pairs: pT 0.3–3 GeV, |eta| < 3.

The mixed sample retains the original five-species 0.7–10 GeV configuration.
Every sample uses independent explicit RNG seeds, two CMSSW threads and one
stream. Two samples run concurrently by default. `--events`,
`--specialized-events` and `--jobs` control these choices. Generated configurations,
seeds, stage exit codes, timestamps and logs live beside `manifest.json` under
`logs/cmssw/*-gun-validation/`. `--resume /absolute/path/to/suite` skips stages
that exited successfully and reruns validation. No existing sample is overwritten.

The suite exports calibrated weights and automatically runs the same legacy/GEN/
input comparisons. Extended diagnostics include closure by particle species and
barrel/endcap, original-root versus assigned-target energy, explicit legacy
mismatch rows, charge disagreement on tracks, and counts of input dR > 0.1 and
energy responses outside 0.5–2. These thresholds are diagnostics, not selection
cuts. Reports are under `reports/cmssw_truth_postprocessing_extended/<suite>/`.
Use `summarize_gun_validation.py REPORT_DIRECTORY` to regenerate `report.txt` and
`sample_comparison.png` from the saved validation outputs.

Completed suite: `logs/cmssw/20261114T214642Z-gun-validation`, with results in
`reports/cmssw_truth_postprocessing_extended/20261114T214642Z-gun-validation`.
All stages exited successfully; the expected 250 events and 1800 status-1 GEN
particles were processed. The actual legacy normalizer agrees on four-vectors,
PID and charge in all 250 events (largest component difference 1.64e-5 GeV).
Four real track splits (three pions, one kaon) and 35 merged rows exercise the
conservation/provenance checks on simulated data.

| Sample | Events | GEN | Targets | Target/GEN energy | GEN ancestor closure within 1% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed | 100 | 1000 | 1012 | 97.77% | 921/1000 |
| Forward electrons | 50 | 100 | 100 | 100.00% | 100/100 |
| Barrel neutrals | 50 | 300 | 285 | 94.54% | 245/300 |
| Low-pT charged | 50 | 400 | 390 | 99.26% | 389/400 |

There are 269 unmatched roots: 265 have no raw input links and four have only
rejected track associations. These losses occur in both normalizers. Input
comparisons expose reconstruction/association limitations despite exact assignment
parity: two forward GSF tracks have the wrong reconstructed charge, including one
with pT/target pT = 10.8; five low-pT charged particles are anchored to ECAL deposits
with energy response 0.045–0.223 and dR 0.91–2.83. The latter are retained by the
hit-overlap policy in both implementations. The measurements do not establish the
physical cause of these outliers or justify an additional angular cut.
`assignment_outliers.json` records large-angle calorimeter anchors, track charge
flips and track pT response > 2. `lowpt_calo_cases.json` also records the five ECAL
cases with original-truth provenance. The assignment policy was unchanged during
this expanded validation; neutral coverage and calorimeter tails remain limitations
for training-label quality.

### PU0 ttbar validation

Run `.venv/bin/python scripts/cmssw/run_ttbar_pu0_validation.py --events 20`
(or `--resume /absolute/path/to/run`). This uses the release's official
`TTbar_14TeV_TuneCP5_cfi` fragment (14 TeV, m(top) = 175 GeV), Run4D121 and the same
GEN-SIM / DIGI / RECO+HGCALVal NanoAOD chain as the gun tests. All stages use
`mixNoPU`; Pythia's underlying event remains enabled. Configurations, RNG seeds,
exit codes and logs are retained beside a sample manifest.

The current PFReconstructable walk starts at `LevelFlag::Signal`. On a collision,
`seedPdgIds=[0]` keeps the full graph but stamps no signal seeds, so this would
produce an empty PF truth selection. Top-only seeds would miss the underlying
event. For this pp sample the runner instead seeds the graph on the two incoming
beam protons (`[2212,-2212]`) and clears the optional reconstructable-PDG list,
retaining a photon-level reference. This is an explicit sample configuration;
no changes to the C++ graph algorithms are required. The exporter records the
actual seed PDGs/statuses, last-copy tops, per-root bunch crossing/event index,
and detector-hit particles not covered by any PF-root subgraph.

The exporter supports `--skip-legacy --sparse`: only positive calibrated
calorimeter overlaps are serialized, and track links are read from NanoAOD.
Sparse exports omit the optional old-Caloparticle metrics (`old_cp`/`cp_link`
are null). On the mixed gun control this reduced 14,381 rows to 235 positive
calorimeter edges with exactly identical calibrated weights, output targets and
provenance. Full dense exports remain available for association-metric studies.

Collision validation retains uncut visible GEN comparisons and adds a separate
fiducial reference: status 1, excluding neutrinos, pT > 0.5 GeV and |eta| < 3.
Full-event energy includes very forward beam remnants outside these inputs'
coverage. `targets_vs_gen_fiducial.png` shows the selected GEN/old/new kinematic
distributions and ancestry-grouped energy closure. Top-row targets receive the
same kinematic cuts on their own vectors; bottom-row contributions are grouped
by the selected GEN ancestors. Species and detector-region summaries distinguish
barrel, endcap (1.5 <= |eta| < 3), and |eta| >= 3.

Completed physics sample: `logs/cmssw/20261114T221700Z-ttbar-pu0`, with reports in
`reports/cmssw_truth_postprocessing_ttbar/20261114T221700Z-ttbar-pu0`.
All three CMSSW stages and the association export succeeded. Generator checks
confirmed a last-copy t/tbar pair in every event. The 20 events contain 12,034
status-1 GEN particles (11,984 visible), 13,345 PF truth roots, 6,814 reconstructed
inputs and 4,498 assigned targets. There are four track splits and 254 merged rows.
All roots have bunch crossing/event index zero, all 54,624 detector-hit particles
are covered by PF-root subgraphs, and every root has a unique status-1 GEN ancestor.
These checks are saved in the run's `sample_checks.json`.

Four-vectors and PID agree with the actual old normalizer in all 20 events, with
maximum component difference 3.31e-5 GeV. Raw charge arrays agree in six events;
33 merged output rows across the other 14 events expose the documented old
PID/charge convention. For example, the old code labels a merged photon with
charge +1 from a lower-energy positron. The new code takes both PID and charge
from the same highest-energy parent. Each difference is independently verified
against provenance and the accepted edge weights; no unexplained legacy
mismatches remain. The raw comparisons are preserved alongside this classification,
and `charge_convention_differences.json` lists all parent contributions.

For visible status-1 GEN particles with pT > 0.5 GeV and |eta| < 3:

- 3,473 GEN particles; 2,735 have assigned descendant energy within 1%.
- Assigned descendant energy is 93.72% of GEN energy; original-root energy is
  100.13%, so most of this deficit comes from missing accepted input support.
- 473 GEN particles have no assigned descendants.

The full-event target/GEN energy ratio is only 11.88%; the denominator includes
very forward beam remnants and these inputs contain no HF collection. It is not
an efficiency for the fiducial region. Of 8,518 unmatched roots, 8,474 have no raw
input links and 44 have only rejected track links.

Assigned ordinary tracks have median pT response 1.00016 and median dR 0.00431.
Calorimeter assignments have broad tails: median dR is 0.0342 (ECAL), 0.2733
(HCAL) and 0.0566 (HGCal); respective counts above dR 0.1 are 178/692, 97/126
and 206/534. Some tiny targets anchor large clusters, producing extreme response
ratios. The log-scale `targets_vs_inputs.png` and `assignment_outliers.json`
retain these cases. Legacy parity and four-momentum conservation do not establish
that every calorimeter anchor is a clean training label. No angular or response
cuts were introduced during validation.

### Transverse-activity checks for physics events

For ttbar, the scalar sum of visible-particle pT is more informative than the
full-event energy sum, which is dominated by beam remnants. Run:

```bash
MPLCONFIGDIR=/tmp/particleflow-matplotlib .venv/bin/python \
  scripts/cmssw/validate_transverse_activity.py \
  reports/cmssw_truth_postprocessing_ttbar/20261114T221700Z-ttbar-pu0/ttbar_pu0/events.pkl \
  --output reports/cmssw_truth_postprocessing_ttbar/20261114T221700Z-ttbar-pu0/transverse
```

`transverse_activity.png`, `.json` and `report.txt` compare scalar sum pT,
transverse-vector imbalance, and ancestor-grouped fiducial pT closure. In the
20-event sample, visible GEN sum pT ranges from 320 to 1137 GeV (median 658 GeV).
Median target/GEN scalar sum pT is 77.90% inclusive and 94.24% for pT > 0.5 GeV,
|eta| < 3; summing over events gives a fiducial ratio of 94.08%. The median
fiducial transverse-vector residual is 6.73 GeV. Merging reduces scalar sum pT
by a median 0.435 GeV/event, while the transverse vector is conserved against
represented roots to 1.3e-13 GeV. Including neutrinos, the full GEN transverse
vector is zero within 1.1e-5 GeV. Scalar sum pT is not a conserved quantity under
non-collinear vector merging.

These measurements still include the underlying event. A naive top-descendant
GEN selection does not uniquely isolate hard-process hadrons: shared color
strings can connect top-decay partons to beam remnants. The diagnostic
`audit_ttbar_ancestry.py` demonstrates this on the same events; its outputs under
`logs/cmssw/20261114T221700Z-ttbar-pu0/ancestry_audit/` are explicitly ancestry
audits, not a hard-only postprocessing dataset. A hard-process reference requires
an explicit parton-level or matched-jet definition. The existing full-event
postprocessing and truth associations remain unchanged.

### Target destinations by detector element

`plot_target_assignments.py EVENTS_PICKLE --output DIRECTORY` produces count and
species-normalized destination heatmaps, a fiducial version, pT-binned destination
fractions, and merge/angular/electron-ancestry diagnostics. The full per-row mapping
is in `assignments.json`; `report.txt` explains the normalization. Counts describe
assigned output targets after splitting/merging, not primary GEN particles or
unmatched original roots. Merged rows use the largest-energy contributor's PID.

For the 20-event PU0 ttbar sample, outputs are in
`reports/cmssw_truth_postprocessing_ttbar/20261114T221700Z-ttbar-pu0/assignments/`.
94.7% of pion, 97.7% of charged-kaon and 89.7% of proton/antiproton targets are
on ordinary tracks. All 24 muon targets are on tracks. Photon targets go to
ECAL (58.2%), HGCal (40.7%) and HCAL (1.1%). Electron targets are mostly photon
descendants: 411/449 have a GEN photon ancestor for their dominant contribution;
this explains why the electron-labelled population is not equivalent to primary
GEN electrons. Nine neutral targets are on charged tracks (seven K0S, one K0L,
one neutron); all are single-root assignments. The current association accepts
branch tracker-hit purity without a charged-target requirement. These are useful
cases for reviewing the assignment policy, separate from mixed-label merges.

For larger productions, the runner accepts explicit concurrency and independent
seeds, for example:

```bash
.venv/bin/python scripts/cmssw/run_ttbar_pu0_validation.py \
  --events 1000 --threads 8 --streams 8 --seed-base 185001
```

The 1000-event run launched with these settings is
`logs/cmssw/20261114T225915Z-ttbar-pu0-1000ev-8threads`.
Its detached launcher log is the adjacent `.launcher.log`; `status.json` records
stage exit codes and timestamps, and `progress.jsonl` records periodic event
progress. `latest_ttbar_1000_launch.json` records the launcher PID and command.
All three CMSSW stages use eight threads and eight concurrent event streams.

The 1000-event production completed GEN-SIM, digitization and RECO/NanoAOD with
exit code zero in all three stages. `production_checks.json` verifies 1000 events
in each ROOT file, unique NanoAOD event IDs, nonempty PF truth in every event,
and zero bunch-crossing/sub-event indices for all truth roots. NanoAOD contains
628,258 truth roots, 150,577 ordinary tracks, 6,099 GSF tracks, 103,037 ECAL
clusters, 24,095 HCAL clusters and 40,289 linked HGCal tracksters. Its file is
`step3_inNANOAODSIM.root` (231,895,447 bytes). Association-export and postprocessing
status are recorded separately by the running pipeline.

The 1000-event sample's postprocessing is complete under
`reports/cmssw_truth_postprocessing_ttbar/20261114T225915Z-ttbar-pu0-1000ev-8threads/`.
`finish_ttbar_validation.py RUN_DIRECTORY` runs/resumes validation, summary,
assignment plots and transverse plots, recording each exit code in
`postprocessing_status.json`. Use a persistent launcher for a long full-sample pass.

The larger sample exposed a reader precision issue: rebuilding truth px/py from
float32 phi can exceed the component-wise identity tolerance near a zero component.
The reader now uses the stored Cartesian truth columns when available, retaining
both the strict identity check and the angular fallback for older schemas. The
regression test uses the actual failing vector (event 19, truth row 72).

Results: 210,263 targets, 277 track splits and 11,833 merged rows. In the historical
implementation compatibility check (archived in `historical_assignment_compatibility/`), target p4 and
PID agree with the actual old normalizer in all 1000 events; 1,327 merged charge
differences are all accounted for by the documented dominant-parent correction.
No unexplained legacy differences remain. Median fiducial sum-pT response is
94.22%, aggregate response 94.44%, and median transverse-vector residual 5.68 GeV.
There are 695 neutral targets on charged tracks, retained as assignment diagnostics.
Fourteen represented roots have no status-1 GEN ancestor; an all-status GEN check
identifies them as status-2 hadrons/resonances. Their 1889.20 GeV total assigned
energy remains in global sums, and their absence is explicitly flagged in
per-status-1-ancestor closure. The pT analysis therefore does not invent a unique
ancestor or silently remove these contributions from event totals.
