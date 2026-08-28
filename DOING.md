# IDEA simulation integration: status and next steps

Last updated: 2026-08-28

## Objective

Add FCC-ee IDEA detector simulation to the Key4HEP event-production workflow
under `mlpf/data/key4hep/gen`, then determine whether the reconstructed
EDM4hep output contains the inputs and truth associations needed to build ML
particle-flow datasets.

## Work completed

- Added an `idea` production entry to `particleflow_spec.yaml` and a Tallinn
  `gen_idea` task to `configs/tallinn/pixi.toml`.
- Updated the Snakemake production setup to recognize generation-only
  detector workflows. IDEA is deliberately generation-only for now because
  compatibility with the common MLPF postprocessor has not been established.
- Added `mlpf/data/key4hep/gen/idea/run_sim.sh`. It performs:
  1. Pythia8 event generation using the existing CLD cards.
  2. IDEA `IDEA_o1_v03` simulation with `ddsim`.
  3. Official FCC-config digitization and reconstruction.
  4. Copying the final EDM4hep ROOT file into the normal production layout.
- Added the official FCC-config repository as a nested submodule, pinned to
  commit `a05a3a9865cb1d54aefc57944de078f971e0cd05`, matching the referenced
  IDEA tutorial.
- Switched the environment to Key4HEP release `2026-04-08`:

  ```bash
  source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2026-04-08
  ```

- Enabled `setupDRCFastSim` in the IDEA steering file. This installs
  `Geant4DRCFiberModel` and Geant4 fast simulation for optical photons in the
  dual-readout fibers. It is not a parametrized fast simulation of the full
  detector: tracker transport and electromagnetic/hadronic showers still use
  full Geant4.
- Added a process heartbeat to `run_sim.sh`. During `ddsim` and reconstruction
  it prints CPU usage, CPU time, memory, process state, and kernel I/O. The
  interval is controlled with `PROGRESS_INTERVAL` and defaults to 60 seconds.
- Added particle-gun support to the same runner. Process names beginning with
  `gun_` bypass Pythia and use the DD4hep gun. The production specification
  contains a one-event `gun_e_10gev` validation sample; `GUN_PARTICLE`,
  `GUN_ENERGY`, and `GUN_DISTRIBUTION` are configurable for local runs.
- Downloaded the IDEA particle-flow workshop presentation to
  `mlpf/data/key4hep/gen/idea/docs/250902_PFworkshop.pdf` and summarized it in
  `mlpf/data/key4hep/gen/idea/docs/IDEA_particle_flow_status.md`.
- Documented the IDEA production entry in
  `mlpf/data/key4hep/gen/idea/README.md`.

## Validation observed so far

- One-event Pythia generation succeeds for `p8_ee_qq_ecm365` with seed
  `424242`.
- IDEA geometry loads successfully from the Key4HEP `K4GEO` installation.
  Geometry initialization is intrinsically heavy: the log reported roughly
  260 million geometry nodes, around 80,000 sensitive paths, and about two
  minutes for conversion to Geant4 in the tested environment.
- The log confirms that the requested fast model is active:

  ```text
  FastPhysicsList INFO Enable fast simulation for particle type: opticalphoton
  FastPhysicsList INFO Constructed and initialized Geant4 Fast Physics
  ```

- The 365 GeV hadronic event entered Geant4 transport successfully. The
  earlier Codex-run validation was interrupted before the event completed, so
  its 222-byte `IDEA_sim.root` was only an unfinalized placeholder and cannot
  be inspected.
- Non-fatal warnings seen during transport included generator/PDG mass
  differences for unstable particles and `G4OpBoundaryProcess` `StepTooSmall`
  warnings. Geant4 explicitly classified these as warnings and continued.
- This Key4HEP build fell back to single-threaded Geant4. Even with
  `setupDRCFastSim`, a 365 GeV qq event can therefore take substantial time.
- Reconstruction initially failed because the pinned FCC-config expects
  `GGTFTrackFinder` and `GenfitTrackFitter`, while Key4HEP `2026-04-08` ships
  k4RecTracker 0.7.0 with the older `GGTF_tracking` component and no compatible
  fitter. The steering now detects this API and falls back to
  `GGTF_tracking`, producing `PrefitTracks` but not `FittedTracks` or
  `FittedTracksWithFilteredHits` on this release.
- The compatibility fallback was configuration-tested under the actual
  Key4HEP `2026-04-08` environment. The complete steering loads, selects
  `GGTF_tracking`, configures its output as `PrefitTracks`, and schedules no
  unavailable fitter.
- The first gun reconstruction produced a 28 KB PODIO file with metadata but
  zero event frames. `GGTF_tracking` rejected `DCHDigis` because the legacy
  finder requires `extension::SenseWireHitCollection`, while `DCHdigi_v02`
  produced `edm4hep::SenseWireHitCollection`. The fallback now selects the
  coherent legacy pair: `DCHdigi_v01` writes `DCH_DigiCollection` in the
  extension EDM and `GGTF_tracking` consumes it. Newer stacks continue to use
  `DCHdigi_v02` with `GGTFTrackFinder`.
- The corrected reconstruction was run against the existing gun simulation
  and completed successfully for all 100 input events. The corrected file is
  `/tmp/idea-gun/work/IDEA_sim_digi_reco_fixed.root` (about 70 MB); it has 100
  `events` frames plus metadata. The original 28 KB
  `IDEA_sim_digi_reco.root` remains the failed metadata-only output.
- The gun simulation used 100 events because `NEV=1` was not present in the
  effective manual invocation. This supplied a useful stability test but was
  not the intended minimal runtime test.
- Measured corrected-output occupancies across the 100 events include:
  - DCH sim/digi/association collections: mean 189.68 entries, populated in
    96/100 events;
  - `PrefitTracks`: mean 2.74, populated in 99/100 events;
  - truth-created tracks and associations: mean 1.18, populated in 93/100;
  - Cherenkov digi hits: mean 235.90, populated in 100/100;
  - scintillation digi hits and sim links: mean 5815.81, populated in 100/100;
  - topological clusters: mean 86.27, populated in 100/100;
  - cluster-to-MC links: mean 20.10, populated in 93/100.
- The corrected reconstruction log contains no ERROR or FATAL messages. One
  warning remains from `topoClusterAll`: the input-cell and system-ID list
  sizes differ, so tower-tool metadata is not written. This did not prevent
  clusters or event output from being produced, but should be investigated.
- The corrected reconstruction was also run successfully on the completed
  one-event 365 GeV qq simulation at
  `/tmp/particleflow-idea-fast-test/work/IDEA_sim.root`. The result,
  `IDEA_sim_digi_reco_fixed.root`, is about 26 MB and contains one event frame.
  Its populated collections include 17,028 DCH hits and associations, 193
  `PrefitTracks`, 46 truth-created tracks and associations, 402,284
  scintillation digi hits, 10,491 Cherenkov digi hits, 3,653 topological
  clusters, 4,826 calorimeter-hit-to-MC links, and 193 cluster-to-MC links.
  Reconstruction completed without ERROR or FATAL messages. Algorithm CPU
  time was about 327 seconds, dominated by `GGTF_tracking` at 266 seconds;
  truth-link creation took 29 seconds. The same non-fatal topocluster metadata
  warning remained. The complete log is
  `/tmp/particleflow-idea-fast-test/work/reconstruction_fixed.log`.
- `run_sim.sh` now stores separate persistent stage logs under
  `${WORKDIR}/logs`: `generation.log` (Pythia samples only), `simulation.log`,
  and `reconstruction.log`. Heartbeat messages are included in the relevant
  stage log as well as printed to the terminal.
- Added an initial IDEA track/cluster path to
  `mlpf/data/key4hep/postprocessing.py` and registered IDEA with a 2 T field in
  `mlpf/conf.py`. This proof of concept:
  - reads truth-seeded `TracksFromGenParticles` and
    `TracksFromGenParticlesAssociation`;
  - reads `TopoClusterAll` and `ClusterMCParticleLinks`;
  - writes track and cluster elements, target particles, truth jets, target
    jets, and target jet indices;
  - writes schema-compatible empty tracker/calo hit arrays; and
  - omits `ycand_track` and `ycand_cluster`, because the IDEA tutorial does
    not produce reconstructed PFOs.
- Updated `tests/validate_parquet.py` so intentionally empty hit collections
  can be validated without losing their expected two-dimensional shape after
  an Awkward/Parquet round trip.
- Successfully postprocessed the one-event qq reconstruction to
  `/tmp/particleflow-idea-fast-test/work/IDEA_sim_digi_reco_fixed.parquet`.
  The validation report and plots are under
  `/tmp/particleflow-idea-fast-test/work/validation-idea`.
- Analyzed status-1 truth coverage in the completed 100-event electron-gun
  reconstruction. Directly, 91 events have both a track and cluster link and
  9 have neither. After recursively propagating descendant associations to
  status-1 ancestors, 92 have both links, one is track-only, one is
  cluster-only, and six have neither. The directly unlinked electrons are all
  very forward (`|eta| >= 3.13`).
- Repeated the association analysis on the completed 10-event qq sample in
  `particleflow-idea-test-n10`. Among 517 visible, non-neutrino status-1
  particles, the direct track/cluster categories are 268 neither, 47
  cluster-only, 169 track-only, and 33 both. After descendant propagation
  they are 233 neither, 45 cluster-only, 195 track-only, and 44 both. Photons
  dominate the missing calorimeter associations: their propagated categories
  are 211 neither, 36 cluster-only, 22 track-only, and 2 both.
- Traced an example unlinked photon in event 0 (MC particle 122, energy
  32.958 GeV, eta 0.137). Its shower family has 4,226 descendants and 1,011
  relevant scintillation truth-link entries covering 807 unique cells. Those
  detector cell IDs occur in 67 topological clusters, but the written cluster
  links do not identify the photon family.
- Identified the missing-link cause in `CreateTruthLinks`. `topoClusterAll`
  combines scintillation and Cherenkov digi-hit collections and, with
  `createClusterCellCollection=True`, copies their cells into
  `TopoClusterAllCells`. The upstream truth linker compares PODIO object
  identities/indices, so copied-cell indices are incorrectly interpreted as
  indices in an original input collection. The resulting cluster links are
  accidental index matches rather than detector-cell matches.
- The same truth-link implementation was also the reconstruction bottleneck.
  For the prior 10-event run, `CreateTruthLinks` used about 177 minutes total
  (17.7 minutes/event average and 38.8 minutes maximum), versus about 15
  minutes total for `GGTF_tracking`. Inspection found nested scans over every
  MC particle for every hit link and cluster, and over every calorimeter truth
  link for every cluster cell. It exposes no upstream tuning properties.
- Added a local checkout of k4RecCalorimeter v0.1.0pre18 at
  `mlpf/data/key4hep/gen/idea/k4RecCalorimeter-local` and patched
  `CreateTruthLinks` to:

  - Associate copied cluster cells through their detector `cellID`.
  - Index calorimeter truth contributions by `cellID` instead of repeatedly
    scanning the complete link collection.
  - Emit only nonzero MC-particle contributions instead of scanning the full
    MC collection.
  - Expose a `minCellEnergy` property and skip digi cells at or below this
    threshold. It is configured to `0 GeV`, which removes the many zero-energy
    scintillation cells retained by the SiPM digitizer without discarding
    positive-energy cells.
- Added `mlpf/data/key4hep/gen/idea/build_k4reccalorimeter.sh` to configure,
  build, and install the local module against Key4HEP `2026-04-08`.
  `run_sim.sh` now activates that install before reconstruction and fails with
  a clear build instruction if it is absent. The reconstruction steering
  keeps `createClusterCellCollection=True`, with a comment explaining that
  the local truth linker supports copied cells by `cellID`, and configures the
  zero-energy cut with a second explanatory comment.
- Built the patched module successfully and validated it on qq event 0. The
  final validation output is
  `particleflow-idea-test-n10/work/IDEA_sim_digi_reco_local_truthlinks_cellid_event0.root`.
  It contains 4,388 clusters, 14,454 cell-level truth links, and 3,977
  cluster-level truth links. The example photon family now has 110 links to
  65 clusters, including 60 direct links to MC particle 122.
- In that validation, `CreateTruthLinks` took 0.66 seconds and the whole event
  took about 331 CPU seconds, now dominated by `GGTF_tracking` at 276 seconds.
  An intermediate ObjectID-based test made truth linking fast but forced slow
  cross-collection cluster relations (`topoClusterAll` took 32 minutes); the
  final `cellID` implementation with copied cells reduces topological
  clustering to 19.5 seconds.
- Completed the full 10-event reconstruction with the local truth linker. The
  finalized 242 MB output is
  `particleflow-idea-test-n10/work/IDEA_sim_digi_reco_local_truthlinks_cellid_full.root`
  and the log is
  `particleflow-idea-test-n10/work/reconstruction_local_truthlinks_cellid_full.log`.
  The ROOT file contains all 10 events, `TopoClusterAll`, and
  `ClusterMCParticleLinks`, with no ERROR or FATAL log entries. Wall time was
  about 22 minutes. `CreateTruthLinks` used 4.2 seconds total (0.42
  seconds/event), while `GGTF_tracking` used 15.4 CPU-minutes.
- Diagnosed a separate postprocessing slowdown. Uproot's default
  multithreaded local-file source stalled for many minutes even when reading a
  single small MC-particle branch. The same branch loads in about 0.01 seconds
  through `uproot.source.file.MemmapSource`. The IDEA postprocessor now uses
  the memory-mapped source, prints flushed timing/progress messages for ROOT
  loading, truth preparation, individual events, and Parquet writing, and
  applies `--first-event`/`--num-events` during the ROOT read rather than only
  limiting the later Python loop.
- Successfully postprocessed all 10 corrected events in 93.7 seconds. The
  1.5 MB output is
  `particleflow-idea-test-n10/work/postprocessing_local_truthlinks_cellid_full/IDEA_sim_digi_reco_local_truthlinks_cellid_full.parquet`.
  ROOT loading took 0.2 seconds; event processing took roughly 6--14 seconds
  per event. No cluster-energy cut was needed, so the validation retains the
  full reconstructed association information.
- Validated the corrected Parquet in report mode. The report and plots are in
  `particleflow-idea-test-n10/work/validation_local_truthlinks_cellid_full`.
  Of 517 visible, non-neutrino status-1 particles, 483 (93.4%) now receive a
  track or cluster representation, compared with 284 (54.9%) after descendant
  propagation in the broken-link reconstruction. The validator reports 14
  PASS, 1 FAIL, 10 WARN, and 1 SKIP.

- Diagnosed and fixed a simulation-side quadratic lookup in
  `FiberDRCaloSDAction`. The action added hits with a `cellID` key but searched
  for them with `find(CellIDCompare(...))`, which falls back to a linear scan
  after checking the previous hit. With hundreds of thousands of fiber hits,
  first-time misses alone required tens of billions of comparisons per event.
  All three fiber-hit paths now use the collection's keyed `findByKey` lookup.
  Dense optical-output map access also avoids duplicate `find` plus `at`
  lookups.
- The same three keyed fiber-hit lookups were independently merged upstream in
  k4geo PR #620 (`a9d5b72`) and are included in tag `v00-26` (`9998cf2`). The
  local runtime benchmark therefore also validates the upstream performance
  fix. The separate Cherenkov ancestry/output changes and the small dense-map
  lookup cleanup are not present in `v00-26`.
- Rebuilt the local k4geo module and reran qq event 0 with the same input and
  IDEA event seed. Event processing and output took 348.72 CPU seconds, versus
  3942.84 seconds/event averaged over the prior 10-event IDEA run. The same
  event takes 30.53 seconds in CLD, leaving an approximately 11.4x IDEA/CLD
  difference after the lookup fix. IDEA startup remains about 339 seconds and
  is dominated by construction and conversion of the 260-million-node
  geometry plus the DRC sensitive-volume scan.
- The keyed-lookup output is
  `/tmp/idea-speedup/IDEA_sim_keyed_event0.root`. It has exactly the same
  collection names and entry counts as the original event 0. Scintillation,
  Cherenkov, DCH, silicon, optical time/wavelength, and calorimeter-truth
  payloads were compared; the 3,086,058 scintillation contributions, 19,903
  Cherenkov contributions, and their particle relations are identical.
- A proposed zero-deposit-step removal was tested but not retained. Although
  most stored scintillation contributions are zero energy, their track marks
  influence DDG4's stored-ancestor selection, and zero-energy cells also enter
  the downstream SiPM dark-count model. Preserving them keeps detector and
  truth semantics unchanged.
- The next 100-event qq simulation plus reconstruction can be launched with:

  ```bash
  NEV=100 \
  PROGRESS_INTERVAL=60 \
  OUTDIR=/tmp/idea-100/output \
  WORKDIR=/tmp/idea-100/work \
  bash mlpf/data/key4hep/gen/idea/run_sim.sh p8_ee_qq_ecm365 424242
  ```

  The reconstructed output will be
  `/tmp/idea-100/output/p8_ee_qq_ecm365/root/reco_p8_ee_qq_ecm365_424242.root`,
  with persistent per-stage logs under `/tmp/idea-100/work/logs`. Based on the
  optimized simulation benchmark and the existing reconstruction timings,
  budget roughly 13--15 hours on one CPU core; individual events can vary
  substantially with tracking occupancy.

### Initial Slurm campaign: memory and progress observations

- The IDEA production is configured for 100 generation jobs per sample and
  exports `PROGRESS_INTERVAL=60` to every generated wrapper. The first
  submitted campaign still used an `8000M` generation-memory request.
- Job `60855092` (`qq`, seed `500041`) was monitored at five-minute intervals.
  It converted the geometry in 167.7 seconds, saved events 0 and 1, and began
  event 2. Initially it made clear progress: at 13:17 elapsed the main step
  had accumulated 12:16 CPU, the log was growing, and peak RSS was about
  7,482,804 KB.
- The job then stopped making application-level progress. The log remained at
  event 2 and was unchanged for about 28 minutes. In the last several
  five-minute intervals CPU advanced by only 48--52 seconds, disk writes were
  flat, and peak RSS was pinned at 8,256,552 KB against the `8000M`
  allocation. This is strong evidence of severe memory pressure or associated
  blocking, but not a proven Slurm OOM: the job was manually cancelled after
  43:19 rather than killed by the memory controller.
- The heartbeat did not provide its intended 60-second records. Its first
  `ps` invocation inside the Key4HEP container failed because
  `libsystemd.so.0` was unavailable, after which no actual `[progress]` lines
  were emitted. Slurm CPU, RSS, and I/O counters were therefore used for the
  monitoring decision. The heartbeat should be changed to avoid depending on
  the container's `ps` executable or its incomplete runtime libraries.
- IDEA generation memory is now `10000` MB in `particleflow_spec.yaml`, and
  the regenerated `snakemake_jobs/idea/Snakefile_gen` contains
  `mem_mb=10000`. Existing submitted jobs keep their original allocation;
  only cancelled and resubmitted or newly submitted jobs receive 10 GB. The
  next run should verify the effective Slurm `ReqMem`, peak RSS, event-save
  cadence, and whether event 2 completes before treating 10 GB as sufficient.

## Current validation state

The original association measurements remain a useful broken-link baseline,
but the conclusion that most photons intrinsically lack calorimeter truth is
no longer valid. The local `cellID` implementation fixes the traced photon and
makes truth association negligible relative to tracking.

A new 10-event qq simulation and reconstruction in
`particleflow-idea-test-n10-newdigi` validates Cherenkov truth end to end. The
simulation contains `DRcaloSiPMreadoutSimHitContributions`, reconstruction
contains `DRcaloSiPMreadoutDigiHit_cheren_link`, `CaloHitMCParticleLinks`, and
`ClusterMCParticleLinks`, and every one of the 62,413 Cherenkov simulation hits
has a corresponding optical digi-hit link. Photon-count closure is exact and
all contribution-to-MCParticle indices are valid. Both scintillation and
Cherenkov source collection IDs occur in the combined cell links. The median
sum of stored MC-particle weights per linked cluster is 1.0 in every event.

Removed the postprocessing renormalization of propagated cluster-link columns.
`ClusterMCParticleLinks.weight` is already the particle's cluster-energy
fraction; ancestry propagation intentionally leaves both an original relation
and its status-1 ancestor copy in the full MC matrix. Renormalizing the column
after propagation diluted the valid status-1 contribution. The IDEA target
measurement now uses `cl_weights @ cluster_energy` directly, with focused
regression tests.

The local `CaloTopoClusterFCCee` now handles dual-readout cells explicitly.
Previously both input collections were concatenated and later inserted into a
map keyed only by `cellID`, so the retained channel could depend on input
order. Equal physical cell IDs are now merged into one topology cell, while
per-input energy maps retain the two signals. Each cluster stores named shape
parameters in this order:

```text
dR_over_E, energy_cherenkov, energy_scintillation
```

The combined cluster energy is the sum of the two channel energies. The local
module builds successfully against Key4HEP `2026-04-08`. The resulting full
10-event reconstruction is
`particleflow-idea-test-n10-newdigi/work/IDEA_sim_digi_reco_separate_dr_energy.root`,
with log
`particleflow-idea-test-n10-newdigi/work/logs/reconstruction_separate_dr_energy.log`.
It finalized without ERROR or FATAL messages. `CreateTruthLinks` averaged
0.487 seconds/event; reconstruction remained dominated by `GGTF_tracking`.

Postprocessing reads the new shape parameters. To preserve the common tensor
width, IDEA maps Cherenkov energy to the legacy `energy_ecal` component slot,
scintillation energy to `energy_hcal`, and sets `energy_other` to zero. These
names denote readout channels rather than detector subsystems for IDEA. Legacy
IDEA files with only `dR_over_E` retain the previous combined-energy fallback.
The processed file is
`particleflow-idea-test-n10-newdigi/work/postprocessing_separate_dr_energy/IDEA_sim_digi_reco_separate_dr_energy.parquet`.
Across 29,388 clusters, the maximum discrepancy in
`energy - energy_cherenkov - energy_scintillation` is `7.63e-6 GeV`.

Validation gate `R2` is now detector-specific. CLIC/CLD and other detectors
retain the deposited-energy fraction check. For IDEA, R2 instead requires
nonnegative channel energies, zero `energy_other`, and closure of combined
cluster energy to the two channel components. The new IDEA validation has no
failures: 15 PASS, 0 FAIL, 10 WARN, and 1 SKIP. Its report is
`particleflow-idea-test-n10-newdigi/work/validation_separate_dr_energy_idea_gate/validation_report.json`.
R2 checks all 29,388 clusters and finds zero bad closures, negative
components, or nonzero `energy_other` values.

The generic single-cluster response warning is not an IDEA energy-closure
measurement. A topological cluster is a shower fragment, and the combined
Cherenkov-plus-scintillation calibrated signal need not be bounded by generator
energy. A future physics-quality response must use a defined dual-readout
calibration and sum all fragments associated with a particle.

`TopoClusterAll` remains highly granular, so an individual cluster should be
treated as a shower fragment rather than expected to contain a particle's
complete energy. All fragments assigned to the same particle should share a
`particle_number`, with one fragment chosen as the target representative.
Energy closure should be evaluated using the sum of all associated fragments,
not the representative cluster alone. The current proof of concept propagates
inclusive particle numbers but chooses the representative by collection
order; that choice still needs improvement.

Cluster features now include separate Cherenkov and scintillation energies in
addition to combined energy, position-based eta/phi, transverse energy,
position, and hit count. Shower widths remain zero because cluster cells are
not loaded by the no-hit IDEA postprocessing path. The separate channels make
a dual-readout calibration or ratio available for subsequent model work.

### 100-event runtime/memory profile and removal of unused GGTF tracking

- The completed `idea-100` simulation took 3:15:15 wall time for 100 events,
  averaging 113.45 seconds per event. Its sampled maximum RSS was 8,644,772
  KiB (8.24 GiB).
- The corresponding GGTF-enabled reconstruction took 3:05:13 wall time,
  averaging about 111 seconds per event. Its sampled maximum RSS was
  32,388,496 KiB (30.89 GiB), reached around event 78; final RSS was still
  about 30.53 GiB.
- `GGTF_tracking` dominated that reconstruction at 85.2 CPU seconds/event on
  average and 302 seconds for the slowest event. `topoClusterAll` averaged
  about 10 seconds/event and the main SiPM emulation about 10.3 seconds/event.
- Memory sampling attributes the large reconstruction footprint to GGTF's
  ONNX inference rather than geometry or normal event data. RSS was about
  2.6 GiB before event processing, jumped to roughly 23.8 GiB during the first
  GGTF call, and then grew in occupancy-dependent steps. The model has dynamic
  hit-count inputs and dense attention operations; ONNX Runtime's arena
  retention is consistent with the high RSS plateaus between events.
- The IDEA MLPF path does not consume `PrefitTracks`, `FittedTracks`, or
  `FittedTracksWithFilteredHits`. It reads the truth-seeded
  `TracksFromGenParticles` and `TracksFromGenParticlesAssociation` collections.
  GGTF and its fitter are therefore no longer scheduled in normal production.
  Set `ENABLE_GGTF=1` to restore the previous tracking sequence for dedicated
  detector-reconstruction studies. DCH digitization remains enabled and uses
  the same release-compatible legacy/modern selection as before.
- Both the default and `ENABLE_GGTF=1` steering configurations were checked
  under Key4HEP `2026-04-08`. A real one-event reconstruction of
  `idea-100/work/IDEA_sim.root` completed successfully without GGTF. It used
  40.2 seconds of event CPU time and had a sampled maximum RSS of 3,342,916
  KiB (3.19 GiB); total wall time including IDEA geometry initialization was
  157 seconds. The 38 MB output contains one event with
  `TracksFromGenParticles`, its association, `TopoClusterAll`, and DCH digis,
  while `PrefitTracks` and fitted-track collections are absent as intended.
- This single-event result reduces peak memory by about a factor of ten and
  removes the dominant runtime module. A full 100-event no-GGTF run is still
  needed to measure the new typical runtime and verify the maximum across the
  complete occupancy distribution.

### Local reproduction of the Slurm invalid-free failure

- The local reproducer uses the directly failing fully hadronic WW simulation
  seed `400009`. Generation completed successfully with the same Key4HEP
  release and local plugins as the Slurm job.
- The resulting 100-event HepMC stream was simulated as ten independent
  10-event ranges using `--skipNEvents`, event seeding, and seed `400009`.
  Every range exited successfully, including PODIO ROOT-writer finalization.
  No individual event is therefore sufficient to reproduce the invalid free.
- A single-process 100-event run is still active under
  `/tmp/idea-memory-repro-ww400009`. At the latest check it had reached event
  12 with about 8.3 GiB RSS. The remaining candidates are cross-event
  accumulation, full-file scale, or nondeterministic heap corruption.

## HEP-KBFI staging PRs and upstream publication plan

The Key4HEP changes have been rebased onto each HEP-KBFI fork's current `main`
branch and split into focused staging PRs. These PRs are for independent
validation inside HEP-KBFI before opening PRs against the official repositories.

- `HEP-KBFI/k4geo`:
  - [#1 Preserve IDEA Cherenkov photon truth ancestry](https://github.com/HEP-KBFI/k4geo/pull/1)
  - [#2 Avoid duplicate IDEA DRC map lookups](https://github.com/HEP-KBFI/k4geo/pull/2)
- `HEP-KBFI/k4RecCalorimeter`:
  - [#1 Index calorimeter truth links by detector cell ID](https://github.com/HEP-KBFI/k4RecCalorimeter/pull/1)
  - [#2 Write optical SiPM digi-hit truth links](https://github.com/HEP-KBFI/k4RecCalorimeter/pull/2)
  - [#3 Preserve dual-readout energies in topological clusters](https://github.com/HEP-KBFI/k4RecCalorimeter/pull/3)
- `HEP-KBFI/FCC-config`:
  - [#1 Enable IDEA dual-readout fast simulation](https://github.com/HEP-KBFI/FCC-config/pull/1)
  - [#2 Configure IDEA reconstruction for MLPF production](https://github.com/HEP-KBFI/FCC-config/pull/2)
- `HEP-KBFI/key4hep-sim`:
  - [#2 Add IDEA simulation and reconstruction workflow](https://github.com/HEP-KBFI/key4hep-sim/pull/2)

All eight staging PRs target `main`. At the initial status check they were
clean and mergeable, with no fork CI checks. Mergeability and CI state must be
checked again before merging. The remote refs were rechecked on 2026-08-28:

- `HEP-KBFI/k4geo`: `main` at `3bb16be`, integration at `73238dd`, PR #1 at
  `1e8c0ec`, and PR #2 at `165c4fb`;
- `HEP-KBFI/k4RecCalorimeter`: `main` at `1df2985`, integration at `155e699`,
  PR #1 at `c73f9e5`, PR #2 at `2ba6569`, and PR #3 at `528c600`; and
- `HEP-KBFI/key4hep-sim`: `main` at `5be058e`, IDEA workflow/PR #2 at
  `a3a3fd2`.

The April-stack preservation branches were created and pushed on 2026-08-28:

- `HEP-KBFI/k4geo:key4hep-2026-04-08-mlpf` at `b313130`;
- `HEP-KBFI/k4RecCalorimeter:key4hep-2026-04-08-mlpf` at `25016a3`;
- `HEP-KBFI/FCC-config:key4hep-2026-04-08-mlpf` at `cdc8b78`; and
- `HEP-KBFI/key4hep-sim:key4hep-2026-04-08-mlpf` at `c09ebd3`, pinning the
  three commits above.

The `idea-mlpf-integration` branches combine the focused commits.
`key4hep-sim` #2 pins those exact integration commits as submodules; those
pins are temporary and must move to stable official or fork `main` commits
after the focused changes merge.

Initial branch validation completed during PR preparation:

- both changed k4geo translation units compile against Key4HEP `2026-04-08`;
- the k4RecCalorimeter truth-link and optical-link units compile against that
  stack;
- an exact current-main export of the five IDEA reconstruction components now
  compiles and installs against the April stack with the validation harness's
  recorded metadata-transport compatibility patch; unrelated components that
  require the newer metadata API are deliberately outside that campaign-stack
  build;
- FCC-config passes Python compilation, and both default no-GGTF and
  `ENABLE_GGTF=1` configurations load successfully; and
- the key4hep-sim runner and build scripts pass `bash -n`.

The staging PRs must now be validated independently, without relying on the
combined integration branches. Each PR should have a minimal test that isolates
its contract and a representative IDEA event test where appropriate. Only
after those results and fork CI are clean should equivalent PRs be opened from
the HEP-KBFI branches against the official repositories, in this order:

1. `key4hep/k4geo` for the two k4geo changes.
2. `HEP-FCC/k4RecCalorimeter` for the three reconstruction-module changes.
3. `HEP-FCC/FCC-config` after the required component PRs are accepted or have
   stable review branches.
4. Update `HEP-KBFI/key4hep-sim` submodule pins from the temporary integration
   commits to the resulting official/fork `main` commits.

The parent `particleflow` repository separately contains IDEA postprocessing,
validation, production-specification, and test changes. Those require their
own branch and validation. Generated reports, ROOT/Parquet outputs, and local
build/install trees must not be pushed.

### Parent particleflow PR and submodule state

Checked on 2026-08-28:

- [`particleflow` PR #495](https://github.com/jpata/particleflow/pull/495),
  `Add IDEA simulation and MLPF pipeline integration`, still has remote branch
  head `9e2546b`. Local commit `85b68e4` applies the outstanding Black changes,
  adds the current validation/production work, and passes the full pre-commit
  suite plus the focused IDEA tests (`12 passed`). The local branch contains
  several additional unpushed commits and must be pushed before CI can
  validate this state.
- The formerly dirty old-base IDEA worktrees are now preserved as clean
  `key4hep-2026-04-08-mlpf` branches in the four HEP-KBFI repositories. The
  k4geo and k4RecCalorimeter branches build and install against the pinned
  stack, FCC-config passes Python compilation, and the outer workflow scripts
  pass `bash -n`. All four remote heads were verified after pushing.
- The parent `mlpf/data/key4hep/gen` gitlink now advances from `a3a3fd2` to
  `c09ebd3`. The nested checkout is clean and on the corresponding tracking
  branch; its FCC-config, k4geo, and k4RecCalorimeter gitlinks resolve to the
  release-specific branches above.
- The release-specific branches preserve the deployable April campaign graph;
  they do not replace the focused, current-main staging PRs. Continue using
  `73238dd` and `155e699` plus their focused branches for upstream acceptance
  evidence, and use `c09ebd3` for exact April-stack production reproduction.
- `key4hep-sim` #2 remains separately at `a3a3fd2`; at the last GitHub status
  check it was open, clean, and mergeable, with no checks, reviews, or
  comments. Recheck that status before acting on it.
- Finalize the dependency graph in two stages. First validate and merge the
  focused component PRs, then add a focused commit to `key4hep-sim` #2 that
  moves all three nested gitlinks to stable official or fork `main` commits.
  Validate that exact graph from a fresh recursive clone, merge `key4hep-sim`
  #2, and finally update the parent `mlpf/data/key4hep/gen` pointer from
  `c09ebd3` to the resulting stable `key4hep-sim/main` commit in a standalone
  parent commit.
- The parent PR description says that a 100-event IDEA workflow completed.
  The completed sample validates the earlier GGTF-enabled path, while the
  current default no-GGTF reconstruction has a successful real one-event
  test and still needs the full 100-event run listed below. Keep that
  distinction explicit in the PR description unless the outstanding run is
  completed first.

### k4geo and k4RecCalorimeter integration action plan

Use fresh clones or worktrees for every acceptance run. The nested checkouts
under `mlpf/data/key4hep/gen/idea` now reproduce the clean release-specific
branches, but they remain old-base campaign artifacts and are not proof that
the rebased focused PR heads pass.

#### Role of `validate_k4geo_changes.sh`

`scripts/validate_k4geo_changes.sh` is now the main execution harness for both
the k4geo and k4RecCalorimeter parts of this plan. It is still not a substitute
for native component tests or the final clean dependency-graph acceptance run.

It already provides the following useful isolation and evidence:

- clones both component repositories into a disposable directory, resolves
  and records immutable refs, exports source trees without checking them out,
  and builds the k4geo and k4RecCalorimeter baseline, every focused branch,
  and each combined branch;
- verifies the exact full-tree file scope of all focused branches, verifies
  exact composition of the three k4RecCalorimeter changes, and verifies that
  the upstream keyed-hit fix can be reversed in isolation;
- creates a unique work directory under `/scratch/persistent/joosep`, clones
  key4hep-sim `c09ebd3` there by default, initializes every recursive
  submodule at its recorded commit, and records the complete workflow graph
  and cleanliness state;
- runs identical fixed-seed CLIC, CLD, IDEA-standard, and IDEA-fast gun cases
  across `main`, both focused branches, and the combined branch;
- requires exact typed EDM4hep equality for PR #2 and for detectors unaffected
  by PR #1, while allowing only the intended DRC truth payload to differ for
  PR #1;
- checks positive integral Cherenkov counts, hit/contribution closure, both
  ObjectID fields and collection membership for relations, charged ancestors,
  empty fast-simulation output, and exact composition of PR #1 with PR #2;
- records CPU, wall time, RSS, output size, loaded library paths, and DDSim
  timing, with non-regression gates and a standalone dense-map microbenchmark;
- isolates the already-upstream PR #620 keyed-hit speedup with both a pion gun
  and identical five-event qq input; and
- simulates one fixed-seed qq sample with the combined k4geo branch, then
  reconstructs that identical file with k4RecCalorimeter `main`, each of the
  three focused PRs, and the combined branch, plus reversed-input runs for the
  cluster and combined branches;
- checks exact loaded-library provenance and ERROR/FATAL logs, scintillation
  and optical digi/sim endpoint closure, normalized cell truth, positive-cell
  filtering, cellID-based cluster truth, copied-cell membership, total and
  per-channel cluster energy/error closure, shape-parameter metadata, and
  normal/reversed input-order invariance; and
- records reconstruction CPU, wall time, RSS, and output size, with focused
  branch non-regression and input-order stability gates. The generated JSON
  summaries retain per-event counts and canonical cluster payloads.

The script therefore covers most of k4geo plan step 2, including branch
composition. Its pre-fix/current 10x gate documents upstream PR #620 and must
not be presented as the benefit of HEP-KBFI k4geo PR #1 or #2. The PR #2
evidence is exact output equivalence plus a no-regression gate and
microbenchmark, not a claim of a 10x application-level improvement.

The following gaps remain:

- The default clean workflow commit still contains temporary integration
  gitlinks. Override it with `--workflow-ref` after those pins move to stable
  component commits. `--workflow-root` remains available for an existing
  checkout, but it is recursively audited and rejected if dirty; use
  `--allow-dirty-workflow-root` only for explicitly non-acceptance development
  runs.
- Default branch names are moving refs. An acceptance run must pass the exact
  immutable SHAs listed above and record the resolved refs in its report.
- The builds use `BUILD_TESTING=OFF`; they prove the exercised plugins build
  and execute but do not run either repository's native unit tests.
- The standard-optical case covers electrons only; pions and the empty case
  currently exercise fast optical transport. Add at least one standard-optical
  hadron case before claiming both transport modes are covered for hadronic
  ancestry.
- The qq matrix exercises realistic integration contracts, but it does not
  replace focused empty-input, collection-name override, single-input
  calorimeter, and scaling tests for the three reconstruction PRs.
- The default release is the campaign stack `2026-04-08`. This is valuable for
  compatibility, but current k4RecCalorimeter uses a newer metadata API. The
  script records a narrowly scoped April compatibility patch and builds only
  the five IDEA components it exercises. This does not replace an unmodified
  full build/test on a release supported by current component `main`.
- The focused compatibility export, output validator, steering generator, and
  syntax checks have been smoke-tested locally, but the full ten-build,
  multi-event acceptance matrix still needs to run and its retained artifacts
  need to be attached to the staging PRs.

Use the script in three passes:

1. Run `--quick` against all exact k4geo and k4RecCalorimeter SHAs. The script
   makes a fresh recursive workflow clone under persistent scratch by default;
   pass the exact workflow SHA with `--workflow-ref`. This is a build, payload,
   reconstruction-contract, and wiring check, not performance acceptance.
2. Run the full default event/repetition matrix with `--keep-workdir`, exact
   SHA arguments and workflow SHA. Preserve `metrics.tsv`, logs,
   component/workflow refs, compatibility patches, reconstruction JSON
   summaries, comparison output, and final plots for the PR evidence.
3. Run native and focused edge-case tests separately, then use the clean
   key4hep-sim integration gate in plan step 4 for the exact final gitlink
   graph and FCC-config contract.

1. Preserve and isolate the old worktrees. Completed on 2026-08-28.
   - The four clean `key4hep-2026-04-08-mlpf` branches preserve FCC-config,
     k4geo, k4RecCalorimeter, and the outer key4hep-sim dependency graph.
   - Local build/install trees and generated ROOT/Parquet output remain
     excluded from every commit.
   - Create clean validation clones at the exact HEP-KBFI `main`, focused-PR,
     and integration refs listed above. Keep one shared fixed-seed input set so
     every branch sees identical events.

2. Establish the k4geo baseline separately from the two proposed changes.
   - Build `3bb16be` first and confirm the supported Key4HEP environment. The
     keyed fiber-hit lookup already merged upstream in k4geo PR #620 is part of
     this baseline; its large speedup is not an acceptance claim for either
     HEP-KBFI PR.
   - Validate PR #1 (`1e8c0ec`) in fast and standard optical transport. Require
     valid charged-parent relations, per-hit and event-level Cherenkov photon
     closure, valid MCParticle indices, empty-event handling, and unchanged
     non-DRC payloads.
   - Validate PR #2 (`165c4fb`) against the same input. Require typed EDM4hep
     output equivalence and no timing or memory regression; report any small
     dense-map lookup improvement separately from the upstream PR #620 gain.
   - Build the combined integration ref `73238dd` and require both independent
     contracts to hold together. Run the existing five-event qq validation
     only after the smaller focused tests pass.

3. Validate the three k4RecCalorimeter PRs independently.
   - First build `1df2985` and each PR head with a Key4HEP release supported by
     current k4RecCalorimeter `main`. Treat the known April-2026 metadata-API
     build failure as a release-compatibility issue, not as a failure of the
     new topocluster event logic. Test the campaign's pinned `2026-04-08` stack
     separately on the compatibility integration path.
   - PR #1 (`c73f9e5`): test copied and non-copied cluster cells, repeated
     `cellID` values, multiple input link collections, `minCellEnergy`, empty
     inputs, normalized weights, and approximately linear runtime scaling.
   - PR #2 (`2ba6569`): test one-to-one optical digi/sim associations, valid
     relation endpoints, configurable collection names, zero-hit events, and
     exact photon-count closure when fed the k4geo PR #1 output.
   - PR #3 (`528c600`): test reversed input order, duplicate physical cells,
     Cherenkov/scintillation and total-energy closure, error propagation,
     shape-parameter metadata, copied-cell output, and ordinary single-input
     calorimeters.
   - Build `155e699` only after all three focused contracts pass. Repeat the
     combined tests to catch cross-PR assumptions, especially the flow from
     optical links through cell-ID truth links into dual-readout clusters.

4. Run a clean end-to-end integration gate.
   - Pin clean k4geo `73238dd` and k4RecCalorimeter `155e699` checkouts with the
     matching FCC-config integration commit; do not overlay files from the
     release-specific old-base branches.
   - Run one fixed-seed electron gun and at least one representative qq event
     with GGTF disabled. Require successful finalization, no ERROR/FATAL log
     entries, the MLPF-required collections, valid relation endpoints, photon
     and channel-energy closure, and reconstruction timings consistent with
     the corrected implementation.
   - Repeat from a fresh recursive `key4hep-sim` clone at `c09ebd3`. This is
     the acceptance test for the exact dependency graph, not just for local
     source directories.

5. Publish in dependency order and replace temporary pins.
   - Attach the focused test commands, release/container identity, compact
     results, and artifact checksums to each HEP-KBFI PR. Recheck mergeability
     and CI immediately before merging.
   - Open or merge the two official k4geo PRs and the three official
     k4RecCalorimeter PRs once their individual evidence is complete. The two
     repositories can progress in parallel, but FCC-config must wait for
     stable component refs.
   - Rebase the integration branches onto the resulting stable official or
     fork `main` commits, rerun the small end-to-end gate, and update
     `key4hep-sim` #2's three nested gitlinks in one focused commit.
   - Validate a fresh recursive clone of that final graph, merge
     `key4hep-sim` #2, then update the particleflow `gen` gitlink in a separate
     parent commit.

The integration is ready to advance only when every focused PR has independent
evidence, the combined graph passes from a fresh clone, no temporary
integration gitlinks remain, and the exact tested commit IDs are recorded in
the PRs and this file.

## Immediate next steps

The highest priority is independent validation of the HEP-KBFI staging PRs.
Do not open official Key4HEP PRs from the combined integration branches before
the focused changes pass their own tests.

`scripts/validate_k4geo_changes.sh` now covers the focused k4geo and
k4RecCalorimeter branch matrices in one reproducible run. It builds the
isolated pre-PR-#620 k4geo control, generates one fixed-seed 365 GeV qq stream,
simulates it once with the combined geometry branch, and reconstructs that
identical file with all five reconstruction variants plus both reversed-input
controls. The companion PODIO validator has passed one event from an existing
real IDEA reconstruction, and the focused April compatibility export compiles
and installs. The full default matrix still needs execution to obtain the
final measurements, JSON summaries, logs, and plots. It becomes
dependency-graph evidence only when supplied a fresh, clean key4hep-sim
workflow root at the exact recorded gitlinks.

1. Run the expanded harness first in `--quick` mode and then with the full
   default matrix, using exact component and workflow SHA arguments. Let the
   script create its clean recursive scratch checkout. Retain its
   component/workflow ref tables, compatibility patches,
   metrics, logs, JSON summaries, comparison output, and plots.
2. Supplement the matrix with focused k4RecCalorimeter tests:
   - #1 with copied and non-copied cells, repeated `cellID` inputs,
     `minCellEnergy`, weight closure, and runtime scaling;
   - #2 with one-to-one digi/sim link checks, empty inputs, and collection-name
     overrides; and
   - #3 with reversed input order, duplicate physical cells, energy/error
     closure, named shape-parameter metadata, and ordinary single-input
     calorimeters.
3. Run the native component tests on releases supported by the current
   repositories, and add a standard-optical hadron case for k4geo #1. Keep the
   campaign-stack compatibility result separate from this unmodified build.
4. Validate FCC-config #1 with a simulation comparison showing that the fast
   model is active and preserves the expected detector output. Validate
   FCC-config #2 only after installing the independently tested component
   branches; exercise both no-GGTF and opt-in modes and compare required output
   collections.
5. Run a full 100-event reconstruction of `idea-100/work/IDEA_sim.root` with
   the default no-GGTF steering. Record per-event Chrono timings and sampled
   peak RSS, compare the MLPF-required collections with the GGTF-enabled file,
   and replace the old 13--15 hour production estimate.
6. Record the independent results in each staging PR and ensure fork CI is
   green. Then open upstream PRs against `key4hep/k4geo`,
   `HEP-FCC/k4RecCalorimeter`, and `HEP-FCC/FCC-config` in dependency order.
   After merges, update the integration branches and key4hep-sim submodule pins.
7. Repair the simulation/reconstruction heartbeat so it works inside the
   Key4HEP container without the failing `ps`/`libsystemd.so.0` dependency.
   Confirm that a test job emits one real `[progress]` record per configured
   interval to both the Slurm log and persistent stage log.
8. Resubmit a small IDEA generation test with the new 10 GB request. Verify
   `ReqMem=10000M`, follow peak RSS and event-save cadence, and determine
   whether the prior event-2 stall clears. Do not infer that 10 GB is adequate
   solely because the job remains in Slurm's `RUNNING` state.
9. Define and validate the physics calibration for the two IDEA readout
   components. Study Cherenkov, scintillation, their ratio, and an appropriate
   combined estimator on gun samples before treating combined cluster response
   as particle energy.
10. Repeat detailed truth checks on representative corrected events, including
   photons, charged hadrons, and neutral hadrons. Verify that recovered links
   follow detector cell IDs and deposited-energy contributors, not merely that
   link counts and particle coverage increased.
11. Change IDEA cluster target construction so the highest-weight or
   highest-associated-energy fragment is the representative, while every
   associated fragment retains the same `particle_number`.
12. Add an IDEA-aware validation metric that compares particle truth energy
   with a calibrated sum of all associated topological-cluster fragments. The
   new R2 gate checks representation integrity, not physics response.
13. Add explicit feature names for dual-readout channels if changing the common
   tensor schema becomes acceptable. The current `energy_ecal`/`energy_hcal`
   mapping is backward-compatible but requires IDEA-specific interpretation.
14. Run a true one-event electron-gun job with the latest patched local module
   as a minimal repeatable runtime and association check,
   making sure `NEV=1` is exported in the effective environment:

   ```bash
   PROGRESS_INTERVAL=15 NEV=1 \
   OUTDIR=/tmp/idea-gun/output WORKDIR=/tmp/idea-gun/work \
   bash mlpf/data/key4hep/gen/idea/run_sim.sh gun_e_10gev 424242
   ```

   Use it to measure the separate-channel response and validate the new
   shape-parameter contract on a simple known-energy shower.
15. Once the dual-readout calibration and aggregate-cluster metric are defined,
   decide whether the completed 10-event sample is sufficient for an initial
   production benchmark or whether a larger sample is needed before enabling
   Snakemake production.

## MLPF pipeline-validation path

Implemented on 2026-08-24 using the 100-event qq parquet in `idea-100`:

- Registered `idea` as a first-class track/cluster MLPF dataset with the same
  17-wide common EDM4hep input feature schema and six output classes as CLD.
- Added the `idea_edm_qq_pf` TFDS builder. It supports a deterministic 80/20
  event-level train/test split even when only one parquet file is available.
- Recorded `track_source=truth_seeded`, `candidate_source=target_oracle`, and
  `suitable_for_physics=false` in the TFDS metadata. The oracle `ycand` exists
  solely to validate evaluator plumbing; it is not a reconstruction baseline.
- Added `prepare_idea_pipeline.py`, which writes the same provenance into a
  prepared parquet, adds oracle candidate fields, and repairs invalid
  representative `jet_idx` values to `-1`.
- Fixed future IDEA postprocessing so representatives not assigned to a target
  jet start at `jet_idx=-1` rather than implicitly belonging to jet zero.
- Made hit-only validation gates skip cleanly for the configured no-hit IDEA
  dataset. G3 is informational for IDEA because `E(selected gen jets)+MET` is
  not total visible energy in an electron-positron event.
- Added IDEA labels to the inference plotting layer and allowed dataloaders to
  run with `num_workers=0` without configuring multiprocessing-only options.
- Added the lightweight `pyg-idea-pipeline-v1` recipe and enabled the IDEA
  postprocessing/TFDS production stages for qq.
- Extended the production mapping to postprocess and build separate TFDS
  datasets for ttbar and fully hadronic WW. The electron gun remains a
  generation-only validation sample.

The prepared parquet passes strict validation with 19 PASS, 0 FAIL, 5 WARN,
and 4 intentional SKIP gates. TFDS contains 80 train and 20 test events. A
complete one-step CPU smoke run successfully produced a checkpoint, validation
losses, an inference parquet, FastJet metrics, and plots under
`idea-100/experiments/smoke-complete`. A 20-step single-event learning check
reduced training loss from 54.49 at step 5 to 50.94 at step 20 and validation
loss from 44.83 to 43.30. These checks validate the software path, not MLPF
physics performance or genuine IDEA tracking.

The initial IDEA production ranges are limited to exactly 100 generation jobs
per configured sample (`CHUNK_SIZE=1`, exclusive seed-range ends): 100 events
for the one-event electron gun and 10,000 events each for ttbar, fully hadronic
WW, and qq.

## Useful files

- `particleflow_spec.yaml`
- `configs/tallinn/pixi.toml`
- `mlpf/snakemake/produce_snakemake.py`
- `mlpf/data/key4hep/gen/idea/run_sim.sh`
- `mlpf/data/key4hep/gen/idea/k4geo-local/plugins/FiberDRCaloSDAction.cpp`
- `mlpf/data/key4hep/gen/idea/k4geo-local/plugins/Geant4Output2EDM4hep_DRC.cpp`
- `mlpf/data/key4hep/gen/idea/build_k4reccalorimeter.sh`
- `mlpf/data/key4hep/gen/idea/k4RecCalorimeter-local/RecCalorimeter/src/components/CreateTruthLinks.h`
- `mlpf/data/key4hep/gen/idea/k4RecCalorimeter-local/RecCalorimeter/src/components/CreateTruthLinks.cpp`
- `mlpf/data/key4hep/gen/idea/k4RecCalorimeter-local/RecFCCeeCalorimeter/src/components/CaloTopoClusterFCCee.h`
- `mlpf/data/key4hep/gen/idea/k4RecCalorimeter-local/RecFCCeeCalorimeter/src/components/CaloTopoClusterFCCee.cpp`
- `mlpf/data/key4hep/gen/idea/README.md`
- `mlpf/data/key4hep/gen/idea/docs/IDEA_particle_flow_status.md`
- `mlpf/data/key4hep/gen/idea/FCC-config/FCCee/FullSim/IDEA/IDEA_o1_v03/SteeringFile_IDEA_o1_v03.py`
- `mlpf/data/key4hep/gen/idea/FCC-config/FCCee/FullSim/IDEA/IDEA_o1_v03/run_digi_reco.py`
- `mlpf/data/key4hep/postprocessing.py`
- `tests/test_postprocessing_idea.py`
- `tests/validate_parquet.py`
