# Local CMSSW debugging

## Current physics validation: fiducial GEN versus new targets

For 10,000 events per species (electron, charged pion, muon, photon), run:

```bash
PYTHONPATH=. .venv/bin/python scripts/cmssw/run_single_gun_suite.py \
  --events 10000 --threads 8 --streams 8 --chunk-events 500
# Resume using the suite directory printed by the command:
PYTHONPATH=. .venv/bin/python scripts/cmssw/run_single_gun_suite.py --resume /absolute/path/to/suite
```

This uses independent single-species particle/antiparticle guns, flat pT
0.7–10 GeV and |eta| < 3, with PU0. Chunks run sequentially to bound disk and
memory usage; every CMSSW stage uses the requested thread/stream settings.
Each chunk retains NanoAOD, sparse overlap weights, assigned targets, configs,
seeds and logs. Intermediate step1/step2/step3, MiniAOD and DQM ROOT files are
removed only after successful extraction, assignment, truth-coverage and event-ID
checks. Event numbers are unique across chunks of each species. The suite writes
`suite_status.json`; each chunk writes stage status and `postprocessing_complete.json`.
After each species, its targets are combined and the GEN-only validation runs.
Outputs are under `reports/cmssw_truth_postprocessing_guns/<suite>/`.

The default validator now compares only visible GEN status-1 particles with the
new truth-graph targets. It does not run the old normalizer as a physics baseline.
Both collections independently require particle pT > 0.5 GeV and |eta| < 3;
neutrinos are excluded and charged leptons included. The inclusive PU0 event,
including underlying event, is the reference.

Jets use anti-kT R=0.4, |eta| < 2.6, GEN pT >= 20 GeV, and target matching
candidates pT >= 5 GeV. Unique angular matching requires dR < 0.2 without a
response cut. MET is the negative vector sum of fiducial visible momenta.
It measures fiducial imbalance rather than full-acceptance neutrino MET.
Magnitude and projected MET response require GEN MET >= 20 GeV.

The plots show median target/GEN response and relative resolution
sigma68/|median|, where sigma68=(Q84-Q16)/2. Error bars are 68% bootstrap
intervals, resampling whole events for jet uncertainties; bins require 20 entries.
Parallel/perpendicular MET residual widths are also provided in GeV.

To plot existing assigned targets:

```bash
PYTHONPATH=. .venv/bin/python scripts/cmssw/plot_gen_target_response.py \
  /absolute/path/to/events.pkl --output /absolute/path/to/gen_target_response
```

The 1,000-event ttbar outputs are under
`reports/cmssw_truth_postprocessing_ttbar/20261114T225915Z-ttbar-pu0-1000ev-8threads/ttbar_pu0/gen_target_response/`.
This directory contains response curves, distribution overlays, matching
efficiency, complete definitions and bin statistics in `response.json`.
Earlier normalizer compatibility plots for this sample were moved to
`historical_assignment_compatibility/` and are excluded from current summaries.
Compatibility comparisons described later in this document are historical
implementation checks, not the current physics validation.

From any working directory, use this repository's `env.sh` to execute commands in
EL9 with the local CMSSW runtime initialized. The defaults match the original job:
`CMSSW_20_1_0_pre5`, `el9_amd64_gcc14`, and workflow
`34506.213_SinglePiFlatPt0p7To10+Run4D121_HGCALNanoVal`.

```bash
./env.sh python3 --version
./env.sh bash -c 'echo "$CMSSW_BASE"; command -v cmsRun'
./env.sh bash -i
```

With no arguments, `./env.sh` runs the original step2 job in its original directory.
For debugging, prefer the isolated runner:

```bash
scripts/cmssw/debug.sh                  # unchanged step2 configuration
scripts/cmssw/debug.sh --events 0       # setup only
scripts/cmssw/debug.sh --events 1       # short reproduction
scripts/cmssw/debug.sh -- python3 -c 'import FWCore.ParameterSet.Config'
scripts/cmssw/debug.sh -- bash -c 'exit 23'  # runner returns 23, also writes exit_code
```

Each attempt gets a unique directory under `logs/cmssw/`, containing copied Python
configs, a symlink to the original `step1.root`, `command.txt`, `run.log`,
`exit_code`, and any generated outputs. The runner does not link `step2.root`,
so the original output is protected. Commands run in that new directory; use
absolute paths for scripts elsewhere. `--events` modifies only a generated wrapper
for the default step2 command. Interactive shells work better through `env.sh`.

Override `CMSSW_RELEASE_DIR`, `CMSSW_JOB_DIR`, or `SCRAM_ARCH` to test another
release/workflow. Use absolute paths. `CMSSW_RUN_DIR` changes the working directory
for direct `env.sh` commands. Arbitrary commands can modify the shared release;
only job configs and outputs are isolated. Existing ROOT inputs are linked, not
copied. Apptainer requires execution outside the Codex sandbox on this host.

## Failure reproduced on 2026-11-14

The original configuration exits with code 90 before event processing:

```
ModuleNotFoundError: No module named 'HeterogeneousCore.CUDAServices.CUDAService_cfi'
```

Evidence is in `logs/cmssw/20261114T161819Z.Lhd0FI/run.log`. The same error occurs
with the one-event wrapper (`20261114T161655Z.sqmPdG`). The local CUDA plugin
library exists, but its per-library EDM cache is empty, the generated
`cfipython/.../HeterogeneousCore/CUDAServices/CUDAService_cfi.py` is missing, and
plugin lookup selects a nonexistent `poisoned/plugin-poisoned-...so` path.

`ldd -r` on `pluginHeterogeneousCoreCUDAServicesPlugins.so` reports missing
`libnvidia-ml.so.1` and unresolved NVML symbols in the original container environment
(`20261114T161749Z.vNdqrY/run.log`). NVIDIA passthrough resolves this dependency:

```bash
APPTAINER_NV=true scripts/cmssw/debug.sh -- bash -ec '
  ldd -r "$CMSSW_BASE/lib/$SCRAM_ARCH/pluginHeterogeneousCoreCUDAServicesPlugins.so"
  nvidia-smi -L
  edmWriteConfigs --path "$CMSSW_BASE/lib/$SCRAM_ARCH/pluginHeterogeneousCoreCUDAServicesPlugins.so"
  ls -l CUDAService*
'
```

That experiment succeeded and generated CUDAService, CUDAMonitoringService, and
NVProfilerService configurations in the isolated directory
(`20261114T161804Z.FaxCBz`). This establishes a dependency problem in the original
container and is consistent with failed plugin discovery during the local build.
It does not establish that the rest of the CMSSW job works.

## Build repaired on 2026-11-14

Ran the following inside EL9 with NVIDIA passthrough (through `env.sh`):

```bash
APPTAINER_NV=true ./env.sh bash -ec '
  cd "$CMSSW_BASE"
  mkdir -p ../logs/cmssw/repair-cuda
  cp -a lib/$SCRAM_ARCH/edm/HeterogeneousCoreCUDAServicesPlugins.cache ../logs/cmssw/repair-cuda/before.cache
  rm lib/$SCRAM_ARCH/edm/HeterogeneousCoreCUDAServicesPlugins.cache
  cd src/HeterogeneousCore/CUDAServices
  scram b -j 4
  python3 -c "from HeterogeneousCore.CUDAServices.CUDAService_cfi import CUDAService; print(CUDAService)"
'
```

The empty cache had to be removed: a normal incremental build considered it up
to date. SCRAM then generated CUDAService, CUDAMonitoringService, and
NVProfilerService configs, registered the plugin, and refreshed the project cache.
The import check passed.

A one-event run with `APPTAINER_NV=true` initialized CUDA and Alpaka on the
RTX 5060 Ti, then exited 65 because `/cvmfs/cms.cern.ch/SITECONF/local` is an
unresolved site symlink on this host. See
`logs/cmssw/20261114T162226Z.9JOjcc/run.log`.

The available Estonia site config can be selected explicitly:

```bash
SITECONFIG_PATH=/cvmfs/cms.cern.ch/SITECONF/T2_EE_Estonia \
  APPTAINER_NV=true scripts/cmssw/debug.sh --events 1
```

This validation is recorded in `logs/cmssw/20261114T162305Z.sXA3Th/`.
It wrote a readable `step2.root` containing one event (verified with
`edmFileUtil -e`), but returned 66 with a configuration-file-open error at the end.
The debug runner now passes an absolute config path to `cmsRun`, and its wrapper
also resolves the underlying configuration with an absolute path.

The rerun with absolute config paths completed successfully (exit 0), processed
event 1, and wrote `step2.root` in `logs/cmssw/20261114T162733Z.nUsPe0/`.
The log still reports an HLT report process-name mismatch (`HLTX`) and a missing
`MuonME0Hits` truth collection; neither terminated the one-event job.
Only one event was validated; the original ten-event job has not been rerun.

## Remaining workflow steps

Steps 3 (RECO/PAT/NANO/validation/DQM), 4 (harvesting), and 5 (ALCA) subsequently
completed with exit code 0 using the one-event step2 output. Logs, framework job
reports, outputs, and `remaining_steps_summary.json` are in
`logs/cmssw/20261114T162733Z.nUsPe0/`.

Step3 initially failed because the installed `data-RecoTauTag-TrainingFiles`
V00-11-00 package only provided TensorFlow models, while the local source requested
ONNX models. Missing ONNX files from `BoostedDeepTauId` and `DeepTauId` were
installed under the local release's
`external/el9_amd64_gcc14/data/RecoTauTag/TrainingFiles/data/` from the official
`cms-data/RecoTauTag-TrainingFiles` repository. The exact commit, URLs, verified Git
blob hashes, and SHA-256 checksums are recorded in `tau_model_downloads.json` in
the run directory. The original failure is preserved in `step3_attempt1/`.

To run steps 3–5 again in a directory containing `step2.root` and the workflow
configs (this overwrites that directory's step3–5 outputs and logs):

```bash
CMSSW_RUN_DIR=/absolute/path/to/run \
SITECONFIG_PATH=/cvmfs/cms.cern.ch/SITECONF/T2_EE_Estonia \
APPTAINER_NV=true ./env.sh bash "$PWD/scripts/cmssw/run_remaining.sh"
```

The helper stops if step3 fails. It runs step5 even if step4 fails, since step5
reads `step3.root` and does not depend on harvesting. Each step records a command,
log, framework report, resource report (when produced), and exit code.

Step3 reports one event in each of RECO, MiniAOD, and NanoAOD. DQM messages include
missing monitoring elements and failed fits; ALCA warns that its MinBias HLT
selection matches no paths. All steps nevertheless completed. ALCA streams with
no selected events are not evidence of useful calibration statistics in this
single-pion, one-event sample.

## Full ten-event workflow

Run all five steps, starting with generation and simulation of ten new events:

```bash
scripts/cmssw/run_workflow.sh
```

The launcher creates a fresh `logs/cmssw/<timestamp>-10events.<id>/` directory,
copies the original configs, records their hashes and runtime settings, and runs
steps 1–5 sequentially. It defaults to NVIDIA passthrough and the Estonia site
configuration used in the successful debugging runs. Existing run directories are
preserved. Steps 1, 2, 3, and 5 retain their configured ten-event limit; harvesting
reads the DQM output and retains its original configuration. Each step writes a
log, command, framework report, resource report (when produced), and exit code.
`workflow.log` captures the full chain, and `workflow_exit_code` records its status.

`run_remaining.sh` also accepts a first-step number (default 3) for resuming from
an existing run directory; for example, pass `4` to run harvesting and ALCA.

Full ten-event run completed successfully on 2026-11-14 in
`logs/cmssw/20261114T184048Z-10events.V08TyD/`: all five steps and the workflow
returned 0. Framework reports confirm ten generated events, ten events read by
steps 2, 3, and 5, and ten events written to DIGI/RAW, RECO, MiniAOD, and NanoAOD.
Harvesting produced a 128,844,655-byte DQM ROOT file. ALCA selected eight events
for `TkAlMinBias.root`; other ALCA streams selected zero. The run retains the
previously observed HLT/DQM monitoring messages. Exact counts, paths, sizes, and
exit codes are recorded in `workflow_summary.json` in the run directory.

## Inspect NanoAOD contents

```bash
.venv/bin/python scripts/cmssw/inspect_nanoaod.py /absolute/path/to/step3_inNANOAODSIM.root
```

Requires uproot, awkward, and NumPy. Writes `nano_inspection/contents.json` and
`branches.tsv` alongside the input file. Reads all branches, summarizes collection
counts, checks array sizes, searches for non-finite floats, and checks selected
track/cluster indices and the six reciprocal truth-association pairs.

The ten-event output contains 695 branches and 91 collection counters. Its full
inspection is saved in `logs/cmssw/20261114T184048Z-10events.V08TyD/nano_inspection/`,
including a readable `contents.txt`. All branches are readable, branch array
lengths are consistent, no NaN/Inf values were found, and truth links are reciprocal.
Four reco candidate track indices and nine sim candidate GSF indices contain
INT32_MIN, corresponding to missing associations in the explicit index tables.
`TruthPV_found` is zero for all events; reconstructed PVs are fake in eight events.

## Association metric comparison

The ten-event metric review is saved in
`logs/cmssw/20261114T184048Z-10events.V08TyD/association_metrics/review.txt`, with
paired distributions, CSV data, and source hashes. The original legacy PFAnalysis
analyzer crashes in its feature extractor on this Phase-2 sample; the legacy
track-association producers were run separately. Their successful job configuration
and output maps are retained in the same directory.

To repeat extraction and plotting after producing the legacy association maps:

```bash
APPTAINER_NV=true ./env.sh python3 "$PWD/scripts/cmssw/extract_association_metrics.py" \
  /absolute/path/to/legacy_associations.root --output /absolute/path/to/pairs.json
.venv/bin/python scripts/cmssw/plot_association_metrics.py /absolute/path/to/audit_directory
```

The plotter expects `pairs.json` and `nano_links.json` in the audit directory.
`nano_links.json` records the corresponding NanoAOD forward/reverse links for the
numerical cross-check. The extractor reads the graph's stamped PFReconstructable
roots; comparison requires those to be the NanoAOD producer's selected root set.
