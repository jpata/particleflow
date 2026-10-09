# Code and Functionality Map: ParticleFlow

This document provides a high-level overview of the repository's structure and the functionality of its core components.
To run the python code in the right environment, use `uv run python3 ...`.

## 1. Configuration & Specification
The project uses a hierarchical configuration system.

- **`particleflow_spec.yaml`**: The single source of truth for the entire project. It defines machine-specific paths (sites), data production scenarios, and model architectures.
- **`mlpf/conf.py`**: Defines the Pydantic models for the configuration, ensuring type safety, path resolution, and model-type schemas including `attention`, `gnnlsh`, `litept`, `hept`, and `heptv2`.
- **`mlpf/pipeline.py`**: Implements hierarchical configuration resolution: base defaults in `mlpf/conf.py` are overridden by scenario-specific values in `particleflow_spec.yaml`, which can further be overridden via command-line arguments (e.g., `--model.num_convs 6`).
- **`configs/`**: Site-specific Pixi environment configurations (`local/`, `lxplus/`, `tallinn/`) and reusable training scenarios/platform profiles under `configs/training/`. The root `pixi.toml` is a symlink into this directory.
- **`pixi.toml` / `pixi.lock` / `uv.lock` / `uv.singularity`**: Project environment, container definitions, and task management. Defines common tasks like `gen`, `post`, `train`, and `validation`.
- **`envs/`**: Isolated virtual environment specifications (e.g., `ort-cpu`, `ort-gpu`) for specific runtimes like ONNX.
- **`validation_cms.yaml` / `validation_key4hep.yaml`**: Specification files for validation scenarios.

## 2. Workflow Orchestration
Complex data production and training pipelines are managed using Snakemake or site-specific shell scripts.

- **`mlpf/snakemake/`**: Contains scripts to generate Snakemake workflows.
  - **`produce_snakemake.py`**: Generates workflows for generation, postprocessing, TFDS creation, and training.
  - **`produce_cms_validation_snakemake.py`**: Orchestrates validation workflows specifically for CMS.
  - **`produce_validation_snakemake.py`**: Orchestrates validation workflows for Key4Hep detectors (CLD, CLIC).
- **`mlpf/pipeline.py`**: The main CLI for training, testing, and hyperparameter optimization. Supports standard and Ray-based execution.
- **`scripts/training/run_scenario.py`**: Resolves a generic scientific training scenario against a hardware profile, validates comparison invariants and global batch size, and launches reproducible seeded jobs.

## 3. Data Production & Preprocessing
- **`mlpf/data/`**: Simulator-specific code for generating and preprocessing data.
  - **`cms/`**: Scripts for CMSSW-based generation (`genjob_pu.sh`), postprocessing (`postprocessing2.py`), validation (`valjob.sh`, `valjob_data.sh`), and plotting (`plot_cms.py`).
  - **`key4hep/`**: Scripts for Key4Hep-based generation (`gen/`) and postprocessing (`postprocessing.py`).
- **`mlpf/heptfds/`**: TFDS (TensorFlow Datasets) builders for various datasets (CMS, CLD, CLIC), including support for both cluster-based and raw hits-based data (`cld_pf_edm4hep_hits`, `clic_pf_edm4hep_hits`). Shared EDM4Hep utilities in `edm4hep_utils/`. ColliderML support: `colliderml_pf/ttbar.py` (clustered view), `colliderml_hits/{ttbar,ttbar_pu200}.py` (hits view)
  + shared conversion in `colliderml_utils/utils.py` reading the MLPF-format parquet written by `mlpf/data/colliderml/postprocessing.py`.

## 4. Machine Learning Core (`mlpf/model/`)
- **`mlpf.py`**: Implementation of the MLPF model, featuring multi-head attention and configurable sub-networks. Supports fused attention, simplified math attention for ONNX export, and HEPT/HEPTv2/LitePT layers that consume detector coordinates.
- **`gnnlsh.py`**: GNN layers with Locality-Sensitive Hashing (LSH) for scalable graph processing, including multi-hash binning, attention kernels, RMSNorm/SwiGLU blocks, and inter-bin attention.
- **`hept.py`**: Implementation of the Hashing-based Efficient Particle Transformer (HEPT).
- **`heptv2.py`**: HEPTv2 point encoder layer with hash-bucketed attention, learned positional encoding over eta/phi coordinates, Qwen-style RMSNorm/MLP blocks, and environment switches for attention/hash implementation variants.
- **`litept.py`**: Integration for the LitePT (Lightweight Point Transformer) architecture.
- **`PFDataset.py`**: Advanced data loading logic, including dataset interleaving and multi-file handling.
- **`training.py`**: Core training loop implementation, supporting DDP and Ray-based distributed training, TensorBoard loss logging, and per-step data-load/model-forward timing.
- **`losses.py`**: Specialized loss functions for particle classification and energy regression.
- **`inference.py`**: Utilities for running model inference and generating predictions.
- **`plots.py`**: Confusion matrix logging and model performance plotting during training.
- **`utils.py`**: Model-level utility functions including target unpacking, object-condensation clustering helpers, learning rate schedules, and metric computation.
- **`distributed_ray.py`**: Integration with Ray for distributed training and HPO.

## 5. Validation, Plotting & Monitoring

- **Local ttbar detector comparison**: see [Section 7](#7-local-ttbar-detector-comparison-report) for inputs, rerun commands, statistical definitions, cache behavior, and limitations.
- **`mlpf/standalone_eval/key4hep/`**: Standalone tools for evaluating model checkpoints and generating performance plots on ROOT files for Key4Hep detectors. Note the ColliderML validator is the shared `tests/validate_parquet.py` (it knows how to handle the `colliderml` detector name).
- **`mlpf/plotting/`**: Comprehensive suite of plotting tools.
  - **`plot_validation.py` / `plot_met_validation.py`**: Standard validation plots for jets and MET.
  - **`corrections.py`**: Derivation of jet energy corrections.
  - **`cmssw_validation_data.py`**: Validation scripts for CMS data.
  - **`cms_fwlite.py`**: CMS FWLite-based event analysis and plotting.
- **`scripts/cms-validate-onnx.py`**: Exports PyTorch models to ONNX (supporting FP32, FP16, Fused Flash Attention, HEPT, HEPTv2, and GNNLSH) and validates inference with per-configuration PyTorch/ONNX sessions.
- **`scripts/plot-onnx-summary.py`**: Generates summary plots for ONNX validation results.
- **`mlpf/model/monitoring.py`**: System resource monitoring and logging.

## 6. Utilities & Miscellaneous
- **`mlpf/utils.py` / `mlpf/logger.py`**: Common utilities and centralized logging.
- **`mlpf/customizations.py`**: Config customization helpers for fast CI/test pipelines.
- **`mlpf/timing.py`**: Performance timing utilities.
- **`mlpf/optimizers/`**: Custom optimizers like LAMB (`lamb.py`).
- **`mlpf/standalone/`**: Standalone, hackable MLPF training and architecture search (`train.py`, `eval.py`, `dsl.py`, `puppi.py`, `plot_evolution.py`, `run_evolution.py`).
- **`mlpf/raytune/`**: Ray Tune integration for hyperparameter search (`search_space.py`, `utils.py`).
- **`mlpf/jet_utils.py`**: Jet clustering and matching logic.
- **`scripts/`**: Miscellaneous utility scripts.
  - **`benchmark.py`**: Benchmarks the forward and backward pass timings and peak memory usage for various model architectures.
  - **`tallinn/`, `lxplus/`, `flatiron/`, `lumi/`, `local/`**: Site-specific and local orchestration scripts for training and evaluation.
  - **`legacy/`**: Older site-specific scripts retained for previous cluster workflows.
  - **`upload_model_hf.py` / `upload_hf.py`**: Utilities for uploading experiment results and model checkpoints to HuggingFace Hub.
  - **`visualize_hits.py`**: Tool for visualizing detector hits and model embeddings using UMAP and 3D plotting.
  - **`local_test_cld.sh` / `local_test_cms.sh`**: Scripts for quick local verification of the pipeline.
  - **`fetch_test_data_cld.sh` / `fetch_test_data_cms.sh`**: Scripts to download test data for CLD and CMS.
- **`tests/`**: Unit tests and validation helpers for configuration, data loading, model forward passes, attention stability, HEPT/HEPTv2 layers, ONNX export, and Key4Hep assignment/ground-truth consistency.

## 7. Local ttbar Detector Comparison Report

### Real ColliderML training validation (1,000 events)

Run `bash scripts/local_test_colliderml_real.sh` outside the restricted GPU sandbox.
This is separate from the synthetic/offline `local_test_colliderml.sh`: it reuses
the pinned real ttbar/no-pileup source shard (downloads it if missing), converts
all 1,000 events with current code, and builds native ArrayRecord TFDS config 10.
Two event-disjoint manual files give 900 training and 100 held-out events. The
held-out split is used for both validation and inference, not as an independent
final test sample. The default run uses one GPU, the production attention
architecture, bfloat16 FlashAttention, batch size 4, and 225 optimizer steps
(one 900-event pass). FlashAttention cannot run with float32; the CPU option
uses float32 math attention instead.
This checks training integration and loss behavior, not physics convergence.

Artifacts live in `local_test_data/colliderml_real/<code-and-event-limit-hash>/`:
converted parquet, PF-only manual shards, TFDS, conversion/build/training logs,
and experiment checkpoints, histories and inference plots. `data_validation.json`
checks every event for finite tensors, compatible shapes/classes, disjoint event
IDs, correct tanLambda/omega slots, and exact native-loader-to-TFDS fidelity.
Cache identity includes conversion/builder code and event limit; old 100-event
comparison caches are not reused. Set `PREPARE_ONLY=1` to stop before training,
`NUM_STEPS` to adjust training duration, and `GPUS=0` for CPU execution.

For a five-pass loss-convergence check with validation/checkpoints each pass:

```bash
NUM_STEPS=1125 VAL_FREQ=225 CHECKPOINT_FREQ=225 LOG_FREQ=25 bash scripts/local_test_colliderml_real.sh
PYTHONPATH="$PWD" .venv/bin/python scripts/plot_training_convergence.py --experiment-dir /path/to/experiment
```

This starts a fresh run with a cosine schedule covering the requested duration;
blindly resuming the one-pass checkpoint restores its old 225-step scheduler.
`convergence.png` shows batch losses, five-point training averages and full
held-out validation losses. Weighted total excludes the first 100 calibration
steps; unweighted component losses remain comparable across calibration.
`convergence.json` stores validation losses and first-to-last relative change.
Training logs are named `training_<num-steps>_steps.log` to retain the short run.

### Running and outputs

From a checkout with its `.venv` prepared (`uv sync --locked`):

```bash
bash scripts/run_detector_feature_comparison.sh
```

The runner uses `.venv/bin/python`, sets `PYTHONPATH` to the checkout, and writes
only local test artifacts. It downloads missing ColliderML source shards,
prepares missing comparison parquet/TFDS, regenerates event displays, and runs
`scripts/compare_detector_features.py`. If MAIA TFDS is missing, it invokes
`scripts/local_test_maia.sh`, which also exercises two-step CPU training.

Outputs are under `local_test_data/detector_comparison/`:

- `real_plots/index.html`: feature sheets, jet matching, checks, and caveats.
- `real_plots/summary.json`: input paths, settings, feature-column maps, finite
  ranges/quantiles, zero fractions, checks, and matching-bin counts.
- `event_displays/index.html`: five-detector event gallery, linked from the report.
- `event_displays/openlab_multi_event_dark.png`: a text-free, wide, dark project
  artwork for ColliderML, CLIC, CLD and IDEA. `scripts/create_openlab_event_mosaic.py`
  builds it from the first two selected event indices in the gallery PNGs,
  using one fixed crop per detector across event rows; cross-detector scales
  differ. The gallery builder regenerates it automatically when two or more
  event indices are requested. Subtle column tints, gutters, and repeated
  circular motifs distinguish sources; the motifs are decorative abstractions,
  not detector geometry.
- `event_displays/openlab_multi_event_dark_2x.png`: 6400 × 3160 high-resolution
  version of the same artwork. Run `bash scripts/run_openlab_event_hires.sh`
  to rerender events 0 and 1 at 300 DPI from the detector data before composing
  the image; this avoids enlarging the lower-resolution gallery PNGs.
- `colliderml_source/provenance.json`: pinned Hugging Face revision and shard paths.

Render only the event gallery, optionally selecting event indices:

```bash
bash scripts/run_detector_event_displays.sh 0 1 2
```

Use `COLLIDERML_DISPLAY_PARQUET=/path/to/current.parquet` to select a particular
converted ColliderML build for this standalone gallery command. The full report
runner supplies the freshly selected build automatically. The default standalone
path is the original `colliderml/train-00000-of-01000.parquet` cache.

### Inputs and available statistics

The default feature limit remains **100 events per detector**, or all available
events if fewer. Parquet uses file/event order; TFDS samples without replacement
using seed **12345** from the combined `train+test` source. The sample is not
balanced between train and test. The actual counts are recorded in the report.

| Detector | Raw input locally available | Comparison parquet | Local PF TFDS train / test |
| --- | --- | --- | --- |
| ColliderML | 1,000 aligned events in the downloaded `ttbar_pu0` shard | First 100 | Built locally: 50 / 50 |
| CLIC | Two 100-event ROOT files, ee ttbar at 380 GeV | First `300000` file: 100 | 45,000 / 5,000, config 1, version 3.2.1 |
| CLD | Two 100-event ROOT files, ee ttbar at 365 GeV | First `300000` file: 100 | 90,000 / 10,000, config 1, version 3.2.1 |
| IDEA | Two 100-event ROOT files, ee ttbar at 365 GeV | First `300000` file: 100 | 8,000 / 2,000, config 1, version 0.1.0 |
| MAIA | Supplied v04 muon-collider ROOT file: **10 events**, despite its `10000` filename | All 10 | Built locally: 5 / 5 |

CLIC/CLD/IDEA source files and TFDS live under `/mnt/work/mlpf/root/<detector>/ttbar/`
and `/mnt/work/mlpf/tensorflow_datasets/<detector>/`. MAIA lives under
`local_test_data/maia_smoke/`; its URL, size, and checksum are recorded in that
directory's `provenance.json`. ColliderML is pp ttbar without pileup, from
`CERN/ColliderML-Release-1` at the revision pinned by the downloader.

Increasing `compare_detector_features.py --num-events` changes only the plotting
limit. To increase ColliderML statistics, reconvert more source events and build
a **new TFDS data directory**; changing the limit does not enlarge existing
caches. CLIC/CLD/IDEA can use 1,000 stored TFDS events immediately, but their local
raw sample totals are only 200 each. More MAIA events require another raw sample.

### Cache behavior after code changes

The runner fingerprints ColliderML converter, track/truth/clustering/reader,
shared target-building, configuration, and native TFDS-builder/loader source
files. It stores the 100-event build in
`colliderml_builds/<fingerprint>/{parquet,manual,tensorflow_datasets}`. Source changes
select a new build instead of silently reusing old features. Existing builds and
the original caches are preserved. This fingerprint is a local cache identity,
not an official dataset-version bump or a record of every environment dependency.

The fingerprint does not cover the downloaded source bytes or configurable event
limits; this runner fixes the pinned shard and 100-event conversion. For different
inputs/statistics, explicitly use separate conversion/TFDS directories. TFDS
`download_and_prepare` otherwise reuses existing dataset versions.

CLIC/CLD/IDEA parquet and MAIA TFDS are reused when present. If their converter
logic changes, explicitly regenerate them into separate directories and update
the comparison paths. Merely rerunning this wrapper does not reconvert every
detector. The local Uproot reader override avoids a FSSpecSource metadata stall;
it does not alter the physics conversion.

### Feature semantics and checks

Both views are normalized to the 17-column track/cluster input matrix and
14-column target matrix. TFDS class indices are mapped back to PDG labels before
comparison. Histograms match **feature meaning**, not raw column index.

After the six shared kinematic columns, ColliderML uses ACTS-derived features,
while the other detectors use EDM4hep track features. In the updated ColliderML
layout, columns 10/11/12 are `tanLambda`/`omega`/`radiusOfInnermostHit`; their
EDM4hep counterparts are columns 11/13/10. `tanLambda = sinh(eta)`, `omega` uses
the nominal 3 T field and the converter's `3e-4` factor, and the innermost radius
is the minimum transverse radius of linked hits. EDM4hep instead reads the
AtFirstHit track-state reference-point radius. ColliderML `n_meas` is a hit count,
not a fitted number of degrees of freedom. Before commit `992f0366`, ColliderML
column 10 incorrectly duplicated `sin(phi)`; archived files require the old
mapping and must not be mixed with the new report mapping.

For each event, `n_target` counts target rows with nonzero PDG. The
`target_active_fraction` is `n_target / (n_track + n_cluster)` (zero for an empty
event). This is label occupancy, **not reconstruction efficiency**: one particle
can leave many input elements while only one representative carries its target.

Histograms count objects, except explicitly event-level quantities. Each row
shares bins and y limits across detectors. The visible range is the union of
each detector's 0.5–99.5% quantiles, so high-multiplicity detectors do not dominate
range selection. Histograms are normalized by the full finite sample count;
out-of-range fractions are annotated, not silently renormalized away. Positive
wide-range features use symlog axes; PDG plots use discrete class bins.

Checks include finiteness, element/target alignment, valid classes and element
types, phi normalization, target jet indices, `p = pt*cosh(eta)`,
`ET = E/cosh(eta)`, and target energy versus momentum. Non-finite values are
counted and excluded from plots. The diagnostic sheet compares `tanLambda` with
`sinh(eta)` for every detector.

Each parquet event is tested through the corresponding TFDS feature
encoder/decoder in float32. Separately, ColliderML and MAIA compare **all** native
loader examples with **all** persisted TFDS records using order-independent
SHA-256 fingerprint multisets. A passing encoder check alone does not prove that
the stored dataset is current. CLIC/CLD/IDEA stored TFDS are independent of the
small local ROOT examples, so their distribution comparisons are not event-wise
round-trip tests. Local smoke TFDS splits are event-disjoint halves; the helper
handles both Key4hep's outer-record format and ColliderML's event-row format.

### Jet matching

The report reuses `mlpf.jet_utils._match_jets_event`: maximum-cardinality,
minimum-DeltaR **one-to-one** matching with wrapped phi and `DeltaR < 0.1`.
There is **no pT-response cut**. Invalid/nonpositive-pT jets are excluded from
matching. Original detector-specific jet algorithms and pT cuts are retained.

- Response: `targetjet_pt / genjet_pt` for matched pairs. Histogram display range
  is 0.5–1.5, with tails annotated. Profiles show median and 16–84% spread versus
  the matched **genjet** pT or eta.
- Solid matched fraction: matched genjets / all genjets (efficiency).
- Dashed matched fraction: matched targetjets / all targetjets (purity).
- Fractions use each collection's own coordinates, common pT/eta bins, and 68%
  Wilson intervals. Empty bins are omitted. Numerator/denominator counts and
  bin edges are saved in `summary.json`.

### Event displays and interpretation

`visualize_key4hep.py` supports CLIC/CLD/IDEA/MAIA; `visualize_maia.py` defaults
to MAIA. `visualize_colliderml.py` adapts release parquet to the same renderer,
uses saved converted clusters when supplied, and derives track helices from
ACTS parameters. The shared view is transverse, with +x left, +y up, +z toward
the viewer and a common ±6500 mm extent. Hits are deterministically sampled up
to 800 per collection/region; their plotted density is not an occupancy measure.
Charged truth guides curve in the nominal axial fields (CLIC 4 T, CLD/IDEA 2 T,
ColliderML 3 T, MAIA 5 T); neutrals are straight. Guides start at the origin,
ignore material/energy loss, and stop after at most one turn. Display envelopes
are approximate, not detailed detector geometry. Equal event indices refer to
**independent collisions**. Truth guides are visible status-1 particles for
Key4hep/MAIA and primary leaves for ColliderML, not final merged/allocated targets.

Different collision types/energies, reconstruction, and target-building rules
mean these plots are **not detector performance rankings**. IDEA tracks are
truth-seeded and candidates are target oracles; ColliderML has no PF baseline.
MAIA has much lower statistics and missing CellIDEncoding metadata; unknown hit
surface fields are zero and are not used by its track/cluster model.

The MAIA validator runs in **report mode**, so a successful smoke script does not
mean all physics gates pass. The report embeds its separate validation JSON;
inspect FAIL/WARN findings rather than relying on exit status. The initial
sample flags H3 (tracker-hit momentum), H4 (calorimeter-hit energy), and P3
(baseline-PF residuals). Polar-boundary eta fallbacks can also violate the cluster
ET relation in stored datasets; the check table reports those rows.

### Regression tests

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/test_detector_jet_comparison.py tests/test_detector_plot_layout.py \
  tests/test_detector_event_displays.py tests/test_maia_pipeline.py \
  tests/test_colliderml_adapter.py tests/test_colliderml_postprocessing.py
```

These cover matching uniqueness/wrapped phi/no response cut, empty-bin behavior,
new ColliderML column mapping, title separation on short and tall sheets,
MAIA identification, charged-trajectory radius/sign, source-event adaptation,
event-wise smoke shard splitting, and native builder/configuration contracts.
