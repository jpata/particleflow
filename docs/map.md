# Repository map

Where to find each part of the code. Run Python entry points from the repository root with `uv run python3 ...`.

## Configuration

| Path | Contents |
|---|---|
| `particleflow_spec.yaml` | Sites, data productions (`productions`), and model recipes (`models`); the single source of truth for detectors, dataset versions, and architectures |
| `mlpf/conf.py` | Typed configuration (Pydantic), feature and class definitions per dataset, input-normalization transforms, and recipe resolution (`MLPFConfig.from_spec`) |
| `configs/training/scenarios/`, `configs/training/platforms/` | [Training scenarios](training/scenarios.md) and hardware profiles; see the [scenario README](https://github.com/jpata/particleflow/blob/main/configs/training/README.md) |
| `configs/{local,lxplus,tallinn}/pixi.toml` | Pixi site profiles for production; the root `pixi.toml` is a symlink to one of them |
| `envs/` | Separate ONNX Runtime environments (`ort-cpu`, `ort-gpu`, `ort-amd`) and the EmpireAI GH200 environment |
| `validation_cms.yaml`, `validation_key4hep.yaml` | Detector-validation campaigns |

## Data production

| Path | Contents |
|---|---|
| `mlpf/snakemake/` | Generates Snakemake workflows for simulation, postprocessing, TFDS, training, and validation |
| `mlpf/data/cms/` | CMSSW generation jobs and `postprocessing2.py` (ROOT ntuple → pickle) |
| `mlpf/data/key4hep/` | Key4HEP generation configurations (`gen/`) and `postprocessing.py` (EDM4hep → Parquet) for CLD, CLIC, IDEA, and MAIA |
| `mlpf/data/colliderml/` | ColliderML reader, track features, truth, calorimeter clustering (`clue.py`, `radius_graph.py`), and `postprocessing.py` |
| `mlpf/data/target_building.py` | Shared target allocator: assignment rules (greedy, `majority_energy`), merging, and energy accounting |
| `mlpf/data/full_targets.py` | Cluster-independent target table and per-element particle, instance, and origin labels of the [general datasets](datasets/full.md) |
| `mlpf/heptfds/` | TFDS builders: `<detector>_pf*` for tracks and clusters, `<detector>_full` for the general datasets with tracks, clusters, and hits; shared helpers in `edm4hep_utils/`, `colliderml_utils/`, and `full_utils/` |
| `scripts/cld_ecal_readout/` | CLD calorimeter granularity scan geometries and conversion; see `GRANULARITY_SCAN.md` |
| `scripts/flatiron/colliderml_*.sh` | ColliderML download, conversion, and TFDS chain on Slurm |
| `scripts/tfds_webdav*.py`, `upload_hf.py` | [Grid WebDAV transfer](datasets/webdav.md) and Hugging Face publication |
| `scripts/compute_input_normalization_stats.py` | Per-version input-normalization statistics, written to `mlpf/model/input_normalization_stats/` |

## Model and training

| Path | Contents |
|---|---|
| `mlpf/pipeline.py` | Command-line entry point: `train`, `test`, `lr-find`, `ray-train`, `ray-hpo` |
| `mlpf/model/mlpf.py` | The MLPF model: input encoding, backbones, elementwise heads, prediction decoding |
| `mlpf/model/{heptv2,hept,gnnlsh,litept}.py` | Alternative backbones; multi-head attention is in `mlpf.py` |
| `mlpf/model/input_normalization.py`, `feature_bag.py` | Per-dataset feature standardization and the shared cross-detector feature vocabulary |
| `mlpf/model/calo_entry.py` | Track extrapolation to the calorimeter and cluster centroid directions for HEPTv2 hashing |
| `mlpf/model/set_decoder.py`, `set_losses.py` | Particle-set decoder, Hungarian matching, hit grouping |
| `mlpf/model/tracker_*.py` | Optional tracker features, momentum readout, circle-fit readout, and fit losses for hit models |
| `mlpf/model/PFDataset.py` | TFDS loading, padding, samplers, and weighted mixing of dataset groups |
| `mlpf/model/full_view.py` | Selection of element kinds (`full_elements`) from the general datasets at load time |
| `mlpf/model/training.py`, `losses.py` | Training loop (single GPU, DDP, Ray), checkpointing, losses |
| `mlpf/model/inference.py`, `plots.py`, `validation_metrics.py` | Test-time prediction, particle and jet plots, validation metrics |
| `mlpf/model/reconstruction_diagnostics.py`, `validation_probes.py`, `*_probe.py` | Optional diagnostics and frozen-embedding probes for hit-model studies |
| `mlpf/model/benchmarking.py`, `mlpf/benchmark_report.py` | Opt-in benchmark telemetry and reports |
| `mlpf/training_scenarios.py`, `training_submission.py` | Scenario resolution and Slurm submission |
| `scripts/training/` | `run_scenario.py`, `compare_scenario.py`, and the portable batch-sweep tools |
| `scripts/{flatiron,lumi,tallinn,empireai,local}/` | Site environments and `train_scenario.sh` wrappers |
| `mlpf/raytune/` | Ray Tune search space |
| `mlpf/standalone/` | A separate, minimal MLPF trainer for architecture-search experiments |

## Evaluation and export

| Path | Contents |
|---|---|
| `tests/validate_parquet.py` | [Dataset validation](validation/dataset-validation.md) for Key4HEP and ColliderML Parquet |
| `mlpf/standalone_eval/key4hep/` | Standalone evaluator and plots on EDM4hep ROOT files |
| `mlpf/plotting/` | CMS jet and MET validation, jet energy corrections |
| `mlpf/jet_utils.py` | Jet clustering and one-to-one matching |
| `mlpf/model/onnx_attention.py`, `onnx_graph.py`, `onnx_runtime.py` | [ONNX export](validation/onnx.md), graph rewrites, provider selection |
| `scripts/cms-validate-onnx.py`, `validate-onnx-runtime.py` | PyTorch/ONNX comparison and runtime replay (all track/cluster detectors despite the name) |
| `scripts/qualify-*.py`, `benchmark-*.py`, `*-cms-*.py` | [Export qualification](validation/export-evidence.md) and CMS physics/latency studies |
| `scripts/compare_detector_features.py`, `run_detector_feature_comparison.sh` | [Local detector comparison](validation/detector-comparison.md) |
| `scripts/create_reconstruction_event_assets.py` | [Event displays](datasets/event-displays.md) |
| `scripts/studies/` | One-off study scripts: target-assignment comparisons, hit-model probes, frozen readouts |

## Tests and smoke checks

`tests/` holds the unit suite (`pytest tests`). `scripts/local_test_{cms,cld,clic,colliderml,maia}.sh` run small end-to-end checks; see the [quickstart](getting-started/quickstart.md).
