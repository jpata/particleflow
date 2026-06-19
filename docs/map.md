# Code and Functionality Map: ParticleFlow

This document provides a high-level overview of the repository's structure and the functionality of its core components.
To run the python code in the right environment, use `uv run python3 ...`.

## 1. Configuration & Specification
The project uses a hierarchical configuration system.

- **`particleflow_spec.yaml`**: The single source of truth for the entire project. It defines machine-specific paths (sites), data production scenarios, and model architectures.
- **`mlpf/conf.py`**: Defines the Pydantic models for the configuration, ensuring type safety and handling path resolution.
- **`mlpf/pipeline.py`**: Implements hierarchical configuration resolution: base defaults in `mlpf/conf.py` are overridden by scenario-specific values in `particleflow_spec.yaml`, which can further be overridden via command-line arguments (e.g., `--model.num_convs 6`).
- **`configs/`**: Site-specific Pixi environment configurations (`local/`, `lxplus/`, `tallinn/`). The root `pixi.toml` is a symlink into this directory.
- **`pixi.toml` / `uv.lock` / `uv.singularity`**: Project environment, container definitions, and task management. Defines common tasks like `gen`, `post`, `train`, and `validation`.
- **`envs/`**: Isolated virtual environment specifications (e.g., `ort-cpu`, `ort-gpu`) for specific runtimes like ONNX.
- **`validation_cms.yaml` / `validation_key4hep.yaml`**: Specification files for validation scenarios.

## 2. Workflow Orchestration
Complex data production and training pipelines are managed using Snakemake or site-specific shell scripts.

- **`mlpf/snakemake/`**: Contains scripts to generate Snakemake workflows.
  - **`produce_snakemake.py`**: Generates workflows for generation, postprocessing, TFDS creation, and training.
  - **`produce_cms_validation_snakemake.py`**: Orchestrates validation workflows specifically for CMS.
  - **`produce_validation_snakemake.py`**: Orchestrates validation workflows for Key4Hep detectors (CLD, CLIC).
- **`mlpf/pipeline.py`**: The main CLI for training, testing, and hyperparameter optimization. Supports standard and Ray-based execution.

## 3. Data Production & Preprocessing
- **`mlpf/data/`**: Simulator-specific code for generating and preprocessing data.
  - **`cms/`**: Scripts for CMSSW-based generation (`genjob_pu.sh`), postprocessing (`postprocessing2.py`), validation (`valjob.sh`, `valjob_data.sh`), and plotting (`plot_cms.py`).
  - **`key4hep/`**: Scripts for Key4Hep-based generation (`gen/`), postprocessing (`postprocessing.py`), and plotting (`plot_postprocessing.py`).
- **`mlpf/heptfds/`**: TFDS (TensorFlow Datasets) builders for various datasets (CMS, CLD, CLIC), including support for both cluster-based and raw hits-based data (`cld_pf_edm4hep_hits`, `clic_pf_edm4hep_hits`). Shared EDM4Hep utilities in `edm4hep_utils/`.

## 4. Machine Learning Core (`mlpf/model/`)
- **`mlpf.py`**: Implementation of the MLPF model, featuring multi-head attention and configurable sub-networks. Supports fused attention and simplified math attention for ONNX export.
- **`gnnlsh.py`**: GNN layers with Locality-Sensitive Hashing (LSH) for scalable graph processing.
- **`hept.py`**: Implementation of the Hashing-based Efficient Particle Transformer (HEPT).
- **`litept.py`**: Integration for the LitePT (Lightweight Point Transformer) architecture.
- **`PFDataset.py`**: Advanced data loading logic, including dataset interleaving and multi-file handling.
- **`training.py`**: Core training loop implementation, supporting DDP and Ray-based distributed training.
- **`losses.py`**: Specialized loss functions for particle classification and energy regression.
- **`inference.py`**: Utilities for running model inference and generating predictions.
- **`plots.py`**: Confusion matrix logging and model performance plotting during training.
- **`utils.py`**: Model-level utility functions including target unpacking, learning rate schedules, and metric computation.
- **`distributed_ray.py`**: Integration with Ray for distributed training and HPO.

## 5. Validation, Plotting & Monitoring
- **`mlpf/standalone_eval/key4hep/`**: Standalone tools for evaluating model checkpoints and generating performance plots for Key4Hep detectors.
- **`mlpf/plotting/`**: Comprehensive suite of plotting tools.
  - **`plot_validation.py` / `plot_met_validation.py`**: Standard validation plots for jets and MET.
  - **`corrections.py`**: Derivation of jet energy corrections.
  - **`cmssw_validation_data.py`**: Validation scripts for CMS data.
  - **`cms_fwlite.py`**: CMS FWLite-based event analysis and plotting.
- **`scripts/cms-validate-onnx.py`**: Exports PyTorch models to ONNX (supporting FP32, FP16, and Fused Flash Attention) and validates inference.
- **`scripts/plot-onnx-summary.py`**: Generates summary plots for ONNX validation results.
- **`mlpf/model/monitoring.py`**: System resource monitoring and logging.

## 6. Utilities & Miscellaneous
- **`mlpf/utils.py` / `mlpf/logger.py`**: Common utilities and centralized logging.
- **`mlpf/customizations.py`**: Config customization helpers for fast CI/test pipelines.
- **`mlpf/timing.py`**: Performance timing utilities.
- **`mlpf/optimizers/`**: Custom optimizers like LAMB (`lamb.py`).
- **`mlpf/standalone/`**: Standalone training and evaluation scripts (`train.py`, `eval.py`, `dsl.py`, `puppi.py`, `plot_evolution.py`, `run_evolution.py`).
- **`mlpf/raytune/`**: Ray Tune integration for hyperparameter search (`search_space.py`, `utils.py`).
- **`mlpf/jet_utils.py`**: Jet clustering and matching logic.
- **`scripts/`**: Miscellaneous utility scripts.
  - **`benchmark.py`**: Benchmarks the forward and backward pass timings and peak memory usage for various model architectures.
  - **`tallinn/`, `lxplus/`, `flatiron/`**: Site-specific orchestration scripts for training and evaluation on different clusters.
  - **`upload_model_hf.py` / `upload_hf.py`**: Utilities for uploading experiment results and model checkpoints to HuggingFace Hub.
  - **`visualize_hits.py`**: Tool for visualizing detector hits and model embeddings using UMAP and 3D plotting.
  - **`visualize_assignment_graphs.py`**: Visualization of particle-to-element assignment graphs.
  - **`visualize_gt_hit_labels.py`**: Analysis and visualization of ground-truth hit labels.
  - **`visualize_oc.py`**: Overlap-counting hit assignment visualizations.
  - **`local_test_cld.sh` / `local_test_cms.sh`**: Scripts for quick local verification of the pipeline.
  - **`fetch_test_data_cld.sh` / `fetch_test_data_cms.sh`**: Scripts to download test data for CLD and CMS.
- **`tests/`**: Unit tests for various components of the pipeline.
