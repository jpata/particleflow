# Code and Functionality Map: ParticleFlow

This document provides a high-level overview of the repository's structure and the functionality of its core components.

## 1. Configuration & Specification
The project uses a hierarchical configuration system.

- **`particleflow_spec.yaml`**: The single source of truth for the entire project. It defines machine-specific paths (sites), data production scenarios, and model architectures.
- **`mlpf/conf.py`**: Defines the Pydantic models for the configuration, ensuring type safety and handling path resolution.
- **`pixi.toml`**: Project environment and task management using Pixi. Defines common tasks like `gen`, `post`, `train`, and `validation`.
- **`validation_cms.yaml` / `validation_key4hep.yaml`**: Specification files for validation scenarios.

## 2. Workflow Orchestration
Complex data production and training pipelines are managed using Snakemake.

- **`mlpf/snakemake/`**: Contains scripts to generate Snakemake workflows.
  - **`produce_snakemake.py`**: Generates workflows for generation, postprocessing, TFDS creation, and training.
  - **`produce_cms_validation_snakemake.py`**: Orchestrates validation workflows specifically for CMS.
  - **`produce_validation_snakemake.py`**: Orchestrates validation workflows for Key4Hep detectors (CLD, CLIC).
- **`mlpf/pipeline.py`**: The main CLI for training, testing, and hyperparameter optimization. Supports standard and Ray-based execution.

## 3. Data Production & Preprocessing
- **`mlpf/data/`**: Simulator-specific code for generating and preprocessing data.
  - **`cms/`**: Scripts for CMSSW-based generation (`genjob_pu.sh`) and postprocessing (`postprocessing2.py`).
  - **`key4hep/`**: Scripts for Key4Hep-based generation and postprocessing (`postprocessing.py`).
- **`mlpf/heptfds/`**: TFDS (TensorFlow Datasets) builders for various datasets (CMS, CLD, CLIC), providing a standardized interface for data loading.

## 4. Machine Learning Core (`mlpf/model/`)
- **`mlpf.py`**: Implementation of the MLPF model, featuring multi-head attention and configurable sub-networks.
- **`gnn_lsh.py`**: GNN layers with Locality-Sensitive Hashing (LSH) for scalable graph processing.
- **`PFDataset.py`**: Advanced data loading logic, including dataset interleaving and multi-file handling.
- **`training.py`**: Core training loop implementation, supporting DDP and Ray-based distributed training.
- **`losses.py`**: Specialized loss functions for particle classification and energy regression.
- **`inference.py`**: Utilities for running model inference and generating predictions.
- **`distributed_ray.py`**: Integration with Ray for distributed training and HPO.

## 5. Validation, Plotting & Monitoring
- **`mlpf/standalone_eval/key4hep/`**: Standalone tools for evaluating model checkpoints and generating performance plots for Key4Hep detectors.
- **`mlpf/plotting/`**: Comprehensive suite of plotting tools.
  - **`plot_validation.py` / `plot_met_validation.py`**: Standard validation plots for jets and MET.
  - **`corrections.py`**: Derivation of jet energy corrections.
  - **`cmssw_validation_data.py`**: Validation scripts for CMS data.
- **`mlpf/model/monitoring.py`**: System resource monitoring and logging.

## 6. Utilities & Miscellaneous
- **`mlpf/utils.py` / `mlpf/logger.py`**: Common utilities and centralized logging.
- **`mlpf/optimizers/`**: Custom optimizers like LAMB (`lamb.py`).
- **`mlpf/jet_utils.py`**: Jet clustering and matching logic.
- **`scripts/`**: Miscellaneous utility scripts for local testing, uploading models to HuggingFace, and visualization.
- **`tests/`**: Unit tests for various components of the pipeline.
