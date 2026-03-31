# Code and Functionality Map: ParticleFlow

This document provides a high-level overview of the repository's structure and the functionality of its core components.

## 1. Configuration & Specification
The project uses a hierarchical configuration system.

- **`particleflow_spec.yaml`**: The single source of truth for the entire project. It defines:
  - **Sites**: Machine-specific paths and execution settings (e.g., Slurm, Condor).
  - **Productions**: Datasets for CMS, CLD, and CLIC, including generation parameters and postprocessing steps.
  - **Models**: Architecture definitions, training recipes, and hyperparameters.
- **`mlpf/conf.py`**: Defines the Pydantic models and global defaults for the MLPF configuration. It handles type validation and path resolution.

## 2. Workflow Orchestration
The project uses Snakemake for managing complex data production and training pipelines.

- **`mlpf/produce_snakemake.py`**: Generates Snakemake files based on the `particleflow_spec.yaml`. It creates separate workflows for:
  - `gen`: Raw data generation using domain-specific simulators (CMSSW, Key4Hep).
  - `post`: Postprocessing raw output into intermediate formats (Pickle, Parquet).
  - `tfds`: Conversion of intermediate data into TensorFlow Datasets (TFDS).
  - `train`: Model training on the generated datasets.
  - `val`: Validation of the trained model.
- **`mlpf/pipeline.py`**: The main command-line interface for:
  - Standard training and testing (`train`, `test`).
  - Distributed training and hyperparameter optimization using Ray (`ray-train`, `ray-hpo`).

## 3. Data Production Pipeline
- **Generation**: Simulator-specific shell scripts located in `mlpf/data/cms/` and `mlpf/data/key4hep/`.
- **Postprocessing**: Scripts like `mlpf/data/cms/postprocessing2.py` and `mlpf/data/key4hep/postprocessing.py` extract relevant features from simulator output and prepare them for machine learning.
- **TFDS Builders**: Located in `mlpf/heptfds/`, these builders standardize the data format for efficient loading during training.

## 4. Machine Learning Core (`mlpf/model/`)
- **`mlpf.py`**: Implementation of the core MLPF model, primarily using multi-head attention mechanisms.
- **`gnn_lsh.py`**: Implementation of Graph Neural Network (GNN) layers, including Locality-Sensitive Hashing (LSH) for efficient graph construction.
- **`PFDataset.py`**: Handles data loading, collation, and interleaving of different datasets during training.
- **`training.py`**: Implements the training loop, supporting single-node, multi-GPU (DDP), and CPU execution. It manages checkpointing, evaluation cycles, and logging.
- **`losses.py`**: Defines custom loss functions tailored for particle flow reconstruction (classification and regression).
- **`inference.py`**: Logic for running predictions and generating performance metrics.

## 5. Validation & Monitoring
- **`mlpf/produce_validation_snakemake.py`**: Orchestrates validation workflows to compare MLPF performance against standard PF algorithms.
- **`mlpf/plotting/`**: A suite of tools for visualizing model performance, including response plots, efficiency curves, and event displays.
- **`mlpf/model/monitoring.py`**: Utilities for system monitoring (GPU/CPU usage) and Tensorboard logging.

## 6. Project Utilities
- **`mlpf/utils.py`**: General utility functions for path resolution, spec loading, and comet-ml integration.
- **`mlpf/jet_utils.py`**: Utilities for jet clustering and matching.
- **`mlpf/logger.py`**: Centralized logging configuration.
