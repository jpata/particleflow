# Installation

## Requirements

MLPF requires Python 3.11. The main Python environment is managed with [uv](https://docs.astral.sh/uv/). Large-scale detector simulation and data production also use [Pixi](https://pixi.sh/), Snakemake, and Apptainer.

The Linux environment includes CUDA-oriented packages such as FlashAttention and LitePT dependencies. A recent Nvidia/CUDA system is the most direct route for full training. The CPU path is useful for tests and very short smoke runs.

## Clone the repository

Some detector configurations are Git submodules, so clone recursively:

```bash
git clone --recurse-submodules https://github.com/jpata/particleflow.git
cd particleflow
```

If the repository was cloned without submodules, initialize them before producing simulation:

```bash
git submodule update --init --recursive
```

## Create the main environment

Install uv, then create the environment from the repository root:

```bash
uv sync
```

Check that the command-line interface can be loaded:

```bash
uv run mlpf --help
```

The first installation can be large because it includes the training, data-processing, plotting, and documentation dependencies.

## ONNX Runtime environments

ONNX Runtime is kept in small, separate projects so that its CPU and GPU packages do not conflict:

```bash
# CPU inference and numerical validation
uv sync --project envs/ort-cpu

# Nvidia GPU inference and numerical validation
uv sync --project envs/ort-gpu
```

Run a command in the matching environment with `uv run --project envs/ort-cpu ...` or `uv run --project envs/ort-gpu ...`.

## Prepared container

A prepared Apptainer image is useful on a machine that already has Apptainer and, for GPU work, Nvidia container support:

```bash
apptainer exec --nv \
  https://jpata.web.cern.ch/jpata/pytorch-20260305-08d6950.sif \
  ./scripts/local_test_cld.sh
```

The image covers the Python training environment. The full CMS and Key4HEP simulation workflows use their own detector-software containers configured in `particleflow_spec.yaml`.

## Full production environment

Pixi is the entry point for the Snakemake workflows that generate and process large datasets. The repository contains site profiles for local use, LXPlus, and Tallinn. These profiles contain storage, scheduler, and container assumptions; review them before launching work.

Do not begin with the production commands if your goal is only to train or evaluate MLPF. Start with the [workflow chooser](choose-a-workflow.md).
