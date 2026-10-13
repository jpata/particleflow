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

# AMD MIGraphX inference only (Linux x86-64, ROCm 10.1.0 installed on the host)
uv sync --project envs/ort-amd
```

Run a command in the matching environment with `uv run --project envs/ort-cpu ...` or `uv run --project envs/ort-gpu ...`.

The CPU and Nvidia projects pin ONNX Runtime **1.29.0**. The Nvidia project
resolves `onnxruntime-gpu` exclusively from Microsoft's CUDA 12 package feed;
its wheel reports CUDA **12.8** and needs a compatible host driver. Do not
replace this with the default PyPI wheel: recent releases there use CUDA 13.
1.29.0 is the newest stable CUDA 12 Python wheel verified in that feed for this
upgrade. No manual copying of provider libraries is needed.

Check the [ORT releases](https://github.com/microsoft/onnxruntime/releases) and
the [CUDA 12 wheel feed](https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/onnxruntime-gpu/)
separately when updating this pin: the latest upstream release need not have a
stable CUDA 12 Python wheel.

The AMD project is a separate Python **3.12** inference environment with
**ONNX Runtime 1.29.0**, AMD's **onnxruntime-ep-migraphx 1.0.0+rocm10.1.0**
plugin, and **MIGraphX / migraphx-libs 2.18.0+rocm10.1.0**. Packages resolve
through explicit AMD indexes; ORT itself comes from PyPI. This replaces the
older monolithic `onnxruntime-migraphx` 1.23.2 package. Run `uv sync` to remove
that package and install the new stack.

The host needs ROCm **10.1.0** and a supported AMD GPU. This environment installs
MIGraphX libraries but omits GPU-specific ROCm extras; it does not install the
host ROCm runtime or driver. Follow AMD's [ONNX Runtime installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/onnxruntime.html)
and [MIGraphX guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/migraphx.html)
for host setup. The plugin must be explicitly registered before device discovery;
`scripts/validate-onnx-runtime.py --provider migraphx` handles registration and
selects the first available MIGraphX device, failing if initialization fails.

The environment remains inference-only, without `particleflow`, PyTorch, TFDS,
or CUDA packages. Export with the CPU/Nvidia environment and replay saved input
batches using the [ONNX validation guide](../validation/onnx.md). Installation
alone does not establish AMD GPU operator support.

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
