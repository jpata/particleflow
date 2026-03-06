# Running ParticleFlow on LXPLUS with HTCondor

This document provides instructions on how to set up and run the ParticleFlow Snakemake workflow on LXPLUS using Pixi for environment management and HTCondor (`lxbatch`) for job execution.

## Prerequisites

1.  **Install Pixi** (if not already installed):
    ```bash
    curl -fsSL https://pixi.sh/install.sh | bash
    # Restart your shell or source your .bashrc
    ```
2.  **Use Work Storage**: Ensure you are working in your `/afs/cern.ch/work/` or `/eos/user/` directory, as the home directory has limited storage quota.

## Setup

### 1. Site Configuration
Run the following command to configure `particleflow_spec.yaml` for LXPLUS site defaults:

```bash
pixi run use-lxplus
```

This ensures that the executor is set to `condor` and that the correct bind mounts and paths for LXPLUS are used.

### 2. Initialize the HTCondor Profile
Run the following command to initialize the Snakemake HTCondor profile:

```bash
pixi run lxplus-init
```

This will download the official HTCondor profile template and configure it for CERN LXPLUS.

### 3. Install Dependencies
Ensure all necessary packages (Snakemake, plugins, etc.) are installed in the local Pixi environment:

```bash
pixi install
```

## Running the Workflow

### 1. Generate the Snakefile
Generate the Snakemake workflow for a specific production (e.g., `cms_2025_main`):

```bash
pixi run lxplus-generate
```

This script reads `particleflow_spec.yaml` and creates a `Snakefile` in `snakemake_jobs/cms_2025_main/`.

### 2. Launch the Workflow
Launch the workflow on HTCondor. It is recommended to run this inside a `tmux` or `screen` session:

```bash
pixi run lxplus-run
```

This command will:
- Use the `lxbatch` profile to submit jobs to HTCondor.
- Use Apptainer for execution (to ensure CMSSW and PyTorch compatibility).
- Automatically mount the necessary paths (`/cvmfs`, `/eos`, `/afs`).

### 3. Monitor Progress
- **Snakemake Output**: Monitor the terminal for Snakemake's progress.
- **HTCondor Logs**: Detailed HTCondor logs and errors can be found in the `.condor_jobs/` directory.
- **Condor Status**: Use `condor_q` to see your running and idle jobs.

## Environment Summary
- **Orchestrator**: Snakemake (managed by Pixi).
- **Execution Environment**: Apptainer containers (specified in `particleflow_spec.yaml`).
- **Batch System**: HTCondor (`lxbatch`).
