# Running ParticleFlow on Tallinn Cluster (KBFI)

This document provides instructions on how to set up and run the ParticleFlow Snakemake workflow on the Tallinn Slurm cluster using Pixi for environment management.

## Prerequisites

1.  **Install Pixi** (if not already installed):
    ```bash
    curl -fsSL https://pixi.sh/install.sh | bash
    # Restart your shell or source your .bashrc
    ```
2.  **Storage**: Ensure you are working on a shared file system accessible by the cluster nodes (`/local`, `/scratch/persistent`).

## Setup

### 1. Site Configuration
Run the following command to configure `particleflow_spec.yaml` for Tallinn site defaults:

```bash
pixi run use-tallinn
```

This ensures that the executor is set to `slurm` and that the correct bind mounts and paths for Tallinn are used.

### 2. Initialize the Slurm Profile
Run the following command to initialize the Snakemake Slurm profile:

```bash
pixi run tallinn-init
```

This will create a default Snakemake profile in `.profiles/tallinn/` optimized for the KBFI cluster.

### 3. Install Dependencies
Ensure all necessary packages (Snakemake, Slurm executor plugin, etc.) are installed in the local Pixi environment:

```bash
pixi install
```

## Running the Workflow

### 1. Generate the Snakefile
Generate the Snakemake workflow for a specific production (e.g., `cms_2025_main`). On Tallinn, this typically includes data generation, postprocessing, and TFDS conversion:

```bash
pixi run tallinn-generate
```

### 2. Launch the Workflow
Launch the workflow on Slurm. It is recommended to run this inside a `tmux` or `screen` session:

```bash
pixi run tallinn-run
```

This command will:
- Use the `tallinn` profile to submit jobs to Slurm.
- Use Apptainer for execution.
- Automatically mount the necessary paths (`/local`, `/cvmfs`, `/scratch`).

### 3. Run Validation Plots
To run the validation plot workflow:

```bash
pixi run tallinn-validation
```

## Environment Summary
- **Orchestrator**: Snakemake (managed by Pixi).
- **Execution Environment**: Apptainer containers (specified in `particleflow_spec.yaml`).
- **Batch System**: Slurm.
