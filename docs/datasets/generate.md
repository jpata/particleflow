# Generate a dataset

Use this workflow only when published data do not meet the study's needs. New users should normally [download a prepared dataset](download.md), because detector production is site dependent and can require days, many batch jobs, and hundreds of GB or more.

This page owns the shared Pixi/Snakemake procedure. Read [CMS data production](cms.md) or [CLD and CLIC production](key4hep.md) before launching a detector-specific campaign. For a two-file software check, use the [quickstart](../getting-started/quickstart.md) instead.

## Stages and gates

| Stage | Command | Input | Output | Gate before continuing |
|---|---|---|---|---|
| Simulation | `pixi run gen` | Production sample and detector configuration | ROOT/EDM files below `gen/` | Expected files exist and generation logs finish successfully |
| Postprocessing | `pixi run post` | Generated ROOT/EDM files | CMS pickle or Key4HEP Parquet below `post/` | Files open; for CLD/CLIC, strict Parquet validation passes |
| TFDS, track/cluster | `pixi run tfds` | Postprocessed files | Versioned TFDS directories below `tfds/` | Builder metadata and one event can be read |
| TFDS, hits | `pixi run tfds_hit` | Key4HEP Parquet containing hit fields | Versioned hit TFDS below `tfds/` | Metadata/event read plus hit validation passes |

Training and physics validation are downstream consumers, not evidence that every upstream event is correct. Dataset integrity should be checked before a long training run.

## Select and review a site

Install Pixi and Apptainer as described in [Installation](../getting-started/installation.md), then initialize the detector submodules:

```bash
git submodule update --init --recursive
```

`pixi.toml` is a tracked symbolic link. Point it explicitly at one profile; `ln -s` without replacement fails because the link already exists:

```bash
ln -sfn configs/local/pixi.toml pixi.toml
```

Use `configs/tallinn/pixi.toml` or `configs/lxplus/pixi.toml` only at those sites. Review its activation variables, scheduler profile, bind mounts, and storage paths. Tallinn is the reference full-production setup; local is intended for constrained work and LXPlus is only partially tested.

Initialize the selected profile:

```bash
pixi run init
```

:::{warning}
Every production task depends on the profile's `setup` task. That task edits the active YAML merge key in `particleflow_spec.yaml` to select the site. Both the `pixi.toml` link and the specification can therefore appear modified in `git status`. Review those changes and do not accidentally commit personal paths or a site-only selection.
:::

Before running anything, review the selected entry under `productions` in `particleflow_spec.yaml`: workspace, sample list, seed ranges, events per job, containers, memory, runtime, and TFDS version. The checked-in paths are examples for the configured sites, not portable defaults.

## Generate and inspect the workflow

Choose exactly one production name: `cms_run3`, `cld`, or `clic`.

```bash
PROD=cld pixi run snakefile
```

This writes step-specific Snakefiles and job scripts below `snakemake_jobs/cld/`. Inspect the generated shell scripts before submission. Preview the simulation DAG and commands without running jobs:

```bash
PROD=cld pixi run gen -- --dry-run --printshellcmds
```

For a first production test, preview one Snakemake batch rather than the full target:

```bash
BATCH=1/100 PROD=cld pixi run gen -- --dry-run --printshellcmds
```

After checking the selected jobs, resource requests, output paths, and container mounts, remove `--dry-run`. `BATCH=1/100` partitions the top-level `all` input set; it is not a promise of exactly one event or one job. Inspect the dry-run output before using it as a limited launch.

## Run one stage at a time

```bash
PROD=cld pixi run gen
PROD=cld pixi run post
PROD=cld pixi run tfds
```

For CLD or CLIC hit inputs, build the additional representation only after validating the Parquet files:

```bash
PROD=cld pixi run tfds_hit
```

Do not use brace syntax such as `PROD={cms_run3,cld,clic}`: it assigns one literal shell value rather than launching three reviewed campaigns. Run and validate each production separately.

Generated scripts skip final data files that already exist, and Snakemake records completion with `.done` sentinels below `snakemake_jobs/<production>/`. A failed invocation can normally be rerun. Before deleting a sentinel, compare it with the corresponding data file and log; removing markers indiscriminately can repeat expensive work.

## Validate the products

For a CLD or CLIC Parquet file:

```bash
uv run python tests/validate_parquet.py \
  --input /path/to/workspace/post/p8_ee_ttbar_ecm365/reco_p8_ee_ttbar_ecm365_300000.parquet \
  --detector cld \
  --max-events 20 \
  --plots-dir validation_plots/cld
```

Strict mode is the default and exits nonzero on a failed gate. It writes `validation_report.json` plus diagnostic plots. Use the matching `clic` detector and process path for CLIC. CMS postprocessing emits compressed pickle rather than this Parquet schema; follow the [CMS-specific checks](cms.md).

After TFDS creation, adapt the metadata and single-event checks in [Download a dataset](download.md) to the workspace's `tfds/` directory and the exact configured version.

## Workspace layout

The resolved `productions.<name>.workspace_dir` contains:

```text
<workspace>/
├── gen/    # detector simulation ROOT/EDM output
├── post/   # postprocessed pickle or Parquet
└── tfds/   # dataset-name/configuration/version directories
```

Generated workflow definitions, job scripts, logs, and `.done` markers remain in the repository under `snakemake_jobs/<production>/`. The detector pages describe the extra directory level used by CMS and the process directories used by Key4HEP.

## Publish selected outputs

Publishing changes a shared external repository. Authenticate with a Hugging Face account that has write access, and always preview the exact selection:

```bash
# One TFDS configuration and version
uv run python scripts/upload_hf.py tfds clic 1 \
  --version 3.2.1 \
  --dry-run

# At most two ROOT files from each configured CLIC sample
uv run python scripts/upload_hf.py root clic \
  --num-files 2 \
  --dry-run

# Parquet files corresponding to ROOT files already published on the Hub
uv run python scripts/upload_hf.py parquet clic \
  --dry-run
```

Use repeatable `--sample NAME` or `--dataset NAME` filters to reduce the selection. Remove `--dry-run` only after verifying the destination paths, version, file count, and total size. Existing files with the same path and size are skipped; a different size is rejected unless `--force` is explicitly supplied.

For a campaign outside `particleflow_spec.yaml`, pass a reviewed `--workspace-dir`. Never copy a developer-specific absolute path into committed documentation or automation.
