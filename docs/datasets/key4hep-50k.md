# Key4hep v1.4.0 production

The `key4hep_50k` campaign in `particleflow_spec.yaml` produces the
track/cluster datasets for CLD, CLICdet, IDEA and ILD@FCC-ee, and the
[general datasets](full.md) with hits for the same detectors, from a single
Snakemake graph. It runs
generation, simulation, reconstruction, postprocessing and TFDS building; it
does not train models.

## Samples

Every generation job produces **10 events**. CLD, CLICdet and ILD use 5,000
jobs (**50,000 events**) per sample; IDEA uses 1,000 jobs (**10,000 events**)
per sample. In total the campaign produces 800,000 events.

| Sample | Energy (CLD, IDEA, ILD) | Energy (CLICdet) | Seeds (CLD, CLICdet, ILD) | Seeds (IDEA) |
|---|---|---|---|---|
| ttbar | 365 GeV | 380 GeV | 300000–304999 | 300000–300999 |
| Fully hadronic WW | 365 GeV | 380 GeV | 400000–404999 | 400000–400999 |
| Inclusive qq | 365 GeV | 380 GeV | 500000–504999 | 500000–500999 |
| Z→qq | 91 GeV | 91 GeV | 600000–604999 | 600000–600999 |
| Z→tautau | 91 GeV | 91 GeV | 700000–704999 | 700000–700999 |

All detectors share the seed ranges; output paths, scratch directories and
rule names are separated by detector. The FCC-ee generator cards are shared,
including for the CLICdet Z samples; CLICdet keeps its own 380 GeV cards for
the other three samples. Particle guns and the additional ZZ and ZH samples
can still be generated manually but are not part of the campaign.

## Datasets

| Dataset | TFDS version | Configurations | Train / test events |
|---|---|---|---|
| CLD and CLICdet tracks and clusters | 3.4.0 | 1–10 | 4,500 / 500 per configuration |
| CLD, CLICdet and ILD general (tracks, clusters, hits) | 1.0.0 | 1–10 | 4,500 / 500 per configuration |
| IDEA general (tracks, clusters, hits) | 1.0.0 | 1–10 | 900 / 100 per configuration |
| IDEA tracks and clusters | 0.5.0 | 1 | 8,000 / 2,000 |
| ILD tracks and clusters | 0.2.0 | 1 | 45,000 / 5,000 |

Configuration 1 of every track/cluster dataset is [published](catalog.md#published-availability) on the Hub; the general datasets are not yet. All datasets are written in ArrayRecord format. CLD, CLICdet and ILD split
train and test by ROOT file; IDEA uses a deterministic per-event split.
Detector-specific inputs and targets are described on the [IDEA](idea.md) and
[ILD](ild.md) pages and in [CLD and CLIC production](key4hep.md).

These datasets have passed software and workflow checks, not physics
validation. In particular, IDEA's fitted-track efficiency and the 91 GeV
samples for all detectors still need to be qualified.

## Running on Tallinn

Build the simulation software once per host, following the
[simulation build instructions](../../mlpf/data/key4hep/gen/README.md). The
build uses the `mlpf/data/key4hep/gen` submodule at the commit pinned in this
repository; it installs the patched ILD mass refitter and downloads the IDEA
tracking model, which jobs then use automatically. Do not rebuild the shared
libraries while production jobs are running.

From the repository root:

```bash
git pull --ff-only
git submodule update --init --recursive
ln -sfn configs/tallinn/pixi.toml pixi.toml
pixi run init
bash scripts/check_ild_runtime.sh

# Show the full job graph without submitting anything.
pixi run datasets_key4hep --jobs 20 --dry-run

# Submit, with at most 20 concurrent Slurm jobs across all detectors.
pixi run datasets_key4hep --jobs 20
```

`datasets_key4hep` writes `snakemake_jobs/key4hep_v1.4.0_50k/Snakefile`, with
one module per detector, and runs it with the Tallinn Slurm profile inside
Apptainer. Each dataset waits for all of its generation and postprocessing
jobs, but different samples and detectors proceed independently. Jobs have a
48-hour time limit.

| Detector | Generation and reconstruction memory | CPUs | Postprocessing memory |
|---|---:|---:|---:|
| CLD | 2,800 MiB | 1 | 1,000 MiB |
| CLICdet | 2,800 MiB | 1 | 1,000 MiB |
| IDEA | 28,400 MiB | 1 | 1,000 MiB |
| ILD | 2,900 MiB | 1 | 1,200 MiB |

The generation and reconstruction requests are the highest measured peak plus
20%, rounded up to the next 100 MiB.

To resume after a failure, rerun the same command. A failed job blocks the
datasets that depend on it; datasets are never built from a partial set of
shards. Every ROOT file must contain 10 events. Events that postprocessing
cannot convert, for example events without any visible particle, are dropped
and reported without failing the file.

Other entry points:

- `pixi run snakefile_key4hep` only writes the Snakefile.
- The graph has `gen`, `post`, `tfds` and `tfds_full` targets in addition to
  `all`.
- For a subset of detectors, run
  `PYTHONPATH=. python3 mlpf/snakemake/produce_key4hep_snakemake.py --productions cld ild`
  and call Snakemake directly with the same profile, Apptainer arguments and
  Snakefile.

## Outputs

Each detector writes to `/local/joosep/mlpf/<detector>/v1.4.0/50k/` with
`gen/`, `post/` and `tfds/` subdirectories. Completion markers are kept under
`snakemake_jobs/key4hep_v1.4.0_50k/`, separate from earlier productions; keep
that directory when resuming. Plan storage for the simulation intermediates,
the ROOT and Parquet files, the TFDS datasets and the per-job scratch space.

Generating the Snakefile or running a dry run does not start any simulation
or batch jobs.

## Smoke test

Before a full production, run ten jobs per detector and sample (200 jobs in
total) with the same setup:

```bash
pixi run gen_key4hep_smoke --dry-run
pixi run gen_key4hep_smoke
```

The smoke test runs 40 concurrent jobs by default; pass `--jobs N` to change
this. It applies the same ROOT event-count checks as the production. Outputs go
to `smoke_10ev/gen/` in each detector's workspace, and completion markers to
`snakemake_jobs/key4hep_v1.4.0_50k_smoke_10ev/`.

Each job attempt writes its logs to
`/local/joosep/mlpf/<detector>/v1.4.0/50k/smoke_10ev/logs/<process>/<seed>/<job-id>-<restart>/`:
`generation.log`, `simulation.log` and `reconstruction.log`. A missing log
means that stage did not start. The Slurm output names the log directory and
contains wrapper and output-validation errors. Jobs remove their node-local
scratch directory on exit.
