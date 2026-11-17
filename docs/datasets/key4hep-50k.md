# Joint Key4hep v1.4.0 50k-event campaign

The `key4hep_50k` campaign in `particleflow_spec.yaml` builds all five samples
for CLD, CLIC, IDEA and ILD in one Snakemake DAG. Each sample has **500 jobs ×
100 events = 50,000 events**, for **250,000 events per detector and 1,000,000
events total**. The same seeds can be used across detectors because scratch
paths, outputs and workflow rules are namespaced by detector.

| Samples | CLD / IDEA / ILD | CLIC | Seeds (end exclusive) |
|---|---|---|---|
| ttbar | 365 GeV | 380 GeV | 300000–300500 |
| WW fullhad | 365 GeV | 380 GeV | 400000–400500 |
| qq | 365 GeV | 380 GeV | 500000–500500 |
| Z→qq | 91 GeV | 91 GeV | 600000–600500 |
| Z→tautau | 91 GeV | 91 GeV | 700000–700500 |

FCC-ee samples use the shared cards, including both CLIC Z samples. ILD passes
the matching 91 or 365 GeV energy into reconstruction. CLIC retains its
existing reconstruction and 380 GeV cards for the three continuum samples.
Particle guns and additional ZZ/ZH samples remain available manually; they
are excluded from this campaign. No model training runs in this workflow.

## Build and launch on Tallinn

Update the checkout and recursive submodules, build the development image and
shared overlays following the [simulation build instructions](../../mlpf/data/key4hep/gen/README.md).
The planned software tag is `v1.4.0`; until published, use the pinned PR branch.
No new C++ compilation is needed specifically for the campaign's Python/shell
routing changes if the qualified stable overlays are already installed.
Do not rebuild shared libraries while production jobs are using them.

From the particleflow repository root on Tallinn:

```bash
git pull --ff-only
git submodule update --init --recursive
ln -sfn configs/tallinn/pixi.toml pixi.toml
pixi run init
bash scripts/check_ild_runtime.sh

# Inspect all detector/sample/stage dependencies without submitting jobs.
pixi run datasets_key4hep --jobs 20 --dry-run

# Submit one graph, with at most 20 concurrent SLURM jobs across all detectors.
pixi run datasets_key4hep --jobs 20
```

The task generates `snakemake_jobs/key4hep_v1.4.0_50k/Snakefile` and its four
namespaced production modules, then runs it through the Tallinn profile with
Apptainer. Every dataset depends on all required generation/reconstruction and
postprocessing jobs; successful shards can progress while other samples run.
It builds track/cluster datasets for all 20 detector/sample combinations and
hit datasets for the ten CLD/CLIC combinations. `--jobs` sets the global job
limit. IDEA requests 64 GB and four CPUs per generation job; ILD requests
16 GB, CLD/CLIC 8 GB. The configured runtime limit is 48 hours.

Rerun the same command to resume. Failed jobs block their datasets; partial
production is disabled. ROOT and Parquet shards must each contain 100 events.
Before publishing TFDS, each builder configuration's actual example count is
checked. TFDS configurations copy into their shared output root using merge-safe
directory creation, allowing independent configurations to finish concurrently.

To generate the graph alone, use `pixi run snakefile_key4hep`. The root graph
also exposes `gen`, `post`, `tfds`, and `tfds_hit` targets. For a subset of
detectors, run `PYTHONPATH=. python3 mlpf/snakemake/produce_key4hep_snakemake.py
--productions cld ild` and invoke Snakemake directly with the same profile,
container arguments and generated Snakefile. The default Pixi dataset task
regenerates the complete four-detector graph.

## Outputs and versions

Outputs are under `/local/joosep/mlpf/DETECTOR/v1.4.0/50k/{gen,post,tfds}`.
The `/50k` workspace and campaign-specific completion markers isolate this run
from the earlier smoke/10k campaigns. Keep the new generated jobs directory
when resuming; old production completion markers are not required.

| Dataset | TFDS version | Configurations | Train / test events |
|---|---|---|---|
| CLD/CLIC tracks and clusters | 3.4.0 | 1–10 | 4,500 / 500 per configuration |
| CLD/CLIC hits | 3.3.0 | 1–10 | 4,500 / 500 per configuration |
| IDEA tracks and clusters | 0.5.0 | 1 | 40,000 / 10,000 |
| ILD tracks and clusters | 0.2.0 | 1 | 45,000 / 5,000 |

All datasets use ArrayRecord. CLD/CLIC/ILD split independent input files;
IDEA retains its deterministic event split. Budget storage for full simulation
intermediates, published ROOT files, Parquet, TFDS, and per-job scratch.
Production is intended for Tallinn; no simulation or batch jobs are launched
by generating or dry-running the workflow locally.

The software compatibility checks and DAG validation do not establish physics
performance. IDEA's fitted-track efficiency and target/truth jet behavior
remain qualification topics; the new 91 GeV reconstruction samples require
validation after generation.

## Generation smoke test

Run this on Tallinn after the same runtime and profile setup.

```bash
pixi run gen_key4hep_smoke --jobs 20 --dry-run
pixi run gen_key4hep_smoke --jobs 20
```

The smoke workflow uses the first ten seeds per detector and sample and keeps
the production setting of 100 events per job. It runs 200 simulation and
reconstruction jobs with the same ROOT event-count checks as production.
Outputs go under each campaign workspace's `smoke/gen/` directory and
completion markers go under `snakemake_jobs/key4hep_v1.4.0_50k_smoke/`.
