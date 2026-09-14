# CLD and CLIC data production

CLD and CLIC share the Key4HEP/EDM4hep production and postprocessing path. This page records their detector-specific inputs and samples; use [Generate a dataset](generate.md) for the common execution procedure.

## Requirements

Both recipes use the Key4HEP AlmaLinux 9 container configured in `particleflow_spec.yaml`. Initialize submodules before generating data:

```bash
git submodule update --init --recursive
```

CLD uses the detector configuration under `mlpf/data/key4hep/gen/cld/CLDConfig`. CLIC uses `mlpf/data/key4hep/gen/clic` and its detector software. Generate files with the EDM4hep collections expected by `mlpf/data/key4hep/postprocessing.py`.

## Training samples

| Production | Centre-of-mass energy | Configured TFDS samples |
|---|---:|---|
| CLD | 365 GeV | `ttbar`, fully hadronic `WW`, inclusive `qq`, and `ZZ` |
| CLICdet | 380 GeV | `ttbar`, fully hadronic `WW`, and inclusive `qq` |

CLD also configures particle-gun and additional 91/240/365 GeV samples for detector studies. The `tfds_mapping` and `tfds_hit_mapping` entries restrict training-data production to the samples in the table.

Track/cluster and hit builders use version 3.2.1 and configuration partitions 1--10. The default model recipes select `ttbar`, `WW`, and `qq`; CLD `ZZ` is available through an additional builder.

## Detector commands and paths

Use one production at a time:

```bash
PROD=cld pixi run gen -- --dry-run --printshellcmds
PROD=clic pixi run gen -- --dry-run --printshellcmds
```

For either detector, the path evolves as:

```text
<workspace>/gen/<EDM4hep-process>/root/*.root
    -> <workspace>/post/<EDM4hep-process>/*.parquet
    -> <workspace>/tfds/<dataset>/<configuration>/<version>/
```

The shared postprocessor writes track/cluster and lower-level hit fields to Parquet. `pixi run tfds` builds track/cluster datasets. `pixi run tfds_hit` builds hit datasets with separate names and model recipes.

## Validate postprocessing

Run strict validation before either TFDS build. For CLD:

```bash
uv run python tests/validate_parquet.py \
  --input /path/to/cld-workspace/post/p8_ee_ttbar_ecm365 \
  --detector cld \
  --max-events 20 \
  --plots-dir validation_plots/cld
```

For CLIC, change the process to `p8_ee_ttbar_ecm380` and `--detector clic`. Passing a directory validates every `.parquet` file directly inside it and gives each file a separate plot/report directory.

The validator checks schema, detector-object relationships, energy accounting, target assignment, hit geometry, and baseline/truth consistency. A zero exit code is the gate for TFDS creation. Review `validation_report.json` and plots as well. A 20-event limit provides a quick gate; campaign-level closure uses a statistically representative sample.

## Build the representations

After validation:

```bash
PROD=cld pixi run tfds

# For studies requiring detector hits
PROD=cld pixi run tfds_hit
```

Read one event from every resulting dataset/configuration using the method in [Download a dataset](download.md). Validate the training Parquet and TFDS artifacts at this stage. The standalone EDM4hep evaluator consumes ROOT files and checkpoints later for model evaluation.
