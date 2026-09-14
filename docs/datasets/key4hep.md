# CLD and CLIC data production

CLD and CLIC share the Key4HEP/EDM4hep production and postprocessing path. This page records their detector-specific inputs and samples; use [Generate a dataset](generate.md) for the common execution procedure.

## Requirements

Both recipes use the Key4HEP AlmaLinux 9 container configured in `particleflow_spec.yaml`. Initialize submodules before generating data:

```bash
git submodule update --init --recursive
```

CLD uses the detector configuration under `mlpf/data/key4hep/gen/cld/CLDConfig`. CLIC uses `mlpf/data/key4hep/gen/clic` and its detector software. The generated file must contain the EDM4hep collections expected by `mlpf/data/key4hep/postprocessing.py`; an arbitrary EDM4hep file is not sufficient.

## Training samples

| Production | Centre-of-mass energy | Configured TFDS samples |
|---|---:|---|
| CLD | 365 GeV | `ttbar`, fully hadronic `WW`, inclusive `qq`, and `ZZ` |
| CLICdet | 380 GeV | `ttbar`, fully hadronic `WW`, and inclusive `qq` |

CLD also configures particle-gun and additional 91/240/365 GeV samples for detector studies. They are intentionally absent from `tfds_mapping` and `tfds_hit_mapping`, so the production workflow does not silently add them to training data.

Track/cluster and hit builders use version 3.2.1 and configuration partitions 1--10. The default model recipes train on `ttbar`, `WW`, and `qq`; CLD `ZZ` builders exist but are not selected by those defaults.

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

The shared postprocessor writes both track/cluster and lower-level hit fields to Parquet. `pixi run tfds` selects the former, while `pixi run tfds_hit` selects the latter; the two outputs have different dataset names and model recipes.

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

The validator checks schema, detector-object relationships, energy accounting, target assignment, hit geometry, and baseline/truth consistency. A zero exit code is the gate for TFDS creation. Review `validation_report.json` and plots as well; limiting to 20 events is suitable for a quick gate, not campaign-level closure.

## Build the representations

After validation:

```bash
PROD=cld pixi run tfds

# Only when the study requires detector hits
PROD=cld pixi run tfds_hit
```

Read one event from every resulting dataset/configuration using the method in [Download a dataset](download.md). The standalone EDM4hep evaluator consumes ROOT files and checkpoints later in the workflow; it does not replace validation of the training Parquet and TFDS artifacts.
