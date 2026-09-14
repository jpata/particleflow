# Artifacts and directories

MLPF uses two main directory trees: a production workspace for data and an experiment directory for a model run. Paths can be changed by site and command-line configuration, so treat the resolved configuration as authoritative.

## Production workspace

```text
<workspace>/
├── gen/       detector-simulation ROOT files
├── post/      detector-specific Parquet or pickle output
├── tfds/      versioned TensorFlow Datasets
├── val/       reconstruction outputs used for validation
└── plots.../  campaign validation products
```

Sentinel files and generated Snakemake job directories describe workflow completion. Validation reports establish the quality of the corresponding data product. See [Dataset generation](../datasets/generate.md).

## Experiment directory

```text
experiments/<run>/
├── checkpoints/
│   └── checkpoint-<step>.pth
├── history/
│   └── step_<step>.json
├── model_kwargs.pkl
├── hyperparameters.json
├── particleflow_spec.yaml
├── train-config.yaml or test-config.yaml
├── train.log or test.log
├── runs/
│   ├── train/
│   └── valid/
├── preds_step_<step>/
│   └── <dataset>/
└── validation/
```

Some entries appear only when their corresponding training, test, plotting, logging, or validation action runs.

| Artifact | Purpose | Keep for inference? |
|---|---|---:|
| `checkpoint-*.pth` | Model weights plus optimizer, scheduler, RNG, and loader state | Yes |
| `model_kwargs.pkl` | Resolved typed configuration used to instantiate the model | Yes |
| `hyperparameters.json` | Human- and tool-readable configuration plus parameter count | Recommended |
| saved `particleflow_spec.yaml` | Complete recipe snapshot used by the command | Recommended |
| `*-config.yaml` | Resolved command configuration | Recommended |
| `history/*.json` | Training and validation loss and selected plot metrics | For audit |
| logs and `runs/` TensorBoard data | Progress, diagnostics, resource and loss history | For audit |
| prediction Parquet and plots | Held-out model and physics checks | With published validation |

Use the name `best` when the bundle records the selection metric, dataset, and step. Periodic checkpoint names encode the training step; validation records describe model quality.

## Minimum reproducible model bundle

Keep the checkpoint, `model_kwargs.pkl`, resolved YAML configuration, repository commit, dataset names/configuration partitions/versions, and a short validation record together. If the checkpoint is moved, pass `--config` explicitly to make configuration discovery independent of directory layout.

Dataset schema versions, Python package versions, production campaign versions, and model release labels are separate identifiers. Compatibility comes from an explicit check of each recorded identifier.
