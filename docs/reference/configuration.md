# Configuration

A training or evaluation run selects a model recipe and a production from `particleflow_spec.yaml`. The recipe defines the model and datasets; the production supplies the detector type and workspace. Command-line options customize the run. Use the [dataset catalog](../datasets/catalog.md) to choose a compatible production, recipe, and dataset version.

## Configuration files

| Location | Contents |
|---|---|
| `particleflow_spec.yaml`: `project` | Site paths, containers, mounts, and scheduler settings |
| `productions.<name>` | Simulation and processing settings, samples, workspace, and production resources |
| `models.defaults` | Common run defaults |
| `models.<name>` | Dataset family, architecture, hyperparameters, dataset groups, and model resource settings |
| `mlpf/conf.py` | Typed run fields, base defaults, and validation rules |
| `validation_key4hep.yaml`, `validation_cms.yaml` | Detector-validation scenarios, checkpoints, samples, and resources |

The [scenario runner](../../configs/training/README.md) also uses files under `configs/training/` to combine scientific training settings with a hardware profile. The [generation guide](../datasets/generate.md) explains the production and site settings used by Pixi and Snakemake.

The selected site is merged into `project` through a YAML anchor. `PF_SITE=local` selects `project.sites.local` when loading the specification; use an existing site name and configure its paths first. References such as `${project.paths.storage_root}` resolve against the loaded specification. Check the resolved paths before running: an unknown reference remains in the string.

## How a run is resolved

`MLPFConfig.from_spec` combines settings in this order:

1. Read `models.defaults`, then the selected recipe. Its `hyperparameters` entries become top-level run fields, and `architecture` becomes `model`.
2. Select the recipe's `dataset`, falling back to the production's `type`. Derive feature dimensions and class labels from that dataset family. Set `data_dir` to `<production workspace>/tfds`.
3. Apply parsed command-line options, followed by extra field overrides such as `--lr 0.0001` or `--model.backbone.num_convs 2`.
4. Expand the recipe's dataset groups, filter configuration partitions with `data_config`, and apply explicit dataset-field overrides.
5. Resolve remaining path references and validate the result with the typed models in `mlpf/conf.py`. Fields still unset receive their typed defaults.

The `--pipeline` flag applies additional CI settings after dataset overrides, including smaller models and selected samples or partitions. The [quickstart scripts](../getting-started/quickstart.md) use this mode for integration checks.

YAML anchors use shallow mapping merges. When a derived recipe supplies an `architecture` or `hyperparameters` mapping, provide the fields needed in that mapping or merge its own anchor explicitly.

## Override a run

Place the specification, recipe, production, and data-root options before the subcommand. For example, after downloading the CLD datasets described in the [training guide](../training/train.md):

```bash
uv run python3 mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cld-v1 \
  --production-name cld \
  --data-dir data/tfds/tensorflow_datasets/cld \
  train \
  --gpus 1 \
  --data_config 1 \
  --num_steps 2000 \
  --gpu_batch_multiplier 4 \
  --lr 0.0001 \
  --model.backbone.num_convs 2
```

Explicit CLI options use their documented spelling, such as `--data-dir`. Extra configuration fields use their Python names, including underscores and dotted paths. Values are parsed as YAML, so `--sort_data false` supplies a boolean and `--data_config '["1", "2"]'` supplies a list. Unknown typed fields and invalid values raise validation errors.

The CLI generates a field/type/default listing from the typed models:

```bash
uv run python3 mlpf/pipeline.py train --help
```

Those defaults describe the typed configuration. The selected recipe and command-line options determine the values for a particular run.

## Select datasets and batches

Recipe sections expand into the following runtime paths:

| Recipe section | Runtime configuration |
|---|---|
| `train_datasets.<group>` | `train_dataset.<dataset>.<group>` |
| `validation_datasets.<group>` | `valid_dataset.<dataset>.<group>` |
| `test_datasets` list | `test_dataset.<sample>` |

For example, `--train_dataset.cld.physical.batch_size 2` changes the CLD training group's base batch size. A sample version can be set with `--train_dataset.cld.physical.samples.cld_edm_ttbar_pf.version 3.3.0`. Keep validation and test versions compatible when changing training data. Detector, group, and sample paths must already exist in the selected recipe.

The sample's `splits` list selects TFDS configuration partitions, such as `"1"` through `"10"`. `--data_config 1,2` filters those lists across training, validation, and testing. Check that the resulting lists contain the intended partitions. `--test-datasets` selects the test sample names to run.

Each training step draws one dataset-group batch. Its per-rank size is the group's `batch_size` multiplied by `gpu_batch_multiplier`. See [dataset mixing](../training/data-mixing.md) for sampling order, group weights, and distributed batch sizes.

## Inspect or extend a recipe

From the repository root, resolve a configuration without loading events or starting training:

```bash
uv run python3 - <<'PY'
from mlpf.conf import MLPFConfig

config = MLPFConfig.from_spec(
    "particleflow_spec.yaml", "pyg-cld-v1", "cld",
    extra_args=[
        "--data_dir", "data/tfds/tensorflow_datasets/cld",
        "--data_config", "1",
        "--gpu_batch_multiplier", "4",
        "--model.backbone.num_convs", "2",
    ],
)
print(config.model_dump_json(indent=2))
PY
```

The output shows resolved paths, model settings, and dataset selections. Successful resolution checks configuration structure; dataset availability is checked when loading data.

For a reusable variant, copy the specification to a new file and add a recipe under `models`. Copy the complete `pyg-cld-v1` mapping as `pyg-cld-small`, set `architecture.attention.num_convs` to `2`, and set `gpu_batch_multiplier` to `4`. Select the new file with `--spec-file` and the new recipe with `--model-name`. The [architecture guide](../concepts/model-architectures.md) explains the model options.

The pipeline logs the final dotted fields and saves the resolved run as `<command>-config.yaml` alongside the loaded `particleflow_spec.yaml`. Keep these with `model_kwargs.pkl` and the checkpoint as described in the [artifact reference](artifacts-and-directories.md). For checkpoint continuation and weights-only loading, follow [resume and fine-tuning](../training/resume-and-fine-tune.md).
