# Download a dataset

This guide downloads configuration 1 of the three CLD track-and-cluster datasets used by the standard model recipe and verifies that TensorFlow Datasets (TFDS) can open them. Choose another compatible selection from the [dataset catalog](catalog.md).

## Prerequisites

Complete the [main environment installation](../getting-started/installation.md). The commands below run from the repository root and use the Hugging Face CLI supplied by that environment. Public files support anonymous downloads.

Choose a destination with enough free space. TFDS shards can be large, especially for hit inputs. The Hub dry run below currently reports 80 files totalling approximately 2.9 GB for this example. Treat that as a point-in-time value: the CLI's own report is authoritative for the revision you download.

## Download the three training datasets

First preview the exact selection:

```bash
uv run hf download jpata/particleflow \
  --include "tensorflow_datasets/cld/cld_edm_ttbar_pf/1/3.2.1/*" \
  --include "tensorflow_datasets/cld/cld_edm_ww_fullhad_pf/1/3.2.1/*" \
  --include "tensorflow_datasets/cld/cld_edm_qq_pf/1/3.2.1/*" \
  --local-dir data/tfds \
  --repo-type dataset \
  --dry-run
```

Remove `--dry-run` to download it:

```bash
uv run hf download jpata/particleflow \
  --include "tensorflow_datasets/cld/cld_edm_ttbar_pf/1/3.2.1/*" \
  --include "tensorflow_datasets/cld/cld_edm_ww_fullhad_pf/1/3.2.1/*" \
  --include "tensorflow_datasets/cld/cld_edm_qq_pf/1/3.2.1/*" \
  --local-dir data/tfds \
  --repo-type dataset
```

The command preserves the Hub layout. The resulting TFDS data root is:

```text
data/tfds/tensorflow_datasets/cld
```

Use this detector-level directory as `--data-dir`; it is the directory that directly contains dataset-name directories.

## Verify metadata and one event

Open each exact dataset/configuration/version and print its registered splits:

```bash
uv run python - <<'PY'
import tensorflow_datasets as tfds

for name in ("cld_edm_ttbar_pf", "cld_edm_ww_fullhad_pf", "cld_edm_qq_pf"):
    builder = tfds.builder(
        f"{name}/1:3.2.1",
        data_dir="data/tfds/tensorflow_datasets/cld",
    )
    print(builder.info.full_name, builder.info.splits)
PY
```

Success prints the full names of all three datasets and metadata for their `train` and `test` splits. The published `ttbar` metadata currently records 90,000 training and 10,000 test events in configuration 1. Then read one `ttbar` event through the same random-access path used by MLPF:

```bash
uv run python - <<'PY'
import tensorflow_datasets as tfds

builder = tfds.builder(
    "cld_edm_ttbar_pf/1:3.2.1",
    data_dir="data/tfds/tensorflow_datasets/cld",
)
event = builder.as_data_source(split="train")[0]
print({name: getattr(value, "shape", None) for name, value in event.items()})
PY
```

Success prints shapes for `X`, `ytarget`, `ycand`, `genmet`, `genjets`, and `targetjets`. This check covers file discovery and decoding. Dataset validation and the detector-specific physics workflow establish data and physics quality.

## Download a different selection

Change all four coupled fields together. A training recipe may require several dataset names, as the standard CLD recipe does:

1. Hub detector directory, such as `cld` or `clic`;
2. dataset name, such as `clic_edm_qq_pf`;
3. configuration partition, currently `1` on the public Hub;
4. version expected by the selected model recipe.

Keep separate detector roots if downloading more than one detector:

```text
data/tfds/tensorflow_datasets/
├── cld/
└── clic/
```

Pass one of those detector directories to MLPF as `--data-dir`. The successful event read above confirms that the downloaded dataset is ready for the [short CLD training example](../training/train.md) or evaluation.

## Common failures

`Dataset ... not found`
: Check the `--data-dir` level and confirm that the dataset/configuration/version directory exists. Keep the required version explicit.

Missing configuration or version on the Hub
: Publication availability can lag behind the recipe. Inspect the [live repository tree](https://huggingface.co/datasets/jpata/particleflow/tree/main/tensorflow_datasets), select an explicitly compatible version, or [produce the dataset locally](generate.md).

Download much larger than expected
: Stop the command and tighten `--include`. A wildcard such as `cld_edm_*` selects several datasets, and omitting the version selects all published versions.

Authentication or rate-limit error
: Public downloads support anonymous access. If the Hub asks for authentication, run `uv run hf auth login` and retry. Store the token and cache outside version control.
