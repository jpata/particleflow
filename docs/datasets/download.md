# Download a dataset

This guide downloads one published CLD dataset configuration and verifies that TensorFlow Datasets (TFDS) can open it. Choose another compatible name and version from the [dataset catalog](catalog.md).

## Prerequisites

Complete the [main environment installation](../getting-started/installation.md). The commands below run from the repository root and use the Hugging Face CLI supplied by that environment. No Hugging Face account is required for public files.

Choose a destination with enough free space. TFDS shards can be large, especially for hit inputs. The Hub dry run below currently reports 39 files totalling approximately 1.3 GB for this example. Treat that as a point-in-time value: the CLI's own report is authoritative for the revision you download.

## Download one configuration

First preview the exact selection:

```bash
uv run hf download jpata/particleflow \
  --include "tensorflow_datasets/cld/cld_edm_ttbar_pf/1/3.2.1/*" \
  --local-dir data/tfds \
  --repo-type dataset \
  --dry-run
```

Remove `--dry-run` to download it:

```bash
uv run hf download jpata/particleflow \
  --include "tensorflow_datasets/cld/cld_edm_ttbar_pf/1/3.2.1/*" \
  --local-dir data/tfds \
  --repo-type dataset
```

The command preserves the Hub layout. The resulting TFDS data root is:

```text
data/tfds/tensorflow_datasets/cld
```

Do not pass `data/tfds`, the dataset directory itself, or the version directory as `--data-dir`.

## Verify metadata and one event

Open the exact dataset/configuration/version and print its registered splits:

```bash
uv run python -c 'import tensorflow_datasets as tfds; b = tfds.builder("cld_edm_ttbar_pf/1:3.2.1", data_dir="data/tfds/tensorflow_datasets/cld"); print(b.info.full_name); print(b.info.splits)'
```

Success prints `cld_edm_ttbar_pf/1/3.2.1` and metadata for the `train` and `test` splits. The published metadata currently records 90,000 training and 10,000 test events in configuration 1. Then read one event through the same random-access path used by MLPF:

```bash
uv run python -c 'import tensorflow_datasets as tfds; b = tfds.builder("cld_edm_ttbar_pf/1:3.2.1", data_dir="data/tfds/tensorflow_datasets/cld"); e = b.as_data_source(split="train")[0]; print({k: getattr(v, "shape", None) for k, v in e.items()})'
```

Success prints shapes for `X`, `ytarget`, `ycand`, `genmet`, `genjets`, and `targetjets`. This checks file discovery and decoding, not the dataset's physics quality.

## Download a different selection

Change all four coupled fields together:

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

The MLPF training command receives one of those detector directories as `--data-dir`. The future training guide will cover model selection and command-line overrides; this page stops after proving that the data can be read.

## Common failures

`Dataset ... not found`
: Check the `--data-dir` level and confirm that the dataset/configuration/version directory exists. Do not silently fall back to another version.

Missing configuration or version on the Hub
: The recipe may describe data that has not been published. Inspect the [live repository tree](https://huggingface.co/datasets/jpata/particleflow/tree/main/tensorflow_datasets), select an explicitly compatible version, or [produce the dataset locally](generate.md).

Download much larger than expected
: Stop the command and tighten `--include`. A wildcard such as `cld_edm_*` selects several datasets, and omitting the version selects all published versions.

Authentication or rate-limit error
: Public downloads normally need no token. If the Hub asks for authentication, run `uv run hf auth login` and retry; do not commit the token or cache.
