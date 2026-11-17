# Quickstart

The local test scripts are end-to-end software checks. The CMS, CLD, and CLIC scripts download small ROOT samples, convert the detector data, build a TFDS split, and train an MLPF model for two CPU steps. Unit tests run separately in CI.

:::{warning}
The CLD and CLIC scripts delete `local_test_data/cld` or `local_test_data/clic` before downloading fresh inputs. All three scripts create or update generated directories such as `tensorflow_datasets/`, `experiments/`, and `plots/`. Run them from the repository root and keep unrelated work outside those paths.
:::

## Run one detector path

After [installing the main environment](installation.md), choose one:

```bash
# CLD: EDM4hep postprocessing, parquet checks, TFDS, and CPU training
uv run ./scripts/local_test_cld.sh

# CLIC: EDM4hep postprocessing, parquet checks, TFDS, and CPU training
uv run ./scripts/local_test_clic.sh

# CMS: postprocessing, TFDS, CPU training, checkpoint loading, and ONNX check
uv run ./scripts/local_test_cms.sh
```

Runtime depends on the machine and dependency/download caches. To run the unit suite separately:

```bash
PYTHONPATH="$PWD" KERAS_BACKEND=torch uv run python3 -m pytest tests
```

The CI detector matrix also runs `bash scripts/local_test_colliderml.sh`, which creates an offline synthetic fixture, converts it, builds TFDS, and trains on CPU. Its validator runs in report mode, so inspect its findings separately from the script exit status. For real ColliderML data and the local MAIA smoke test, see the [detector-comparison runbook](../validation/detector-comparison.md).

## What success looks like

For CLD or CLIC, a successful run produces:

- detector-specific data under `local_test_data/`;
- data-validation plots under `plots/`;
- a TFDS dataset under `tensorflow_datasets/`; and
- a new two-step training under `experiments/` with checkpoints and saved configuration.

The CMS script also starts a second experiment from the first checkpoint and writes an ONNX comparison under `onnx_validation_cms/`.

## Run only data preparation

To focus on data preparation, fetch and postprocess the two Key4HEP files directly:

```bash
./scripts/fetch_test_data_cld.sh

uv run python3 tests/validate_parquet.py \
  --input local_test_data/cld/p8_ee_ttbar_ecm365/reco_p8_ee_ttbar_ecm365_300000.parquet \
  --detector cld \
  --max-events 20 \
  --plots-dir plots/cld
```

Use `scripts/fetch_test_data_clic.sh` and `--detector clic` for the equivalent CLIC path.

## Scope of this check

A two-step CPU run checks installation, schema, data loading, and model shapes. [Dedicated validation](../validation/overview.md) on statistically useful samples establishes convergence, particle reconstruction quality, jet resolution, missing momentum, and inference throughput.
