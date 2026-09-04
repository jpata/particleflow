# Quickstart

The local test scripts are end-to-end software checks. Each script runs the unit tests, downloads two small ROOT files, converts the detector data, builds a TFDS split, and trains an MLPF model for two CPU steps.

:::{warning}
The CLD and CLIC scripts delete `local_test_data/cld` or `local_test_data/clic` before downloading fresh inputs. All three scripts create or update generated directories such as `tensorflow_datasets/`, `experiments/`, and `plots/`. Run them from the repository root and do not store unrelated work in those paths.
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

The scripts run the full unit-test suite, so runtime depends on the machine and an existing dependency/download cache. The first run is slower than later runs.

## What success looks like

For CLD or CLIC, a successful run produces:

- detector-specific data under `local_test_data/`;
- data-validation plots under `plots/`;
- a TFDS dataset under `tensorflow_datasets/`; and
- a new two-step training under `experiments/` with checkpoints and saved configuration.

The CMS script also starts a second experiment from the first checkpoint and writes an ONNX comparison under `onnx_validation_cms/`.

## Run only data preparation

If you do not want to run unit tests and training, fetch and postprocess the two Key4HEP files directly:

```bash
./scripts/fetch_test_data_cld.sh

uv run python3 tests/validate_parquet.py \
  --input local_test_data/cld/p8_ee_ttbar_ecm365/reco_p8_ee_ttbar_ecm365_300000.parquet \
  --detector cld \
  --max-events 20 \
  --plots-dir plots/cld
```

Use `scripts/fetch_test_data_clic.sh` and `--detector clic` for the equivalent CLIC path.

## What this does not prove

A two-step CPU run can catch installation, schema, data-loading, and model-shape failures. It is not long enough to test convergence, particle reconstruction quality, jet resolution, missing momentum, or inference throughput. Those require dedicated validation on statistically useful samples.
