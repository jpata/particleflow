# Evaluate and validate CLD or CLIC

The Key4HEP path reads an EDM4hep ROOT file, reconstructs particles with a compatible MLPF checkpoint, writes a Parquet summary, and produces particle- and jet-level plots. It supports `cld`, `clic`, `cld_hits`, and `clic_hits`.

This page begins with one published CLD model and goes directly to inference. The later workflow scales the same evaluator to several files.

## Published-model example

Complete the [main installation](../getting-started/installation.md), then run from the repository root.

Download the checkpoint and its saved configuration:

```bash
uv run hf download jpata/particleflow \
  --include "cld/clusters/v3.1.0/pyg-cld-v1_cld_20260328_101206_533260/*" \
  --local-dir models \
  --repo-type model
```

Download one CLD EDM4hep event file using the maintained helper:

```bash
./scripts/fetch_test_data_cld.sh
```

Run inference on one event first:

```bash
uv run python3 mlpf/standalone_eval/key4hep/evaluator.py \
  --input local_test_data/cld/p8_ee_ttbar_ecm365/root/reco_p8_ee_ttbar_ecm365_300000.root \
  --checkpoint models/cld/clusters/v3.1.0/pyg-cld-v1_cld_20260328_101206_533260/checkpoints/best_weights.pth \
  --config models/cld/clusters/v3.1.0/pyg-cld-v1_cld_20260328_101206_533260/model_kwargs.pkl \
  --detector cld \
  --num-events 1 \
  --device cpu \
  --dtype float32 \
  --outpath eval_results.parquet
```

If `--config` is omitted, the evaluator searches for `model_kwargs.pkl` two directories above the checkpoint. Passing it explicitly makes the checkpoint/configuration pairing visible.

Success prints the number of processed events and creates `eval_results.parquet`. This one-event check establishes model/input compatibility and completed inference. A statistically useful sample establishes performance.

## Make diagnostic plots

```bash
uv run python3 mlpf/standalone_eval/key4hep/plots.py \
  --input eval_results.parquet \
  --outdir eval_plots \
  --detector cld
```

The evaluator output contains event-level input, target, baseline-candidate, and MLPF-prediction quantities needed by the plotting code. The plots cover particle kinematics and identity, multiplicities, summed momentum, jet response, and related comparisons. On a one-event check, inspect them only for missing or malformed content.

## Use a checkpoint you trained

Keep `checkpoint-*.pth` and `model_kwargs.pkl` from the same experiment. Change the checkpoint and config paths together. The detector mode and input representation must also match:

| Model data | Evaluator detector |
|---|---|
| CLD tracks and clusters | `cld` |
| CLIC tracks and clusters | `clic` |
| CLD hits | `cld_hits` |
| CLIC hits | `clic_hits` |

Pair hit checkpoints with hit inputs and track-and-cluster checkpoints with track-and-cluster inputs, including within the same detector.

## Run held-out TFDS evaluation

Use the main pipeline's `test` command when the input is already TFDS. It writes predictions and the standard particle/jet plots into the chosen experiment directory:

```bash
uv run python3 mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cld-v1 \
  --production-name cld \
  --data-dir data/tfds/tensorflow_datasets/cld \
  --experiment-dir experiments/cld-heldout-evaluation \
  test \
  --data_config 1 \
  --load /path/to/checkpoint.pth \
  --test-datasets cld_edm_ttbar_pf \
  --gpus 1 \
  --dtype bfloat16 \
  --ntest 1000 \
  --make-plots
```

Use a held-out split and a sample size appropriate to the tails and categories being studied.

## Scale out with Snakemake

For a production workspace, copy `validation_key4hep.yaml`, replace its checkpoint and scenario settings, and review `num_files`, detector mode, resources, sample paths, and output location. Preview the generated jobs before submission:

```bash
uv run python3 mlpf/snakemake/produce_validation_snakemake.py --scenario cld
snakemake --snakefile "$(cat .last_jobs_dir)/Snakefile" --dry-run
```

On a configured Pixi site, the maintained entry point is:

```bash
PROD=cld pixi run validation_key4hep --dry-run
PROD=cld pixi run validation_key4hep
```

Outputs are stored below the experiment's `validation/` directory: one evaluator Parquet per ROOT file, combined plots and `metrics.json` per sample, plus an ONNX summary. Dry-run output is the authoritative preview of files and jobs for the selected site.

## Interpret comparisons carefully

`ytarget` is the reconstructable training target, while `ycand` is the detector's baseline particle-flow reconstruction. They answer different questions. Inspect particle-level behavior before jets, report matching and fiducial selections, and state whether a plot compares MLPF to the target, baseline, or both. Current target-jet comparisons have known methodological development tracked in [issue #370](https://github.com/jpata/particleflow/issues/370).
