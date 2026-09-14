# Distributed training and hyperparameter search

Use distributed execution after the same model, dataset, and overrides complete on one GPU. It adds process coordination and cluster resource requirements while retaining the same dataset and checkpoint compatibility requirements.

## Choose an execution mode

| Goal | Command | Scope | Status |
|---|---|---|---|
| One GPU | `train --gpus 1` | One process on one machine | Supported |
| Several GPUs on one machine | `train --gpus N` | PyTorch DistributedDataParallel (DDP) | Supported |
| Ray-managed training | `ray-train` | Ray workers, locally or on a cluster | Research workflow |
| Hyperparameter search | `ray-hpo` | Several Ray Tune trials | Research workflow |

`--gpus` gives the number of PyTorch DDP processes for the standard `train` command. For Ray commands, `--ray-gpus` gives the number of GPUs assigned to each training worker or trial. Select the visible CUDA devices through the execution environment.

## Single-node DDP

This example uses four visible GPUs on one machine:

```bash
uv run python3 mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cld-v1 \
  --production-name cld \
  --data-dir /path/to/cld/tfds \
  --prefix cld-ddp- \
  train \
  --gpus 4 \
  --dtype bfloat16 \
  --num_steps 100000
```

Each rank gets its own data-loader process and GPU. The effective event mix depends on dataset-group batch sizes, `gpu_batch_multiplier`, world size, and sampler behavior. Record all of these values alongside the nominal batch size. Use `interleaved-shards` when early mixing across several TFDS shards or domains is required:

```bash
... train --gpus 4 --sampler_mode interleaved-shards
```

All ranks must see the same files and experiment directory. Typical failures are mismatched visible GPUs, NCCL networking, insufficient file-descriptor limits, and one rank running out of memory.

## Ray Train

Run a small local Ray check before connecting to a cluster:

```bash
uv run python3 mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cld-v1 \
  --production-name cld \
  --data-dir data/tfds/tensorflow_datasets/cld \
  --experiments-dir experiments \
  --prefix cld-ray-local- \
  ray-train \
  --data_config 1 \
  --ray-local \
  --ray-cpus 2 \
  --ray-gpus 0 \
  --num_steps 2 \
  --ntrain 10 \
  --nvalid 10 \
  --ntest 10
```

For a GPU worker, use `--ray-gpus 1` and settings appropriate to that GPU. `--ray-local` creates the local Ray environment; cluster execution uses the Ray environment supplied by the site. Ray keeps its own run and checkpoint metadata under the configured experiment storage.

## Ray Tune hyperparameter search

Searches use `mlpf/raytune/search_space.py`. Review that file before starting: a search can launch many full training trials. The selected model's `raytune.local_dir` must be a writable absolute storage path in a custom specification.

After setting that path, a bounded local check is:

```bash
uv run python3 mlpf/pipeline.py \
  --spec-file /path/to/reviewed-spec.yaml \
  --model-name pyg-cld-v1 \
  --production-name cld \
  --data-dir /path/to/cld/tfds \
  ray-hpo \
  --name cld-search \
  --ray-local \
  --ray-cpus 2 \
  --ray-gpus 0 \
  --raytune-num-samples 2 \
  --num_steps 2 \
  --ntrain 10 \
  --nvalid 10
```

An existing compatible Tune directory is resumed automatically, including unfinished and errored trials. Use a new search name when the search space or scientific question changes.

## Reproducibility checklist

Record the repository commit, complete resolved configuration, dataset versions and partitions, checkpoint, visible hardware, world size, worker resources, sampler mode, precision, and random seeds. Compare runs at equal effective data exposure and validation cadence. TensorBoard is written locally; Comet logging is optional through `--comet` or `--comet-offline`.
