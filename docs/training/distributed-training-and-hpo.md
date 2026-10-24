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

Each training step draws one batch from one dataset group (`train_datasets.<group>`). An optional `weight` per group sets its share of the steps, spread evenly over training; every group cycles through its own data independently. Set it on every train group or on none. Without weights, each group's share is its number of batches per epoch, so one nominal epoch is one pass over every group.

Weights act on groups (the keys directly under `train_datasets`), not on the samples inside a group. The samples of one group are concatenated, and the group's sampler (`sampler_mode`) decides how they mix: with the default `shard-consecutive`, each GPU reads one sample's shard at a time, so its batches mostly come from a single sample; with `interleaved-shards`, every batch mixes the group's samples (more precisely its shards, one per sample and TFDS config) in proportion to their sizes, steadily over the whole epoch with no tail of only the largest ones. Either way each sample's share follows its number of events. To weight samples against each other, put each in its own group. For example, weighting a no-pileup group 3:1 against a pileup group:

```yaml
train_datasets:
  physical_nopu:
    batch_size: 16
    weight: 3
    samples: [...]
  physical_pu:
    batch_size: 1
    weight: 1
    samples: [...]
```

At startup rank 0 logs one `train mix:` line per group with its step share, batches per epoch and expected passes over its data in `num_steps`. TensorBoard tracks `data/<group>/{step_fraction,passes,events_seen}` during training.

Dataset sections accept dot-notation overrides like every other key, e.g. `--train_dataset.colliderml.pu200.weight 3` or `--test_dataset.clic_edm_ttbar_pf.version 2.5.0`. With several groups, the losses are also reported per group, next to the combined ones. Validation: each group's batch average. Training: every step trains on one group's batch, so the per-step loss jumps between the groups' loss levels. Comet logs every step under its group; TensorBoard, the terminal and the step history, which only see every `tensorboard_step_freq`-th step, log each group's mean over its steps since the last logging step. TensorBoard logs them as `group/<group>/loss_<loss>` in both the train and valid runs, Comet as `<group>/train_loss_<loss>` and `<group>/valid_loss_<loss>` (one Comet panel group per dataset group), the step history as `train_group` and `group/<group>/<loss>` in `valid`, and the terminal prints each group's unweighted total.

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
