# Dataset mixing and loss monitoring

Training can combine several physics samples on one GPU or with distributed execution. Two settings control the mix: the sampler orders events within a dataset group, and a group weight controls how often training draws that group's batch.

## Samples, groups, and batches

A model recipe in `particleflow_spec.yaml` defines `train_datasets.<group>`, each with a `batch_size` and a list of `samples`. A sample identifies a TFDS dataset, version, and configuration partitions. Samples in one group are concatenated; their relative event counts determine their shares within that group.

Each training step draws one batch from one group. The per-rank batch size is the group's `batch_size` multiplied by `gpu_batch_multiplier`; distributed training multiplies the total events per step by the number of ranks. Groups can use different batch sizes to accommodate different event sizes.

## Order events within a group

Set `--sampler_mode` on the training command:

| Mode | Behavior | Use when |
|---|---|---|
| `shard-consecutive` (default) | Reads sample/configuration shards consecutively in an epoch-dependent shuffled order. Distributed ranks read contiguous blocks of that order. | Shards have similar statistics, such as partitions of one sample. |
| `interleaved-shards` | Spreads each shard across the epoch in proportion to its size. | A group combines different processes or pileup conditions. |

Interleaving avoids long stretches dominated by one sample. Individual finite batches need not contain every sample. It changes ordering, while each sample's overall share still follows its event count.

The standard CLD recipe combines `ttbar`, `WW`, and `qq` in one group. Use `--sampler_mode interleaved-shards` for that training path.

## Set the training share of each group

Set a positive `weight` on every training group or leave it unset on all groups. Validation groups cannot have weights. The scheduler spreads group draws throughout training with the following shares:

| Configuration | Share of training steps |
|---|---|
| No explicit weights | Proportional to each group's number of batches per epoch |
| Explicit weights | Group weight divided by the sum of active group weights |

Each group cycles through its own data independently. Empty groups are skipped with a warning; training fails if all groups are empty. With explicit weights, the reported nominal epoch (the total batches in one pass over all groups) need not represent one pass through each group. Inspect per-group passes to measure data exposure.

Weights control batch selection. They do not multiply loss terms or directly specify event fractions. For example, weights 3:1 and per-rank batches 12:1 yield roughly 75%:25% of steps but 36:1 events, before any common batch multiplier. To weight two samples independently, place them in separate groups.

## Run the two-group ColliderML recipe

Download configuration 1 of both ColliderML 1.2.0 track/cluster datasets listed in the [catalog](../datasets/catalog.md), preserving their shared `colliderml` data root. With the [training environment](../getting-started/installation.md) and a compatible GPU, this bounded run sets a 3:1 step mix:

```bash
uv run python3 mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-colliderml-pu0pu200-groups-v1 \
  --production-name colliderml \
  --data-dir data/tfds/tensorflow_datasets/colliderml \
  --prefix colliderml-mix- \
  train \
  --data_config 1 \
  --gpus 1 \
  --dtype bfloat16 \
  --gpu_batch_multiplier 1 \
  --train_dataset.colliderml.pu0.weight 3 \
  --train_dataset.colliderml.pu200.weight 1 \
  --num_steps 2000 \
  --val_freq 1000 \
  --checkpoint_freq 1000
```

The recipe uses batch sizes 12 for `pu0` and 1 for `pu200`. Adjust these for available memory. Success creates an experiment under `experiments/colliderml-mix-*`, logs `train mix:` lines showing shares 0.75 and 0.25, and writes finite per-group validation losses and checkpoints. This checks training behavior; use held-out physics validation to assess reconstruction quality.

The specification uses plural sections (`train_datasets`, `validation_datasets`, `test_datasets`). Resolved configuration and CLI overrides use `train_dataset.<detector>.<group>`, `valid_dataset.<detector>.<group>`, and `test_dataset.<sample>`. Dataset overrides are applied after the recipe builds those sections. Unknown detectors, groups, or sample paths raise an error; add new groups in a custom specification first.

## Monitor exposure and losses

Rank 0 logs each group's step share, batches per epoch, events per step, and expected passes for the requested run length. TensorBoard records `data/<group>/step_fraction`, `passes`, and `events_seen`.

With several groups, losses are reported both jointly and per group. Validation losses are batch averages for each group. Training group losses in TensorBoard, terminal output, and step history are means over that group's steps since the previous logging interval; Comet records each step under its group.

| Output | Per-group loss location |
|---|---|
| TensorBoard train and valid runs | `group/<group>/loss_<loss>` |
| Comet | `<group>/train_loss_<loss>`, `<group>/valid_loss_<loss>` |
| Step history JSON | `train_group`; `group/<group>/<loss>` within `valid` |
| Terminal | Each group's unweighted total |

The unweighted total removes task-loss weighting; it does not undo dataset sampling. Compare per-group validation losses and data exposure when changing a mix. The scenario comparison tool, `scripts/training/compare_scenario.py`, tabulates and plots these losses.

## Resume a weighted run

A full [checkpoint resume](resume-and-fine-tune.md) restores each group's epoch, batch position, and scheduling state. If weights change, sampler positions are retained and the scheduling counters restart so the new shares apply immediately. Keep group identities and ordering compatible with the checkpoint.

Loader state from before weighted mixing can be converted, but subsequent batch order changes and a warning is logged. Use weights-only loading for a fresh training run, or `--sampler_from_scratch true` to reset iteration while retaining the optimizer and training step.
