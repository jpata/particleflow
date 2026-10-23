# Continue training and fine-tune

`--load` reads a checkpoint and restores its model state. With a compatible configuration, MLPF also restores the optimizer, learning-rate schedule, random-number state, and train and validation loader positions. This supports two related workflows:

- **Continue** an interrupted run in the same experiment directory.
- **Fine-tune** from existing weights in a new experiment directory, usually with different data or a deliberate training change.

For a full resume, `--num_steps` gives the final step number. Loading `checkpoint-1000.pth` with `--num_steps 2000` therefore adds 1,000 updates by running steps 1001 through 2000. With `--load_weights_only true`, the step counter starts fresh at 1, so `--num_steps 2000` requests 2,000 new updates.

## Before loading a checkpoint

Keep the checkpoint with its `model_kwargs.pkl`, saved specification, and resolved training configuration. Compatibility requires matching input features, particle classes, architecture shapes, and dataset schema in addition to the detector name.

The examples below assume the CLD configuration-1 data from [Train a model](train.md). Set the checkpoint path explicitly:

```bash
export MLPF_CHECKPOINT=experiments/cld-ttbar-config1-EXAMPLE/checkpoints/checkpoint-1000.pth
```

Replace `EXAMPLE` with the actual experiment directory name.

## Continue in the same directory

Pass both the original `--experiment-dir` and its checkpoint:

```bash
uv run python3 mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cld-v1 \
  --production-name cld \
  --data-dir data/tfds/tensorflow_datasets/cld \
  --experiment-dir experiments/cld-ttbar-config1-EXAMPLE \
  train \
  --data_config 1 \
  --load "$MLPF_CHECKPOINT" \
  --gpus 1 \
  --dtype bfloat16 \
  --gpu_batch_multiplier 4 \
  --num_steps 2000 \
  --val_freq 1000 \
  --checkpoint_freq 1000
```

This preserves the experiment location and, by default, resumes the saved loader positions. Back up a completed experiment before reusing its directory: configuration and log files in that directory are updated by the new invocation.

## Start a new fine-tuning experiment

Omit `--experiment-dir` and give a new prefix:

```bash
uv run python3 mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cld-v1 \
  --production-name cld \
  --data-dir data/tfds/tensorflow_datasets/cld \
  --prefix cld-finetune- \
  train \
  --data_config 1 \
  --load "$MLPF_CHECKPOINT" \
  --load_weights_only true \
  --gpus 1 \
  --dtype bfloat16 \
  --gpu_batch_multiplier 4 \
  --num_steps 2000 \
  --val_freq 1000 \
  --checkpoint_freq 1000
```

`--load_weights_only true` restores model weights while starting a fresh optimizer, learning-rate schedule, random-number stream, loader state, and step count. The new recipe controls the learning rate and training duration.

For a full resume that only restarts data iteration, use `--sampler_from_scratch true` instead. That option retains the checkpoint step, optimizer, and scheduler. A full resume also retains the saved learning-rate schedule; a changed configured learning rate is ignored with a warning. Simply extending `--num_steps` does not create a fresh schedule.

## Strict and relaxed loading

Loading is strict by default. MLPF checks tensor shapes and normally restores optimizer state. A shape mismatch remains an error. `--relaxed_load true` permits missing model keys and skips optimizer restoration. The user remains responsible for establishing input and output compatibility.

Use relaxed loading only after reviewing the reported missing and unexpected keys:

```bash
... train --load "$MLPF_CHECKPOINT" --relaxed_load true
```

Treat a checkpoint/configuration mismatch as an explicit model change. Record the old checkpoint, old configuration, new configuration, missing keys, and scientific reason for the change.

## Choosing trainable modules

The model has an advanced `model.trainable` setting for selecting named submodules. The current maintained detector recipes train `all` parameters, which is the supported default. Selective freezing depends on the exact architecture and belongs in a reviewed, model-specific recipe. Check the parameter table printed at startup to confirm the selection.

For cross-detector work, begin with the methodology and archived configuration associated with the [2025 fine-tuning study](../science/publications.md). Reproduction uses that study's archived configuration, data, and checkpoint.

## Success checks

The log should state the loaded checkpoint and the restored starting step. Confirm that:

- a full resume starts after the checkpoint step, while weights-only fine-tuning starts at step 1;
- the printed trainable and non-trainable parameter counts match the intended experiment;
- new validation history and checkpoints appear in the selected experiment directory; and
- validation loss is finite before interpreting longer-term trends.
