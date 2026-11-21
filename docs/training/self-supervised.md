# Input-only student–teacher pretraining

MLPF has an experimental, single-device self-supervised entry point that trains
the existing input encoder and backbone without particle targets. It supports
shared backbones through `MLPF.encode_backbone`; transformer/HEPTv2 layers and
supervised particle heads are unchanged. Split classification/regression
backbones are rejected because that interface exposes only their ID branch.

This is inspired by [Panda](https://arxiv.org/abs/2512.01324) and
[Panda Diplomacy](https://arxiv.org/abs/2609.00611), not a reproduction of their
architecture or full augmentation recipe.

## Objective

The teacher sees the complete event. Each student view independently keeps a
random subset of elements, stratified by observed element type, retaining at
least one element of each type. Dropped elements are physically removed before
feature engineering and backbone execution. Original row indices establish
teacher/student correspondence; MC particle ownership is never used.

A normalized projection head predicts soft assignments over learned prototypes.
The teacher targets use a moving center and a lower softmax temperature. Cross
entropy is averaged over elements within each type, types within each event,
events within the batch, and finally student views. The EMA teacher is updated
after each optimizer step and always runs in evaluation mode without gradients.
Prototype similarities, softmax, loss, and centering use FP32 even with BF16
backbone execution. The PF heads remain frozen during SSL.

Centering/sharpening supplies collapse prevention. Monitor `teacher_entropy` and
`effective_prototypes` alongside loss: low loss alone does not establish useful
representations, and broad prototype usage alone does not rule out trivial
type/geometry encodings. This version has no masked-location prediction,
geometric augmentation, or particle-slot objective.

## Local CLD run

From the repository root, using the existing `.venv` and local GPU:

```bash
bash scripts/local/pretrain.sh --steps 100 --output experiments/cld_ssl
```

The default recipe is `configs/training/cld_ssl.yaml`. It uses the existing
`pyg-cld-v1` track/cluster transformer with ttbar, fully hadronic WW, and qq
configuration `1:3.3.0` from `/mnt/work/mlpf/tensorflow_datasets/cld`, batch size
768, two student views, 60% element retention, and 256 prototypes. The local
TensorFlow dataset path matches `configs/training/platforms/local.yaml`.

The batch size is measured for the local RTX 5060 Ti (16 GB), BF16 CLD
track/cluster transformer. Repeating the largest event (310 elements) from all
12,288 configured training events 768 times used 10.51 GiB reserved / 9.27 GiB
allocated memory in three full student/teacher training steps, including Adam
state. Batch 1024 used 13.96 GiB reserved, so 768 leaves more headroom for other
processes and diagnostics. See `configs/training/references/cld_ssl_local_batch.json`
for all measurements and the exact configuration. This bound applies to this
dataset pool and architecture, not larger raw-hit collections. The learning rate
is unchanged; increasing batch size increases event exposure per training step.

Override the batch size with `--batch-size N`. An older run created with batch
size 4 must use `--batch-size 4 --resume`; batch changes are rejected on resume
to preserve the sampling/optimization contract. Benchmark another recipe with:

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 .venv/bin/python scripts/local/benchmark_ssl_batch.py \
  --config configs/training/cld_ssl.yaml --scan-per-dataset 4096 \
  --batches 256 512 768 1024 --output experiments/ssl_batch_benchmark.json
```

Training draws with replacement from up to 4096 events per sample in
`train[:95%]`; validation uses up to 32 events per sample in `train[95%:]`.
TFDS resolves the slices and the runner rejects overlap. The official test
split is untouched. Validation uses fixed view-sampling seeds. The loader
requires only `X` and dataset identity, and returns no target or candidate fields.
Existing serialized records may still contain labels, which are ignored.

Resume to a final step count:

```bash
bash scripts/local/pretrain.sh --steps 2000 --output experiments/cld_ssl --resume
```

An existing output directory cannot be overwritten without `--resume`. Resuming
restores student, teacher, projection heads, center, optimizer, and Torch RNG
states; event sampling is determined by seed and step. Configuration changes
other than step limit, logging/checkpoint frequencies, loader worker/prefetch settings,
timing frequency, and isolated probe/snapshot settings
are rejected. Probes can therefore be enabled when resuming an older SSL run.

Outputs:

- `pretrain-config.yaml`: resolved pretraining settings and dataset selection.
- `train-config.yaml`: resolved MLPF architecture for downstream use.
- `metrics.jsonl`: per-step losses, gradient norms, and prototype diagnostics,
  with held-out validation at the configured interval and final step.
- `latest.pth`: complete SSL resume checkpoint.
- `student.pth`: standard MLPF weights for supervised fine-tuning. These include
  the original, untrained PF heads; the SSL projection head is not exported.
- `embeddings/step-N.npz`: fixed-cohort student/teacher embeddings and input rows.
- `probes/step-N-predictions.npz`: bounded held-out probe predictions and targets.

For downstream training use `student.pth` with `--load_weights_only true`, not
as a full supervised-training resume. Match the exported MLPF architecture:

```bash
PF_SITE=local uv run python -m mlpf.pipeline \
  --spec-file particleflow_spec.yaml --model-name pyg-cld-v1 \
  --production-name cld train \
  --data_dir /mnt/work/mlpf/tensorflow_datasets/cld --data_config 1 \
  --model.backbone.mode shared --model.attention.use_flash_attn_varlen false \
  --load experiments/cld_ssl/student.pth --load_weights_only true
```

Set an appropriate supervised batch size and training budget using the ordinary
[training guide](train.md). Pretraining alone does not train particle reconstruction.

## Other representations

Copy the YAML and change `model_name`, `production`, `datasets`, `data_dir`, and
`model_overrides`, then run:

```bash
CONFIG=/path/to/pretrain.yaml bash scripts/local/pretrain.sh \
  --output experiments/my_ssl
```

The SSL wrapper uses only element type for stratification and the model's own
encoder for all feature interpretation. Tracks/clusters, tracks/calorimeter
hits, and raw hits therefore use their respective existing MLPF model recipes
and schemas. This is not automatic conversion between feature schemas or a
joint model for incompatible representations. Keep normalization statistics and
dataset versions consistent when changing recipes.

For HEPTv2 track/cluster experiments, override `conv_type: heptv2` and
`model.type: heptv2` with the desired existing HEPTv2 settings. For raw hits,
select the corresponding hit model and hit datasets, and reduce batch size to
fit the additional teacher and student passes. The head width is inferred from
the encoder output, including concatenated representations.

The implementation is single-device with bulk input-only decoding and prefetched
worker batches; it does not yet integrate DDP, Ray, or the supervised scenario launcher. CPU execution is
available with `--device cpu`, `dtype: float32`, and a model configuration using
math attention without jagged/Flash-only execution.

## Throughput and phase timings

The local recipe uses `num_workers: 4`, `prefetch_factor: 2`, pinned host
batches, and nonblocking device transfers. ArrayRecord reads are grouped by
dataset/shard and the TFDS parser decodes only `X`. Workers prepare whole batches
using a seed derived from the training step, retaining request order and duplicate
events. Worker startup and prefetching do not consume the model's RNG stream.
Set `num_workers: 0` to debug without multiprocessing.

View sampling and event/type-balanced reductions are vectorized across the
batch. Sampling still keeps `max(1, floor(fraction * count))` uniformly selected
elements of each observed type, without replacement. The reduction is the same
equal-event/equal-type objective, with value and gradient parity tests. CUDA
training also uses fused AdamW and batched EMA updates. The new sampler consumes
random numbers differently from the original per-event implementation: older
checkpoints remain loadable, but switching implementations does not reproduce
the original future random views bit for bit. Continuation within the optimized
implementation retains deterministic event sampling and RNG restoration.

Every JSONL training record includes `performance.training_seconds`,
`events_per_second`, and `step_seconds`. Training time includes waiting for data,
transfer, forward/backward, optimizer and EMA; step time also includes scheduled
diagnostics/checkpoints. Detailed `timings` are recorded at the first step and
every `timing_frequency` steps (10 by default):

- `data_wait`, `transfer`, `forward`, `backward`, `optimizer`, `ema`;
- within forward: `teacher`, `view_sampling`, `student`, `loss`, `statistics`;
- scheduled `validation`, `probe`, `snapshot`, and `checkpoint` work.

`cpu_wall_seconds` measures host wall time. `cuda_stream_seconds` uses CUDA
events and a single synchronization after the complete timed step, avoiding
barriers between phases. These are **stream elapsed times**, including idle gaps
between enqueued operations, not a sum of GPU kernel durations. Forward subphases
are nested inside `forward`, so do not sum them twice. Worker startup appears in
the first data-wait measurement; use later steps for steady-state comparisons.

On the RTX 5060 Ti at batch 768, a 200-step CLD real-data benchmark measured
0.270 s median / 0.267 s mean training time over steps 10–199 and 84.3% average
GPU utilization during sampling over steps 108–192. Median prefetched data wait
was 0.264 ms. GPU-stream phase medians included 54 ms teacher, 85 ms student
passes, 3.5 ms sampling, 6 ms loss, and 110 ms backward. The earlier live run
averaged 17.7% GPU utilization and roughly 5 s/step; this is an observed before/after
comparison, not a paired kernel benchmark. Probes/snapshots were disabled for
this throughput measurement and remain enabled in the normal local recipe.
Full evidence is in `configs/training/references/cld_ssl_optimized_performance.json`
and locally in `experiments/cld_ssl_optimized_timing_v1`.

## Periodic supervised MLPF probe

The CLD YAML enables a supervised task probe at SSL step 0, every 500 steps,
and the final step. The general `ProbeConfig` default is disabled, so input-only
pretraining on unlabeled records remains possible. Configure it with:

```yaml
probe:
  enabled: true
  frequency: 500
  at_start: true
  steps: 200
  batch_size: 8
  train_events_per_dataset: 64
  valid_events_per_dataset: 32
  lr: 0.001
  seed: 31415
```

Each probe extracts the current **student** embeddings once, in evaluation mode,
and caches them detached on CPU. It then trains a separate copy of the existing
MLPF readout using simulation particle targets and the standard supervised loss.
The input encoder and backbone stay frozen. This is a nonlinear PF task probe,
not a linear probe; existing task-query readouts or the particle-set decoder are
included when configured. Set-mode probes use the existing Hungarian matching
and configured auxiliary objectives.

Each probe starts from the identical untrained PF heads retained in the SSL
student. It uses the same event subsets, sampling seed, optimizer, and update
budget, without warm-starting from the previous probe. The training subset comes
from the SSL training slice and the evaluation subset from its disjoint validation
slice; the official test split remains untouched. Labels are loaded only by the
separate probe loader. No supervised gradient, calibrated task weight, optimizer
state, or RNG change is carried back into SSL. Probe train and evaluation subsets
are loaded from the configured datasets and therefore require particle labels.

The JSONL log includes a `probe` result at each probe step. Additional artifacts
are `probes/step-N.json` and `probes/step-N.pth`: the latter contains the frozen
encoder and trained PF readout in the standard model checkpoint format. The main
`student.pth` export remains purely SSL-trained with untrained PF heads.

Metrics include initial and final validation loss components and, for elementwise
models, input-aligned presence precision/recall/F1 and PID accuracy conditional
on a target-bearing input. These are not geometrically matched particle metrics
or jet-performance measurements. `unweighted_task_loss` sums the seven base PF
task terms with fixed coefficients. Prefer that sum and its individual components
for comparisons: `losses.Total` uses the standard task-weight calibration, whose
scale changes during training and may differ between probes. Set-mode probes
currently report losses rather than matched-particle physics metrics.

The local GPU probe at SSL step 102 trained for 200 updates on 192 events and
evaluated on 96 events. Presence F1 increased from 0.634 to 0.747, and conditional
PID accuracy from 0.107 to 0.881. Its artifacts are in
`experiments/cld_ssl_smoke_v2/probes/step-102.*`. This verifies readout learning;
measuring the benefit of SSL requires comparison with the step-0 probe and later
checkpoints under the same probe settings.

## Verification and interpretation

### Embedding and probe history

The local recipe saves input-only embedding snapshots initially, every 100
steps, and at the final step. Enable them in other recipes with:

```yaml
snapshots:
  enabled: true
  frequency: 100
  events_per_dataset: 4
  elements_per_event: 256
  teacher: true
```

The same first four held-out events per dataset are encoded with their complete
context. At most 256 rows per event are selected deterministically for saving;
this cap does not remove context from the encoder. Archives contain FP32
`student` and `teacher` embeddings, `X`, `dataset_index`, `event_index`, and
`element_index`. Dataset names and the TFDS split are recorded in
`embeddings/cohort.json`; event indices are relative to that slice and element
indices refer to original input rows. A hash guards against silently changing
the cohort on resume. Snapshots do not read particle labels or alter SSL state
or Torch RNG streams. If enabled on an older run, the first snapshot records
the resumed checkpoint step rather than implying a step-0 baseline.

Probe metrics and full probe checkpoints remain saved every 500 steps. In
addition, `probe.save_events_per_dataset: 4` saves raw presence/PID logits,
momentum outputs, targets, and inputs for a small fixed validation cohort.
The NPZ uses separate input/prediction/target offsets so set-mode axes are
unambiguous. Momentum is in the model's **training coordinates**, not decoded
physical four-vectors; consult the saved model configuration when converting.
Set `save_events_per_dataset: 0` to disable prediction archives.

Generate or refresh plots without loading data or using the GPU:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/local/plot_ssl_evolution.py \
  experiments/cld_ssl
```

This writes `evolution/embeddings.png`, `embedding_drift.png`, `probes.png`, and
`embedding_history.json`. The embedding panels use the first snapshot's PCA
basis after centering and orthogonal Procrustes alignment, to avoid confusing
arbitrary rotations with changes in structure. Drift metrics include linear
CKA (rotation invariant), centered cosine similarity before alignment, and
mean feature standard deviation. Colors show input element type, not inferred
particle identity. Probe curves include both the initial and trained readout
at each checkpoint; useful representation learning must be judged against
the step-0 trained probe, not just an untrained head.

A two-step GPU integration run at batch 768 saved three embedding snapshots
and three probe prediction/checkpoint sets under
`experiments/cld_ssl_evolution_smoke`. It used five-update probes on small
subsets to test scheduling and artifacts, not to measure physics performance.

Tests cover absent/ignored labels, correspondence and padding, event/type loss
balancing, frozen teacher and PF heads, EMA updates, BF16 statistics, transformer
inputs from CLD/CMS/raw-hit schemas, HEPTv2 backward propagation, strict checkpoint
transfer, and exact CPU continuation against uninterrupted training.

The local CLD BF16 GPU smoke run completed 20 steps, then resumed to 100 steps,
with validation on 96 events. Validation loss was 3.120 at step 20 and 1.944 at
step 100; effective prototype count was 97.3/256 and 48.6/256 respectively.
Artifacts are under `experiments/cld_ssl_smoke_v2` in the local workspace.
Targets evolve during SSL, so decreasing loss is not a fixed-target quality
comparison. This verifies execution and basic numerical behavior, not reconstruction gains.
Assess benefit with frozen probes and supervised fine-tuning at matched label
fractions and compute budgets, using independent final evaluation events.
