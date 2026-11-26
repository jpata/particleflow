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
also report `particle_metrics`, `particle_f1`, and `matched_pid_accuracy`, using
the ordinary validation matcher in physical coordinates and the configured
presence threshold. Matching accepts delta-R below 0.1 and relative pT error
below 0.5. Undefined conditional metrics are null.

The local GPU probe at SSL step 102 trained for 200 updates on 192 events and
evaluated on 96 events. Presence F1 increased from 0.634 to 0.747, and conditional
PID accuracy from 0.107 to 0.881. Its artifacts are in
`experiments/cld_ssl_smoke_v2/probes/step-102.*`. This verifies readout learning;
measuring the benefit of SSL requires comparison with the step-0 probe and later
checkpoints under the same probe settings.

### Matched probe learning curves

To distinguish a short readout-training budget from representation quality, run
longer frozen probes at selected saved SSL checkpoints:

```bash
PF_SITE=local OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m mlpf.probe_learning_curves \
  --run experiments/cld_ssl_fast \
  --ssl-steps 0 500 5000 --seeds 31415 31416 31417 31418 31419 --updates 2000 \
  --parallel 3
```

This extracts the frozen encoders from the saved `probes/step-N.pth` files. Their
previously trained readouts and calibrated loss weights are replaced completely.
The original probe seed uses the original untrained readout from `student.pth`,
so its result at the original update budget can be checked against the periodic probe. Other seeds
use fresh readout initializations. Within each seed, all SSL checkpoints receive
identical initial heads and minibatch sequences. The original data subsets,
batch size, AdamW settings, loss calibration, and precision are retained.

Embeddings are cached once per SSL checkpoint. Training and validation metrics
are recorded at 0, 50, 100, 200, 500, 1000, 2000, and the final requested update budget
(excluding budgets above that final value). Evaluation preserves RNG state, and
hash checks verify unchanged encoders and identical paired head initializations.
The default comparison uses five seeds, 192 training events, 96 validation events,
and 2,000 updates per trial: 15 trials across the three SSL checkpoints.

Results go to `RUN/probe_learning_curves`: each trial has `metrics.jsonl`,
`result.json`, and final `readout.pth`. `learning_curves.png/pdf` show training
and validation means with one seed standard deviation; `paired_differences.png/pdf`
show differences from the initialized backbone within each seed. `summary.json`
retains individual seed values. Bands describe seed variation, not confidence
intervals over the event population. The official test split remains untouched.
Rerunning the exact command skips completed trials and resumes incomplete trials
from their most recent evaluation-budget checkpoint, including readout weights,
optimizer, minibatch generator, and Torch RNG state. Older unfinished trials
without a `latest.pth` checkpoint restart from initialization;
use a new `--output` directory for a different experiment configuration.
`--parallel` controls concurrent GPU workers (default 1). Workers are separate
processes with independent model and RNG state; the coordinator alone writes the
combined plots. Their logs are saved as `worker.log` in the trial directories.
Concurrency can be changed when restarting the same experiment.

For the completed raw-hit run, use its available checkpoints and bound allocator
memory when running two workers on the local 16 GB GPU:

```bash
PF_SITE=local OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m mlpf.probe_learning_curves \
  --run experiments/cld_hits_ssl --ssl-steps 0 500 1000 \
  --seeds 31415 31416 31417 31418 31419 --updates 2000 \
  --parallel 2 --gpu-memory-fraction 0.46
```

Set-mode curves report matched-particle F1 and matched PID accuracy. Additional
`particle_counts_*` and `energy_*` plots/summaries track precision, recall,
particle counts, event energy response, MET error, and matched-particle energy
error. Each trial retains all standard particle metrics at every evaluation
budget. Curves and paired differences refresh as trials finish; worker logs and
`metrics.jsonl` expose intermediate budgets. Trials are scheduled by seed, then
SSL checkpoint, so paired comparisons become available before all seeds finish.
The memory fraction is an execution limit, not a change to probe batch size or
training budget. Initialization hashes and frozen-encoder hashes are verified.
Parallel campaigns first extract `embeddings-step-N.pth` caches in a single GPU
process, then share those CPU caches across all seeds. This avoids overlapping
the encoder-extraction memory peaks of multiple workers. Each cache records its
encoder hash, dataset selection, split, event counts, precision, and extraction
batch size; incompatible caches are rejected. Cache preparation logs go to
`cache.log` separately from the per-trial logs.

To compare the completed region and thinning runs, use:

```bash
bash scripts/local/probe_hits_comparison.sh
```

This schedules 40 trials: five paired seeds, four checkpoints per method, and
5,000 probe updates per trial. Region steps 0/60k/80k/100k and thinning steps
0/30k/40k/50k match sampled-event exposures. The existing 1,024/256 probe event
cohorts and batch size 16 are retained. Two workers share the GPU; each method's
embedding caches are prepared serially before fitting. The methods run in
sequence to keep the global worker count at two. The original trained probe
heads are discarded, and initial-head hashes are checked within and across
methods. This varies probe seeds, not SSL pretraining seeds.

Artifacts are under `experiments/cld_hits_ssl_comparison_5k_5seeds`: `manifest.json`
records the primary endpoint (particle F1 at 5,000 probe updates and final SSL
exposure), `regions/` and `thinning/` contain resumable trials and per-method
plots, and `comparison.json/png/pdf` are produced once all trials complete.
The analysis script `compare.py` is retained in that experiment directory.
`status.txt`, `run.log`, each method's `cache.log`, and each trial's `worker.log`
record progress. An exclusive lock prevents duplicate launchers. Rerunning the
command resumes the same experiment; it does not start new SSL pretraining.

To add the intermediate region checkpoints at 20k and 40k with the same five
seeds and 5,000-update budget:

```bash
bash scripts/local/probe_hits_early_checkpoints.sh
```

This runs ten additional trials in `regions_early/` under the comparison directory
and reuses the completed controls. `early_run.log` and `early_status.txt` track
progress. On completion, `ssl_step_benefit.py` refreshes the benefit plots and JSON
using all six region checkpoints. Only complete five-seed groups are merged;
configurations and paired initial-head hashes must match the original trials.

## Raw-hit CLD student/teacher training

```bash
bash scripts/local/pretrain_hits.sh --output experiments/cld_hits_ssl
```

The recipe `configs/training/cld_hits_ssl.yaml` pins the local dataset to
`cld_edm_ttbar_hits/1:3.2.1` under `/mnt/work/mlpf/tensorflow_datasets/cld`.
It uses the existing six-layer, width-256 HEPTv2 backbone with tracker-neighborhood
features and the 256-slot particle-set decoder. The full teacher and two 60%
student views use the same objective as the track/cluster baseline. Each
encoding call pads to HEPTv2's bucket size with masked rows and restores the
original row axis afterwards. This includes compact student views, probe
extraction, and snapshots; no backbone layers are modified.

The local default is SSL batch 2 with activation checkpointing disabled, and PF
probe batch 16. Stress testing repeats the largest event in the configured
512-event pool (13,446 hits). SSL batch 2 reserved 9.1 GiB and processed
4.10 events/s on the 16 GB RTX 5060 Ti; batch 3 reserved 13.5 GiB for less than
1% more throughput, and batch 4 ran out of memory. Activation checkpointing
allowed batch 16 but was slower at 3.56 events/s. The PF probe stress test at
batch 16 reserved 8.4 GiB, including the resident SSL model and optimizer.
These bounds apply to this configured pool and architecture, not arbitrary hit
datasets. Increasing the batch changes event exposure per update; retain the
same probe settings when comparing SSL checkpoints.

Recheck capacity when changing the data pool or model:

```bash
PF_SITE=local PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/local/benchmark_ssl_batch.py \
  --config configs/training/cld_hits_ssl.yaml --scan-per-dataset 512 \
  --batches 1 2 3 4 --probe-batches 8 16 24 \
  --output experiments/cld_hits_ssl_batch_benchmark.json
```

The benchmark includes warmup, Adam state, both student views, and the teacher.
Per-step training logs include a Unix start time and CUDA peak allocated/reserved
memory to correlate training intervals with external GPU-utilization samples.
CPU hit-pair fits and checkpoint I/O are separate diagnostic overhead.
On 51 real-data SSL steps after warmup, batch 2 sustained 4.46 events/s with
82.3% mean GPU utilization (22 device-wide one-second samples), 8.34 GiB peak
allocated / 9.13 GiB reserved memory, and 0.21 ms median exposed data wait.
These utilization numbers exclude periodic probes, snapshots, and validation.
The set probe uses grouping chunks of 8192 instead of 1024: a ten-update
batch-16 stress comparison reduced readout training time from 10.36 to 6.69 s
without changing the objective. Both the batch sweeps and real-data measurements
are saved in [the local evidence record](../../configs/training/references/cld_hits_ssl_local_batch.json).

SSL reads only inputs. Periodic diagnostics use disjoint training/validation
slices of the training split; the official test split remains untouched:

- Fresh PF set-decoder probes use the same initialization, data, seed, and
  update budget at each SSL checkpoint. They report supervised losses and
  matched-particle metrics and save trained readout checkpoints/predictions.
- `probe.hit_pairs.enabled: true` fits four additional linear logistic probes:
  tracker/raw, tracker/embedding, calorimeter/raw, and calorimeter/embedding.
  Each uses absolute pairwise feature differences on the same union of spatial
  and directional nearest-neighbor edges. Pairs are sampled deterministically,
  standardization uses training events only, and fitting balances events and
  the classes present in each event. Validation retains natural pair prevalence.
  Particle IDs are used only for within-event equality, never as global classes.
- Pair probes deliberately use an **oracle foreground mask** (positive
  `particle_number`). They report equal-event mean AP, ROC AUC, positive-pair
  prevalence, and per-event values. These measure pair discrimination, not full
  clustering or background rejection. Degenerate evaluation events are excluded
  from AP/AUC and the number with both classes is recorded.
- Pair coefficients and training scalers are saved in
  `probes/step-N-hit-pairs.npz`; metrics and settings are in `step-N.json`.
  Frozen embeddings are reused from the PF probe cache. No labels, fitted probe
  weights, or probe RNG changes feed back into the SSL training.

Refresh embedding, PF-probe, hit-pair-probe, and combined training plots with:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/local/plot_ssl_evolution.py \
  experiments/cld_hits_ssl
```

Do not pass the track/cluster-specific `--input-schema` flag for hits.
The pair plot includes the positive-pair fraction because calorimeter neighbor
pairs are often overwhelmingly from the same particle. Compare raw and learned
features within a detector, not absolute AP between detectors.

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

Additional offline diagnostics are generated by the same script. For CLD
track/cluster physics maps and kinematic probe bins, opt into the input schema:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/local/plot_ssl_evolution.py \
  experiments/cld_ssl_fast --input-schema cld-track-cluster --max-step 1200
```

Omit `--max-step` to include all available artifacts. The extra plots are saved
as both PNG and PDF, with numerical results in JSON:

- `training_and_probes`: training/validation distillation loss, teacher entropy,
  prototype usage, and trained-probe metrics from `metrics.jsonl`. Every standard
  plot refresh updates this dashboard; its title and JSON metadata record the
  latest included steps. An incomplete final log record is ignored while training
  is running. `--max-step` also applies to this dashboard.
- `neighborhood_stability`: checkpoint-by-checkpoint overlap of the 15 nearest
  neighbors, using cosine and Euclidean distances, overall and within each input
  type. Queries and candidates come from the same fixed cohort, excluding the
  query itself. `neighborhood_history.json` records selected rows, steps, and
  overlap matrices. Default limits of 12 checkpoints and 2,000 rows bound the
  quadratic computation; adjust `--max-snapshots`, `--max-elements`, and
  `--neighbors` as needed. Subsampling is deterministic and shared across steps.
- `embedding_physics`: aligned first/middle/last embedding maps colored by input
  pT/ET, eta, and phi. Axes and color scales are shared across steps. This requires
  `--input-schema cld-track-cluster`; other modalities still support the generic
  embedding and neighborhood plots without assuming the meaning of input columns.
- `probe_subgroups`: presence F1 and PID accuracy by input type, plus fixed
  pT/ET and eta quantile bins when the CLD schema is selected. Counts refer to
  input elements; PID accuracy includes only rows with nonzero particle targets.
- `probe_precision_recall`: presence threshold sweeps and average precision,
  overall and per input type. Dots mark the existing probability-0.5 decision.
  Average precision groups tied scores and uses a stepwise integral, allowing
  changes in ranking to be distinguished from changes at one decision threshold.
  `probe_subgroups.json` contains subgroup counts, metrics, and average precision.

The probe detail plots use the bounded prediction export cohort (12 events in
the default CLD recipe), rather than the full 96-event probe validation sample.
They are descriptive estimates without uncertainty intervals. The script checks
that inputs, targets, and event identities agree across prediction exports.
It skips input-aligned probe details for set outputs, whose predictions require
particle matching. Embedding comparisons also verify row identities and inputs,
in addition to the cohort hash. All plots run on CPU without loading checkpoints,
TFDS, or modifying the ongoing training job.

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

## Raw-hit context and individual-hit diagnostics

Run the input-only audits on a completed raw-hit SSL run:

```bash
PF_SITE=local OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m mlpf.ssl_diagnostics --run experiments/cld_hits_ssl_large
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/local/plot_ssl_diagnostics.py \
  experiments/cld_hits_ssl_large/diagnostics
```

The context audit loads the final student/teacher from `latest.pth` and the
initialized encoder from `probes/step-0.pth`. It holds 16 query hits of each
detector type fixed in each of 64 validation events and compares repeated full
context, the actual 60% sampling augmentation, same-type hits from other events,
query-only context, and individually isolated hits. Queries are selected from
the sampled view so they remain present in every condition. Replacement preserves
event length, detector types, query measurements, and row positions. Four other
events supply donor hits; donor reuse is recorded if the pool is insufficient.
Feature engineering is recomputed for every view. Isolated and foreign contexts
are stress tests outside the training distribution, not realistic detector models.

Individual-hit predictors fit the final full-event teacher's soft prototype
targets using only raw measurements of one hit. Defaults are 256 training events,
128 disjoint validation events, up to 256 hits/type/event, two 256-wide hidden
layers, 4,000 updates, and three seeds. Geometry-only predictors omit energy,
transverse energy, and time; all-feature predictors use all 15 raw columns.
Both exclude neighborhood features. Standardization uses training samples only,
and sampling/evaluation balance events and detector types. A detector-type
marginal predictor controls for prototype imbalance. No particle labels are read.

Outputs include event-level context metrics, cached teacher targets, predictor
checkpoints/learning curves, and PNG/PDF plots under `RUN/diagnostics`.
Context intervals bootstrap events; predictor bands show seed standard deviation.
Agreement refers to prototype argmax, not particle ID or PF reconstruction.
The initialized encoder is an embedding-context control only: the original SSL
projection head is not saved with its probe checkpoint. Strong single-hit
prediction establishes target predictability, not that the backbone ignores
context or that all of its information is geometric.

## Masked detector regions

`configs/training/cld_hits_ssl_regions.yaml` selects `ssl.mode: regions` for
the EDM4hep 15-column raw-hit schema. Existing configurations default to
`thinning`; their state dictionaries and exported MLPF architecture are unchanged.
The new mode adds an auxiliary query predictor outside the backbone. Track/cluster
inputs continue to use the thinning mode; region masking rejects incompatible schemas.

The fixed region address is `(element type, subdetector, system, side, layer,
eta bin, phi bin)`. Default coarse bin widths are 0.5 in eta and 2*pi/16 in phi;
each region has a fixed 2x2 prediction grid. This first implementation uses angular
bins on detector surfaces, not exact sensor pixels or a volumetric voxel grid.
Phi is periodic. Resolution can be changed when building a new catalogue.

The provided `configs/training/references/cld_hits_region_geometry.json` is an
empirical support map from 2,048 training events, with 23,617 regions. It is not
a complete active-channel map. Only fine bins observed somewhere in the training
catalogue contribute to measurement losses. No current-event occupancy is used to
mark a cell valid. `catalogue_target_coverage` monitors the fraction of removed
hits covered by this fixed support. New geometries need their own catalogue;
build it before training, using training inputs only:

```bash
PF_SITE=local OPENBLAS_NUM_THREADS=1 .venv/bin/python -m mlpf.ssl_region_geometry \
  --config configs/training/cld_hits_ssl_regions.yaml \
  --output /path/to/new_geometry.json --events 2048
```

Set `ssl.regions.geometry_path` to that path. The builder refuses to overwrite
an existing catalogue. Catalogue tensors are included in SSL checkpoints.

For each view, the sampler first chooses target surfaces independently of the
event's hidden contents. It selects query regions using angular activity on
other, non-target surfaces. Sampling weights are the square root of visible hit
counts in each angular bin. Defaults select up to 128 tracker and 512 calorimeter
regions, on random subsets of 25% and 40% of catalogue surfaces respectively.
Regions are layer-local patches; this version does not force consecutive-depth
calorimeter blocks. These surface fractions are not hit-removal fractions.
A 64-event held-out audit measured mean hit removal of 9.6% tracker / 19.1% calo,
with ranges 3.3–28.9% / 9.5–36.0%. Mean catalogue target coverage was 99.991%.

All hits inside selected regions are removed before student feature engineering.
The student predictor sees fixed region-center metadata and visible backbone
embeddings; it receives no hidden point positions, centroids, counts, energies,
or per-hidden-hit tokens. Query validity depends on the fixed catalogue and
visible context. Empty and occupied query cells both contribute to the loss.

The objective is occupancy BCE + 0.1 * count Poisson NLL + log1p-energy Huber
loss + 0.1 * optional teacher-prototype CE. Energy targets are sums of measured
calorimeter hit energies, including zero-energy cells; tracker energy is excluded.
Reduction balances detector types and events, averaging supported cells within
each type. Precision/recall/F1 and occupied fraction accompany occupancy accuracy
to expose the all-empty shortcut. All targets are derived from input hits.

The full-event EMA teacher supplies mean embeddings of hits in each nonempty
hidden region for the optional latent term. Empty regions have no latent target.
The existing centering/sharpening projection is reused on these pooled embeddings;
`latent_valid_queries` reports target support. Set `latent_weight: 0` for the
measured-content-only ablation, or set the three measurement weights to zero
for a latent-only ablation. A zero total objective is rejected. Legacy per-hit
prototype diagnostics deliberately reject region-mode checkpoints: their targets
are different. Context-dependence and coordinate-only baselines for the new
region targets remain a follow-up evaluation, not a claimed result of this implementation.

Start a new run (the objective cannot be changed with `--resume`):

The local recipe uses batch 1 and 100,000 updates, retaining the prior run's
100,000 sampled-event exposure. On the largest event in the full 20,000-event
pool (14,703 hits), batch 1 reserved 7.54 GiB at 2.66 events/s; batch 2 reserved
14.59 GiB at 2.83 events/s. Batch 1 leaves room for stochastic variation in mask
density. Evidence is in `configs/training/references/cld_hits_ssl_regions_local_batch.json`.
Equal event exposure is not equal wall time or optimizer-step count.

```bash
CONFIG=configs/training/cld_hits_ssl_regions.yaml \
OUTPUT=experiments/cld_hits_ssl_regions \
  bash scripts/local/pretrain_hits.sh
```

The ordinary periodic frozen PF probes, optional hit-pair probes, snapshots,
checkpoint resume, and standard `student.pth` export remain available. The
auxiliary predictor is not included in the downstream MLPF export. A fixed
`region_view.npz` preview is saved at initialization without changing training RNG:

```bash
.venv/bin/python scripts/local/plot_ssl_regions.py experiments/cld_hits_ssl_regions
.venv/bin/python scripts/local/plot_ssl_evolution.py experiments/cld_hits_ssl_regions
```

The combined training/probe dashboard switches to region total loss, occupancy
loss, and energy loss for this mode. No detector-resolution jitter, calibration
variation, timing shift, or background overlay is enabled. These are separate
ablations; this implementation establishes the masked prediction task first.
