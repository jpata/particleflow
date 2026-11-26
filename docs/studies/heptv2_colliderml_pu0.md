# HEPTv2 and Attention on ColliderML PU0

The Tallinn study replays the two v3.3.0 ColliderML PU0 recipes with their
published step-100,000 checkpoints. One NVIDIA L40S ran each replay in BF16 with
PyTorch 2.11.0+cu128. The Slurm `gpu:l40:1` resource in the Tallinn profile
allocated a 46,068 MiB L40S. The dense HEPTv2 reference is `224de191`, whose
ColliderML layers and hashing match the v3.3.0 release. [Issue 588](https://github.com/jpata/particleflow/issues/588) ·
[Published checkpoints](https://huggingface.co/jpata/particleflow/tree/main/colliderml/clusters/v3.3.0)

## Implementation

- Deterministic HEPTv2 computes fixed-point coordinates once after the backbone
  coordinate transforms. Every layer uses the resulting integer coordinates
  with its own hash projections and region parameters. Checkpoint replay uses
  the same tensor.
- `model.heptv2.pack_padding=true` keeps the original padded width for hashing
  and quantile boundaries. It projects tokens in occupied buckets, including
  the padded lanes in partially filled buckets, and gathers only those buckets
  for attention. The residual MLP runs only on valid tokens. Packing applies
  in eager mode with deterministic hashing and either evaluation mode or zero
  backbone dropout. It is opt-in.

## Replay method

- Data is ColliderML ttbar PU0 TFDS `colliderml_ttbar_nopu_pf/1:1.5.0` under
  `/scratch/persistent/joosep/colliderml/tfds/`. Each arm receives the same
  ordered event indices and padding. The training replay uses sequential
  training events and the production loss, forward and backward passes,
  LAMB, and scheduler. Each batch size starts with the release weights and a
  fresh optimizer and scheduler.
- Batched inference repeats the first held-out events after the release
  padding to multiples of 128. The unpadded sweep uses the same held-out
  prefix for both backbones, using a slice of one event where it fits and
  combining real elements from multiple test events for longer inputs.
- Host wall time is measured with CUDA completion at each step. TFDS loading
  and host-to-device copies occur outside the timing interval. CUDA-event
  durations, median and p95 wall time, events/s, valid elements/s, and padded
  widths are saved in the replay JSON. Batch scaling efficiency is the
  ratio of a batch's events/s to batch-one events/s, divided by batch size.
  Padding efficiency is the number of valid elements divided by the number
  of allocated token slots.
- Allocated and reserved VRAM are PyTorch's peak allocator counters. Device
  VRAM is the maximum `nvidia-smi` memory reading sampled during the
  measurement window, including runtime overhead. The separate 1,280-element
  profiler pass counts CUDA kernels and sums their durations.

## Training throughput and VRAM

The HEPTv2 before/after replays ran on the same allocated GPU for each
checkpointing mode (Slurm array 62896139). The Attention reference uses its
release recipe in array 62895832. Training at batch 32 fills about 61% of
token slots. At batch 128 it fills about 58%.

| Events | Backbone | Checkpointing | Median step (ms) | Events/s | Batch efficiency | Peak GiB allocated / reserved / device |
|---:|---|:---:|---:|---:|---:|---:|
| 32 | Attention | off | 175.7 | 183.8 | 92% | 2.42 / 3.88 / 4.44 |
| 32 | Release HEPTv2 | on | 316.2 | 101.5 | 84% | 2.71 / 3.49 / 4.06 |
| 32 | Packed HEPTv2 | on | 252.6 | 128.2 | 92% | 1.88 / 2.57 / 3.13 |
| 32 | Release HEPTv2 | off | 261.8 | 121.5 | 50% | 12.72 / 15.67 / 16.24 |
| 32 | Packed HEPTv2 | off | 177.2 | 177.6 | 63% | 8.05 / 10.21 / 10.78 |
| 128 | Attention | off | 244.2 | 523.2 | 66% | 9.26 / 12.12 / 12.68 |
| 128 | Release HEPTv2 | on | 1329.5 | 95.3 | 20% | 10.36 / 13.31 / 13.88 |
| 128 | Packed HEPTv2 | on | 901.5 | 142.0 | 25% | 6.99 / 10.65 / 11.21 |
| 128 | Release HEPTv2 | off | OOM | | | |
| 128 | Packed HEPTv2 | off | 688.3 | 186.0 | 17% | 31.51 / 38.47 / 39.04 |

At batch 32 without checkpointing, packing cut peak allocated memory by
37% and increased throughput by 46%. At batch 128 with checkpointing,
packing cut peak allocated memory by 33% and increased throughput by 49%.
The highest measured packed training throughput in the paired runs was
192 events/s at batch 64 without checkpointing.

## Batched inference

The three arms ran sequentially on one L40S (Slurm array 62896139).
Entries show events/s and batch scaling efficiency.

| Events | Attention | Release HEPTv2 | Packed HEPTv2 |
|---:|---:|---:|---:|
| 1 | 17 / 100% | 13 / 100% | 19 / 100% |
| 8 | 132 / 99% | 108 / 106% | 154 / 100% |
| 16 | 260 / 98% | 216 / 105% | 314 / 102% |
| 32 | 536 / 101% | 411 / 100% | 533 / 87% |
| 48 | 794 / 99% | 418 / 68% | 618 / 67% |
| 64 | 1,030 / 97% | 399 / 49% | 594 / 48% |
| 96 | 1,414 / 89% | 390 / 32% | 549 / 30% |
| 128 | 1,555 / 73% | 354 / 22% | 509 / 21% |

Packed HEPTv2 peaked at batch 48 at 618 events/s. At batch 128, its
device VRAM fell from 6.03 to 4.75 GiB and median request latency from
362.1 to 251.3 ms. Attention used 2.70 GiB of device VRAM and took
82.1 ms per batch at that size.

## Unpadded batch-one inference

The three HEPTv2 arms ran sequentially on one L40S (Slurm job 62896004).
Timings are median / p95 in milliseconds. The fusion experiment used
`torch.compile` on the single coordinate conversion.

| Valid elements | Release HEPTv2 | Reuse coordinates | Reuse and fuse |
|---:|---:|---:|---:|
| 128 | 58.6 / 69.4 | 34.6 / 38.4 | 31.7 / 42.3 |
| 512 | 56.9 / 93.7 | 34.5 / 38.5 | 30.0 / 32.6 |
| 1,280 | 56.8 / 67.8 | 36.3 / 39.8 | 31.4 / 41.3 |
| 4,096 | 71.0 / 90.1 | 47.6 / 55.2 | 37.2 / 43.8 |
| 8,192 | 132.4 / 142.2 | 51.9 / 65.0 | 39.8 / 49.7 |
| 12,288 | 70.5 / 84.4 | 50.2 / 67.0 | 41.0 / 49.2 |
| 16,384 | 77.9 / 90.3 | 49.2 / 57.6 | 40.9 / 44.2 |
| 19,968 | 79.1 / 90.1 | 50.5 / 65.1 | 41.9 / 45.0 |

At 1,280 elements, reuse reduced the kernel count from 3,556 to 1,701
and their summed execution time from 6.81 to 3.99 ms. All the HEPTv2
arms peaked at 100 MiB allocated and 661 MiB device VRAM at that length.
At 19,968 elements, they peaked at 603 MiB allocated and 1,307 MiB on the
device. The fused trial reached 1,332 kernels and 3.42 ms of summed kernel
time at 1,280 elements. The branch uses eager reuse, which improved p95 at
small inputs with the existing execution path. The fused trial had higher
p95 at 128 and 1,280 elements and adds first-use compilation.

In a separate paired replay (job 62896421), Attention and HEPTv2
with coordinate reuse ran on the same L40S. Each entry gives median /
p95 latency and sampled device VRAM.

| Valid elements | Attention ms | HEPTv2 ms | Attention MiB | HEPTv2 MiB |
|---:|---:|---:|---:|---:|
| 128 | 36.1 / 47.6 | 31.4 / 36.6 | 591 | 617 |
| 512 | 37.6 / 43.9 | 33.2 / 41.4 | 595 | 643 |
| 1,280 | 38.2 / 54.3 | 31.7 / 35.9 | 613 | 661 |
| 4,096 | 41.1 / 47.6 | 37.5 / 64.3 | 663 | 775 |
| 8,192 | 59.7 / 62.9 | 38.8 / 63.8 | 721 | 931 |
| 12,288 | 62.9 / 68.1 | 46.7 / 69.6 | 775 | 1,043 |
| 16,384 | 62.1 / 65.1 | 42.3 / 47.0 | 845 | 1,177 |
| 19,968 | 51.4 / 66.6 | 45.3 / 50.7 | 865 | 1,307 |

Attention launched 472 kernels with 1.15 ms of summed kernel time at
1,280 elements. Absolute wall times shifted across runs. The paired
numbers preserve the same inputs and GPU within each comparison.

## Correctness

Deterministic BF16 replay used the release checkpoint, a padded batch of
eight held-out events, unpadded sequences of 1,280 and 8,192 elements,
and a separate training batch of eight with activation checkpointing on.

- Coordinate reuse matched all 36 valid inference tensors, all four
  training prediction tensors, the loss, and all 168 gradient tensors
  exactly.
- Packing matched the same predictions and loss exactly. Across the 168
  parameter gradients, 87 were exact and the aggregate relative L2
  difference was 0.240%. The maximum absolute difference was 0.00342.
- The HEPTv2 and checkpoint test suite passed all 17 tests on L40S,
  including a padding case with interior masked slots, tied coordinates,
  partially filled buckets, and an empty event.
- CPU and ONNX coverage passed 22 tests with six GPU cases skipped. Four
  standalone ONNX TopK export cases fail with the same shape-inference
  error on the dense reference and this branch.

The full sweep and replay JSON are in the worktree's
`experiments/issue588/`. The first-pass matrix spans 1–128 events per batch.
These short replays measure performance from the published weights. The issue
records the full-run physics results. [Issue 588](https://github.com/jpata/particleflow/issues/588)

## Reproduce

Run on a single GPU with the release weights downloaded to `attention.pth`
and `heptv2.pth`. The Python driver defaults to the Tallinn profile and the
scratch TFDS path above. Use the same driver in the dense reference checkout
to reproduce the release path.

```bash
export KERAS_BACKEND=tensorflow
export CUBLAS_WORKSPACE_CONFIG=:4096:8

uv run python3 scripts/studies/benchmark_colliderml_heptv2.py \
  --phase infer --variant attention --weights attention.pth \
  --repeats 40 --output experiments/issue588/attention-infer.json
uv run python3 scripts/studies/benchmark_colliderml_heptv2.py \
  --phase infer --variant heptv2 --weights heptv2.pth \
  --repeats 40 --profile-length 1280 --output experiments/issue588/heptv2-infer.json
uv run python3 scripts/studies/benchmark_colliderml_heptv2.py \
  --phase train --variant heptv2 --weights heptv2.pth --pack-padding \
  --no-checkpointing --repeats 8 --output experiments/issue588/packed-train.json
```

`--phase parity --batches 8 --lengths 1280 8192 --checkpointing` saves a
reference snapshot of held-out predictions and a training batch's predictions,
loss, and gradients. Repeat with `--reference-snapshot` to produce a JSON
comparison against that snapshot.
