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

## Capacity-matched follow-up

The release Attention model has 3.114M trainable parameters. For the
follow-up, its feed-forward width was widened from 256 to 2,048, which
gave 8.630M trainable parameters versus packed HEPTv2's 8.639M (0.12%
apart). Both models have six layers, 256-wide embeddings, 16 heads, and
zero backbone dropout. Attention uses the Tallinn profile's native
jagged Flash SDPA. HEPTv2 uses the packed path with three hashes.
The comparison starts from seeded weights, uses the same ColliderML PU0
inputs and BF16, and runs LAMB with the same learning-rate schedule and
activation checkpointing disabled. The training measurements replay
short optimizer-step windows from initialization.

### Throughput and memory at matched capacity

Training arms ran on one L40S (job 62896983), and inference arms ran
on one L40S (array 62896883, task 1).

| Phase | Events/batch | Backbone | Median (ms) | Events/s | Peak GiB allocated / reserved / device |
|---|---:|---|---:|---:|---:|
| Training | 32 | Flash Attention | 138.2 | 231.5 | 3.44 / 4.60 / 5.17 |
| Training | 32 | HEPTv2 | 177.7 | 180.2 | 8.06 / 10.22 / 10.79 |
| Training | 128 | Flash Attention | 245.9 | 521.6 | 13.14 / 24.38 / 24.94 |
| Training | 128 | HEPTv2 | 710.7 | 179.8 | 31.52 / 38.48 / 39.05 |
| Inference | 32 | Flash Attention | 40.2 | 741 | 0.42 / 0.61 / 1.15 |
| Inference | 32 | HEPTv2 | 56.0 | 578 | 0.84 / 1.02 / 1.55 |
| Inference | 128 | Flash Attention | 73.7 | 1,734 | 1.59 / 2.41 / 2.95 |
| Inference | 128 | HEPTv2 | 246.9 | 518 | 3.41 / 4.22 / 4.75 |

At batch 128, Flash Attention delivered 2.90× the training throughput
and 3.35× the inference throughput. Its forward/loss and backward
phases took 76 and 147 ms, versus 258 and 409 ms for HEPTv2. Training
batch scaling efficiency at 128 was 62% for Flash Attention and 15% for
HEPTv2. Inference efficiency was 94% and 19%, respectively.

### Batch-one latency

The two models alternated request by request on one GPU (job 62896982).
Every point has 40 measured pairs. Peak device VRAM is from separate
single-model replays in array 62896883.

| Valid elements | Flash Attention median / p95 (ms) | HEPTv2 median / p95 (ms) | Flash / HEPTv2 device MiB |
|---:|---:|---:|---:|
| 1,280 | 58.7 / 61.8 | 47.9 / 51.6 | 653 / 667 |
| 8,192 | 61.8 / 65.0 | 63.4 / 66.7 | 813 / 931 |
| 19,968 | 61.5 / 65.2 | 64.2 / 67.3 | 1,055 / 1,311 |
| 32,768 | 90.3 / 95.0 | 75.6 / 80.5 | 1,349 / 1,803 |
| 65,536 | 310.6 / 313.9 | 144.3 / 148.8 | 2,073 / 3,005 |

At 1,280 elements, the CUDA-only profiler measured 472 kernels and
1.27 ms of summed kernel time for Flash Attention, and 1,704 kernels
and 3.94 ms for HEPTv2. A separate interleaved diagnostic held the
Attention weights fixed and used dense or jagged Flash SDPA on fully
valid inputs (job 62897041).

| Valid elements | Jagged Flash median (ms) | Dense Flash median (ms) | HEPTv2 median (ms) |
|---:|---:|---:|---:|
| 1,280 | 57.3 | 20.1 | 47.7 |
| 8,192 | 57.9 | 20.4 | 51.6 |
| 32,768 | 85.7 | 80.2 | 68.6 |
| 65,536 | 299.9 | 297.1 | 139.2 |

The layout change matched the raw Attention outputs exactly at these
sizes. The 37 ms gap between the two Flash paths at 1,280 elements
shows how strongly the end-to-end result depends on jagged dispatch
when GPU work is small. At 32,768 elements, summed kernel time in the
first interleaved replay was 80.8 ms for Flash Attention and 50.5 ms
for HEPTv2. HEPTv2's long-sequence speedup persists with dense Flash.

### Where the cost comes from

Flash Attention computes all pair scores within each event using tiled
SDPA. It keeps the score matrix out of device memory. Packed HEPTv2
sorts by hash, gathers Q/K/V into each hash's buckets, runs local
attention, undoes the permutations, and combines the results using
their log-sum-exp values. Its position embedding and RMSNorm also keep
backward activations.

Per head and layer, global Attention computes `sum(L_i ** 2)` pair
scores for a batch with event lengths `L_i`. HEPTv2 computes roughly
`sum(3 * 128 * round_up(L_i, 128))`. At batch 32 the measured event
lengths gave HEPTv2 about 56% as many pair scores. The routing,
copies, and per-layer saved activations outweighed that savings. A
batch of independent short events keeps the per-event global attention
problem small. On one 65,536-element sequence HEPTv2 computes about
0.6% as many pair scores, so the local attention has room to pay for
routing overhead.

At batch 32, unique CUDA storage saved for backward was 2.79 GiB for
Attention and 6.89 GiB for HEPTv2. The widened feed-forward blocks
accounted for about 1.16 GiB in each model. HEPTv2 retained an extra
1.81 GiB in bucket attention and hash mixing, 1.40 GiB in its learned
position embedding, and 0.84 GiB in normalization. The diagnostic
excludes live weights, buffers, and batch inputs, and assigns shared
storage to the first scope that saves it.

As an ablation, limiting packed HEPTv2 to one hash lowered batch-32
saved attention storage from 2.16 to 0.91 GiB and raised batch-128
training throughput from 180 to 240 events/s. At batch 128 it lowered
peak allocated memory from 31.52 to 26.06 GiB, while sampled device
VRAM rose from 39.05 to 40.35 GiB with the larger allocator reservation.
The ablation changes routing. It measures system cost using seeded
weights.

Raw capacity-matched results, interleaved timing drivers, plots, and
Slurm logs are in the worktree's `experiments/issue588/capacity/`.
Use `--matched-capacity` with the benchmark driver on either variant
and omit `--weights`. Use `--pack-padding` for HEPTv2 and
`--diagnose-training-batch 32` for storage attribution. Both training
arms use `--batches 1 8 16 32 64 128 --repeats 12`.

## Follow-up optimization experiments

The L40S prototypes kept the three hashes, the padded quantile geometry,
and the original model parameters. Training uses the matched-capacity
setup above, with four warmup and eight timed optimizer steps per batch
size. A separate deterministic BF16 replay loads the published HEPTv2
checkpoint and compares valid predictions, training loss, and gradients
against packed HEPTv2.

- **Permutation and output dtype.** Invert the known permutation with
  scatter and unsort the BF16 attention output separately from the FP32
  log-sum-exp. The weighted sum still accumulates in FP32. The
  existing flags are `HEPTV2_FAST_INVERT_PERM=1` and
  `HEPTV2_COMBINED_UNSORT=0`.
- **Bucket layout.** Gather the Q/K/V rows directly into the
  `[hash × bucket, head, slot, dim]` layout FlexAttention consumes.
  This removes the subsequent full-tensor rearrangement and copies.
- **RMSNorm fusion.** Compile the existing normalization arithmetic so
  it can avoid retaining FP32 intermediates for every reduction.
- **Position embedding checkpoint.** Recompute only the learned
  position embedding during backward. The attention and feed-forward
  computations take their ordinary backward paths.
- **Hash mixing fusion.** Compile unsorting and the LSE-weighted hash
  reduction together to reduce the number of intermediate buffers.

The permutation and output-dtype flags raised batch-128 training
throughput from 187.5 to 209.5 events/s and reduced sampled device VRAM
from 38.13 to 35.37 GiB in their paired run. Inference throughput rose
from 522 to 606 events/s at that batch size. Both paths matched all 41
prediction and loss tensors and all 168 parameter gradients exactly in
the deterministic replay. Direct bucket layout raised the paired
inference throughput to 611 events/s and lowered sampled VRAM to
3.86 GiB (4.75 GiB baseline). Its tested predictions and loss were
exact, and its aggregate gradient difference was 1.32% relative L2.

The first RMSNorm fusion reduced batch-32 storage saved by normalization
from 993 to 435 MiB. A checkpoint around the positional embedding saved
another 1,430 MiB in the same storage diagnostic. The default compiled
RMSNorm changed prediction tensors and produced a 12.50% relative L2
gradient difference. Hash mixing fusion with the default compiler
settings produced a 3.25% relative L2 gradient difference. These
measurements separate the potential savings from checkpoint parity.

The paired replay without RMSNorm fusion (job 62897653) measured the
following batch-128 training results.

| Packed HEPTv2 | Events/s | Peak allocated GiB | Sampled device GiB |
|---|---:|---:|---:|
| Baseline | 185.5 | 31.32 | 38.13 |
| Permutation and dtype flags + positional checkpoint | 197.7 | 23.53 | 26.92 |

That combination matched all 41 prediction and loss tensors and
168 parameter gradients exactly against packed HEPTv2. It is the
lowest-memory tested option with exact checkpoint replay. Enabling
Inductor's eager-precision cast emulation for compiled RMSNorm
reduced its gradient difference to 9.44% relative L2. The replay still
detected prediction differences. The cast-emulation training arm with
the positional checkpoint measured 220.7 events/s, 21.21 GiB peak
allocated, and 24.10 GiB sampled device VRAM.

A same-GPU training replay of the combined prototypes gives the
following batch-128 results (array 62897498, task 0).

| Backbone or prototype | Events/s | Peak allocated GiB | Sampled device GiB |
|---|---:|---:|---:|
| Matched Flash Attention | 479.1 | 13.14 | 20.57 |
| Packed HEPTv2 | 182.3 | 31.32 | 38.14 |
| Permutation and dtype flags + compiled RMSNorm | 241.7 | 27.13 | 32.98 |
| Direct layout + flags + compiled RMSNorm | 242.3 | 26.25 | 34.39 |
| Direct layout + flags + compiled RMSNorm + positional checkpoint | 226.7 | 20.34 | 25.15 |

The final three rows used default Inductor rounding and changed
checkpoint predictions. They measure the potential performance of
the implementation strategies.

The routing ablation and saved-storage diagnostic point to a larger
throughput opportunity inside bucket attention. The attention kernel
could load Q/K/V through the bucket indices directly from the original
token-major tensors. In backward, it would accumulate across hashes
and retain gradients through the LSE weights. This removes the explicit
per-hash Q/K/V copies that remain even with the improved layout.
Caching the stable geometry ranks once per backbone call would also
remove repeated sorts while retaining each layer's quantile region sizes.
A different model could use global Flash attention for short events and
hashed attention for long sequences. That changes which tokens interact
and would need training and a physics comparison.

The prototype source, Slurm drivers and raw metrics are under
`experiments/issue588/next/` in the worktree.

## Indexed bucket attention prototype

`HEPTV2_INDEXED_BUCKET_ATTENTION=1` enables an experimental Triton
kernel for 128-token buckets with BF16 Q/K/V and head dimension 16
on NVIDIA Ampere and newer GPUs. The kernel uses the same integer
bucket assignments, loads Q/K/V indirectly from their token-major
tensors, and stores the output and FP32 LSE directly to original
token order. This removes the per-hash Q/K/V tensors, inverse
permutation, and output gather in the FlexAttention path. The
implementation uses 64-element online softmax tiles and includes
gradients of the LSE used in hash mixing. It rounds each hash's
gradients to BF16, accumulates across hashes in FP32, and processes
hashes in order when deterministic algorithms are enabled.
The experimental switch is off by default.

The paired BF16 measurements below used an A100 80 GB PCIe (array 62904029), the
matched-capacity recipes and the same ColliderML PU0 inputs as the
earlier studies. HEPTv2 arms used packed padding, three hashes,
`HEPTV2_FAST_INVERT_PERM=1` and `HEPTV2_COMBINED_UNSORT=0`.
Training used four warmup steps and eight timed steps with
checkpointing disabled. Inference used 28 timed requests per point.

| A100 replay | Matched Flash Attention | HEPTv2 FlexAttention | HEPTv2 indexed |
|---|---:|---:|---:|
| Training at batch 128, events/s | 733.4 | 337.8 | 403.5 |
| Training at batch 128, peak allocated GiB | 13.13 | 29.43 | 24.68 |
| Training at batch 128, sampled device GiB | 20.65 | 35.17 | 33.00 |
| Inference at batch 128, events/s | 2,422 | 987 | 1,232 |
| Inference at batch 128, sampled device GiB | 2.94 | 4.50 | 3.90 |
| Unpadded 1,280-element median (ms) | 33.0 | 25.9 | 24.1 |
| Unpadded 32,768-element median (ms) | 130.0 | 50.3 | 41.8 |

At batch 32, bucket attention's saved storage fell from 1,755 to
606 MiB. The 19 HEPTv2 and checkpointing tests passed on A100 MIG.
The indexed kernel tests compare outputs and gradients to a dense
FP32 reference, exercise LSE-only loss, and check exact repeatability
in deterministic mode. The kernels also compiled for SM80 and SM89.

Holding each layer's actual input fixed, the maximum relative L2
difference in its indexed attention output was 0.013% across the
ColliderML replay. The whole-model replay on the same A100 MIG instance
showed a 0.63% difference in raw training predictions and a 9.00%
aggregate gradient difference. The one-tile prototype had similar
performance and a 15.99% aggregate gradient difference. These results
support the throughput and memory gains while keeping the tiled backend
an explicit experiment for checkpoint evaluation.

Raw measurements and Slurm launchers are in the worktree's
`experiments/issue588/indexed-a100/` and `experiments/issue588/indexed/`.

The L40S training and inference tasks in array 62903767 also completed.
At batch 128, indexing raised training throughput from 206 to 215
events/s and inference throughput from 603 to 699 events/s. The
training allocation peak fell from 29.43 to 24.68 GiB. The L40S
checkpoint replay measured an 11.72% aggregate gradient difference.

## Closing the remaining throughput gap

The batch-128 L40S training replay had 1,024 timed events with
a median of 734 valid elements, p95 of 1,038, and a maximum of 1,368.
Global Flash runs separately inside each event, so increasing the
number of short events per batch preserves short attention sequences.

A100 job 62904484 compared five variants on the same GPU and identical
ordered TFDS inputs. Each point uses three warmup and eight timed
optimizer steps. The HEPTv2 global-Flash probe keeps its coordinate
transform, positional MLPs, Q/K norms, 256-wide projections, and gated
feed-forward blocks. It substitutes jagged Flash over the valid tokens
within each event for the hash routing and attention. Both the global
and one-hash probes change the model's token neighborhoods.

| Batch-128 training variant | Events/s | Peak allocated GiB | Sampled device GiB |
|---|---:|---:|---:|
| Matched Flash Attention | 735.3 | 13.13 | 20.65 |
| Indexed HEPTv2, three hashes | 404.7 | 24.68 | 33.00 |
| Indexed HEPTv2 + positional checkpoint | 387.7 | 18.77 | 22.64 |
| Indexed HEPTv2, one hash | 466.6 | 23.80 | 32.01 |
| HEPTv2 block with global Flash | 485.7 | 22.50 | 32.09 |

Indexed HEPTv2 spent 107 ms in forward/loss and 180 ms in backward,
versus 55 and 94 ms for matched Flash. The backward phase accounts for
about 86 ms of the 142 ms step-time gap. Substituting global Flash
inside HEPTv2 recovered about 52 ms per step. Its remaining gap
motivates work on the tokenwise projections, per-layer position
embeddings and normalization alongside routing.

The positional checkpoint alone reduced sampled device VRAM by
31% with a 4% throughput cost. It recomputes the same position
embedding, which preserves the indexed path's routing and precision.

A second A100 replay (job 62904571) tested sharing the positional
embedding across all six layers. One learned positional MLP ran once
on the corresponding packed tokens, and the other five position
embeddings were removed. This probe has 7.30M trainable parameters,
compared with 8.64M for unmodified HEPTv2.

| Batch-128 training variant | Events/s | Peak allocated GiB | Sampled device GiB |
|---|---:|---:|---:|
| HEPTv2 block with global Flash, repeat | 484.4 | 22.50 | 32.09 |
| HEPTv2 block with global Flash + positional checkpoint | 461.6 | 17.07 | 21.79 |
| Indexed HEPTv2 + shared positional embedding | 456.7 | 19.74 | 27.08 |
| HEPTv2 block with global Flash + shared positional embedding | 557.0 | 17.95 | 26.64 |

Global Flash plus the shared embedding reached 76% of the matched
Flash recipe's throughput, with 27% less peak allocated memory than
the indexed baseline. The short-event probe changes the model. A
length-based variant could use this path for ordinary PU0 events and
indexed buckets on the long-sequence regime, with its threshold
chosen from the hardware crossover. Training and physics evaluation
would establish the effect on reconstruction quality.

For the three-hash architecture, follow positional checkpointing with
a backward profile of RMSNorm, the gated MLP, and the indexed attention
gradient accumulation. Candidate fusions should account for the
precision sensitivity in the published-checkpoint comparison. The two
A100 runs and the probe source are in `experiments/issue588/gap/`.

## Training with a shared positional embedding on L40S

Slurm array 62904824 compared the default learned positional MLP in
every layer with one learned MLP evaluated once and reused across all
six layers. Both arms kept width 256, 16 heads, three hashes, Flex
Attention, packed padding, and BF16. They started from seed 12345 with
identical initial weights for all common parameters and trained from scratch
for 10,000 optimizer steps at batch 32 on separate L40S GPUs. They
used ColliderML ttbar PU0 TFDS 1.5.0 from
`/scratch/persistent/joosep/colliderml/tfds/`. The 2,048 held-out
events came from the test split of configuration 10. Validation
used configurations 1–9. Both runs used the same LAMB and OneCycle
settings. The validation comparison uses unweighted loss because
the training loop calibrates its task weights independently.

Both jobs completed all 10,000 steps and processed 320,000 event
presentations each. The per-event sequence lengths matched at every
paired step. The training timing excludes the first 100 steps,
validation, and checkpoint writes.

| Measure | Per-layer embedding | Shared embedding |
|---|---:|---:|
| Trainable parameters | 8,639,442 | 7,301,842 |
| Training events/s | 197.84 | 212.37 |
| Median training step (ms) | 154.24 | 140.02 |
| p99 training step (ms) | 216.97 | 272.19 |
| Peak PyTorch allocated VRAM (GiB) | 8.51 | 7.16 |
| Peak sampled device VRAM (GiB) | 11.48 | 11.07 |
| Slurm elapsed | 31m 38s | 30m 04s |
| Final unweighted validation loss | 3.09365 | 3.09754 |
| Final matched jet pT response median | 0.99672 | 0.99619 |
| Final matched jet pT response IQR | 0.12441 | 0.12327 |
| Final response-qualified jet F1 | 0.75989 | 0.76068 |

Sharing raised training throughput by 7.3% and reduced peak PyTorch
allocated VRAM by 15.8%. The recorded p99 step time increased on the
shared run's node, where the p95 data wait was 22.9 ms versus 4.4 ms
for the per-layer run.

The per-layer model led response-qualified jet F1 through 8,000
steps. At 10,000 steps the difference was +0.08 percentage points
for the shared model. Paired resampling of the 2,048 test events
gave a 95% interval of −0.23 to +0.38 percentage points for the
F1 difference. The paired intervals for the jet response median
and IQR differences also included zero. These intervals reflect
test-event variation for this training seed.

The scenario is
`configs/training/scenarios/colliderml_heptv2_position_embedding_10k.yaml`
with `configs/training/platforms/tallinn_l40.yaml`. Both tasks used
`HEPTV2_FAST_INVERT_PERM=1`, `HEPTV2_COMBINED_UNSORT=0`, and
`HEPTV2_INDEXED_BUCKET_ATTENTION=0`. The Slurm launcher, training
logs, histories, checkpoints, and per-event predictions are in
`experiments/issue588/position_embedding_10k/` in the worktree.

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
