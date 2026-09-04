# Set-prediction MLPF for hit-based training

## Overall goal

Add an optional slot/cross-attention MLPF training scheme that predicts a compact,
unordered set of particles directly from the input objects. In this mode, the model
must learn the relationships between detector inputs and output particles instead of
emitting one particle candidate for every input object.

The first implementation targets the CLD and CLIC hit datasets, where an event can
contain more than 10,000 input hits but only about 100 target particles. It must be
computationally and memory efficient at that scale, while preserving the existing
elementwise MLPF path as the default and keeping existing datasets usable.

Relevant references:

- [Better Queries, Cheaper Attention: Adapting Transformers for Efficient Sparse Reconstruction](https://arxiv.org/abs/2606.17631)
- [HEPTv2: End-to-End Efficient Point Transformer for Charged Particle Reconstruction](https://arxiv.org/abs/2606.20437)

## Initial scope and decisions

- Do not regenerate the existing TFDS datasets for the first implementation.
- Derive a compact target particle collection in memory from the existing `ytarget`.
  Existing hit datasets contain one row with full particle properties for each target
  particle; other associated hits contain only `particle_number`. Therefore rows with
  a nonzero particle class form the compact target set.
- Extract the compact target before the current input-relative `pt` and energy
  transformations. Give set targets their own absolute, slot-compatible transforms.
- Treat input padding and target padding as independent axes, with separate masks.
- Add an architecture-level output mode. The existing elementwise mode remains the
  default and must retain its present behavior.
- For the first model, use a fixed bank of learned particle queries and a small
  cross-attention decoder. Dynamic queries, sectorization, and explicitly sparse
  cross-attention are follow-up optimizations.
- Use a scalable existing encoder (currently packed attention) for large hit collections. A
  dense all-to-all hit encoder is not acceptable for the 10k-hit use case.
- Use permutation-invariant Hungarian matching between predicted slots and compact
  target particles. This output-to-target matching does not associate targets with
  individual input hits.
- Do not require hit-to-particle association labels for the initial set-prediction
  loss. Existing `particle_number` information may be used later for optional
  auxiliary supervision.

This compatibility path cannot undo decisions already made during dataset
preprocessing. In particular, some particles may already have been merged, dropped,
or assigned a representation-dependent PID. The initial model will therefore learn
the same particle collection as the legacy model, but without using the stored
input-target alignment in its forward pass or loss. Producing truly pre-assignment
targets is a separate, later dataset revision.

## Proposed data flow

```text
X [B, N, F] + input_mask
        |
        v
scalable hit encoder
        |
        v
H [B, N, D]
        |
        +-------------------------------+
                                        |
learned particle queries [B, S, D]      |
        |                               |
        v                               |
cross-attention: queries attend to H <--+
slot self-attention + feed-forward
        |
        v
presence [B, S, 2], PID [B, S, C], momentum [B, S, 5]

compact targets [B, K, Y] + target_mask
        |
        v
Hungarian matching and set loss
```

Here `N` is the number of input hits, `K` is the event's target multiplicity, and
`S` is the configured maximum number of particle slots. Start with `S = 256`, then
choose the production value from the observed target multiplicity distribution. An
event with `K > S` must fail explicitly and must never be silently truncated.

For a local/block-sparse encoder with fixed neighborhood size `w`, the intended
complexity is approximately

```text
O(L_encoder * N * w + L_decoder * S * N + L_decoder * S^2).
```

The cross-attention implementation must use a memory-efficient SDPA/FlashAttention
path and must not materialize or retain the full attention matrix.

## Implementation plan

### 1. Build compact targets from existing datasets

In `TFDSDataSource.__getitem__`, before modifying `ytarget` in place:

1. Select rows whose particle class is nonzero.
2. Copy those rows to a new in-memory `ytarget_set` field.
3. Validate that every selected row has a unique, nonzero `particle_number` for
   dataset versions that provide it.
4. Apply absolute set-target transformations, independent of `X`. Candidate initial
   parameterization: `log(pt)`, `eta`, `sin(phi)`, `cos(phi)`, and `log(energy)`.
5. Continue applying the current input-relative transformations only to the legacy
   `ytarget` field.

Sorting or padding the input axis must not reorder or pad `ytarget_set` to the input
length.

Extend `PFBatch` and `Collater` so that `X` and `ytarget_set` are padded independently.
Expose both:

- `batch.mask`: valid input objects;
- `batch.target_mask`: valid compact target particles.

Keep the original `ytarget`, `ytarget_pt_orig`, and `ytarget_e_orig` fields intact for
legacy training and evaluation.

### 2. Add configuration for set prediction

Add an output mode orthogonal to the encoder type, conceptually:

```yaml
architecture:
  type: attention
  output_mode: set
  set_decoder:
    num_slots: 256
    num_layers: 2
    num_heads: 8
    query_init: learned
    cross_attention: flash
```

Requirements:

- Default `output_mode` to `elementwise`.
- Initially enable `set` only for `cld_hits` and `clic_hits`.
- Validate dimensions and require `num_slots > 0`.
- Do not overload the existing `task_queries` option; those queries are per-element
  task readouts, not output-particle slots.
- Give unsupported combinations, including initial ONNX export if necessary, clear
  configuration errors.

### 3. Implement the set decoder

Create `mlpf/model/set_decoder.py` and reuse/refactor `MLPF.encode_backbone()` as the
common encoder interface.

The initial decoder should contain:

- a learned bank of `S` query embeddings;
- two pre-normalized decoder layers;
- slots-to-input cross-attention;
- slot self-attention to coordinate and suppress duplicate predictions;
- a feed-forward block and residual connections;
- presence, PID, and absolute-momentum output heads.

Use packed variable-length Flash cross-attention where available so padded hits do
not consume decoder compute or memory. Provide a dense masked SDPA fallback for CPU
and unit tests. Do not request attention weights in the production path.

Keep the external prediction representation close to the existing one, but make its
particle axis the slot axis rather than the input axis.

### 4. Add Hungarian matching and set losses

Create `mlpf/model/set_losses.py`. For each event, construct a detached matching cost
between its valid slots and targets using:

- PID classification cost;
- distance in `log(pt)`;
- distance in `eta`;
- cyclic phi cost, for example `1 - cos(delta_phi)`;
- distance in `log(energy)`.

Use `scipy.optimize.linear_sum_assignment` initially. The matching problem is small
and independent of the number of input hits. Profile its CPU/GPU synchronization
overhead before considering a GPU matcher.

After matching, optimize:

- particle-presence loss over every slot;
- PID loss over matched slots;
- kinematic losses over matched slots only;
- no-particle targets for unmatched slots.

Normalize losses per event or per target particle so high-multiplicity events do not
dominate. Keep matcher cost weights separate from optimized loss weights. Reuse the
existing task-loss calibration only if its assumptions remain valid for the new loss
normalization.

### 5. Route training, validation, and inference

Route the model and loss in `training.py` according to `output_mode`, without changing
the elementwise path.

For set-mode inference:

- select active slots using the presence prediction;
- restore physical `pt` and energy without referring to an input object;
- unpad targets using `target_mask`;
- serialize predictions using the number of selected slots, not the input-hit count;
- keep serializing the complete input hit collection separately;
- build jets and MET from the selected particle slots.

Update diagnostic tables and particle-quality metrics so they no longer assume that
predictions, targets, and inputs share an axis.

### 6. Verify correctness and scalability

Add unit tests for:

- compact-target extraction from representative and `particle_number`-only rows;
- independent input and target padding;
- target permutation invariance;
- known Hungarian assignments;
- unmatched slots and zero-target events;
- `K = S` and explicit `K > S` failure;
- padding masks and finite mixed-precision forward/backward results;
- set output shapes being independent of `N`;
- unchanged legacy configuration, output shapes, and losses.

Add an integration test with approximately `N = 10,000`, `K = 100`, and `S = 256`,
including backward propagation.

Extend `scripts/benchmark.py` to sweep at least `N = 1k, 5k, 10k, 20k` and record:

- encoder, decoder, loss, forward, and backward time;
- peak allocated and reserved GPU memory;
- matcher time;
- valid input and target multiplicities.

The implementation should scale approximately linearly with `N` for fixed encoder
block size and fixed `S`. Profiling must confirm that no persistent `[B, S, N]`
attention tensor is allocated.

Compare elementwise and set prediction using:

- particle multiplicity, efficiency, and fake rate;
- PID confusion and performance versus `pt` and `eta`;
- particle response and resolution;
- jet matching, response, and resolution;
- MET response and resolution;
- summed event energy;
- training throughput and peak memory.

## TODO

### Data and batching

- [x] Add an `output_mode` or equivalent argument to the data-loading path.
- [x] Extract `ytarget_set` before legacy relative-target transformations.
- [x] Add uniqueness and consistency checks using `particle_number`.
- [x] Define and test the absolute set-target transformation and its inverse.
- [x] Add `ytarget_set` and `target_mask` to `PFBatch`.
- [x] Pad input and target collections independently in `Collater`.
- [x] Verify compact target counts, PIDs, and summed energy against legacy nonzero
      target rows on existing CLD/CLIC hit samples.

### Configuration and model

- [x] Add `output_mode` configuration with backward-compatible defaults.
- [x] Add a validated `SetDecoderConfig`.
- [x] Add a hit-dataset set-mode example to `particleflow_spec.yaml`.
- [x] Refactor/reuse the backbone encoder without changing legacy forward behavior.
- [x] Implement learned fixed queries and two decoder layers.
- [ ] Implement memory-efficient packed cross-attention plus a CPU test fallback.
- [x] Implement presence, PID, and absolute-momentum heads.
- [x] Assert target-slot overflow instead of truncating; add aggregate logging later.

### Matching and loss

- [x] Implement per-event Hungarian matching.
- [x] Implement matching costs with cyclic phi handling; expose them in model configuration later.
- [x] Implement matched presence, PID, and regression losses.
- [x] Decide and test event/particle normalization and no-particle weighting.
- [x] Integrate calibrated task-loss weighting for set mode.
- [ ] Log target count, active-slot count, matched cost, and unmatched-slot statistics.

### Training and inference

- [x] Route training and validation loss calculation by output mode.
- [x] Log scheme-independent particle matching, count, PID, kinematic, and event
      closure metrics during validation.
- [ ] Update validation diagnostics for independent axes.
- [x] Update `predict_particles` for set outputs and absolute inverse transforms.
- [x] Update parquet inference serialization to use separate input, target, and
      prediction counts.
- [ ] Update particle, jet, and MET metrics for set outputs.
- [ ] Add an explicit error or support path for set-mode ONNX export.

### Testing and performance

- [x] Add compact-target extraction and batching tests.
- [x] Add matcher and target-permutation tests.
- [x] Add decoder masking, shape, and numerical-stability tests.
- [x] Run the existing legacy model and loss regression tests.
- [x] Add a 10k-hit forward/backward integration test.
- [ ] Extend the benchmark script with set-mode timing and memory measurements.
- [x] Run a small CLD-hits overfit test and confirm that loss and matching converge.
- [x] Add a local ttbar launcher for paired elementwise and set-output training.
- [x] Add reusable seeded comparison scenarios with local, Tallinn, LUMI, and
      Flatiron hardware profiles.
- [x] Keep partial validation batches in distributed runs so small `nvalid`
      samples do not produce zero per-rank batches.
- [ ] Run a short CLD-hits training comparison against the elementwise baseline.
- [x] Document initial correctness, timing, memory, and scaling measurements here;
      add physics accuracy after training.

## Initial implementation measurements

Measurements from 2026-09-04 using the existing CLD `cld_edm_ttbar_hits/1:3.2.1`
dataset:

- The first 50 events contain 4,388--13,219 valid hits and 33--148 compact target
  particles. Compact target counts matched the number of nonzero legacy target rows
  in every event.
- A real event with 6,423 valid hits, 52 targets, and 256 slots completed the full
  four-layer HEPTv2 forward pass, Hungarian loss, and backward pass on an NVIDIA
  GeForce RTX 5060 Ti. Peak allocated GPU memory was approximately 0.70 GB.
- A real event with 13,219 valid hits, 135 targets, and 256 slots completed the same
  path with approximately 1.39 GB peak allocated GPU memory.
- A 20-step single-event AdamW overfit check with the full HEPTv2 encoder reduced the
  uncalibrated total loss from 41.04 to 16.49, confirming end-to-end gradients through
  the encoder, decoder, and matched loss.
- The current comparison recipes use a three-layer packed attention backbone. Both
  elementwise and set modes completed a GPU forward-pass smoke test on the 6,423-hit
  event, using approximately 0.10 GB and 0.08 GB of allocated GPU memory respectively.
- `uv run pytest -q tests` passed 241 tests with 3 skips. Running pytest from the
  repository root without restricting collection still encounters two unrelated
  pre-existing collection errors under `baselines/HEPTv2` and `scripts/legacy`.

## Follow-up work after the initial baseline

- [ ] Add input-conditioned dynamic queries. Charged-particle queries may use
      innermost tracker-hit candidates; neutral-particle queries will need a separate
      calorimeter seeding strategy.
- [ ] Evaluate phi-sector decoding and boundary behavior.
- [ ] Evaluate local strided/block-sparse cross-attention if dense Flash
      cross-attention remains a material compute cost.
- [ ] Add optional encoder contrastive/background supervision from
      `particle_number` without making it necessary for set prediction.
- [ ] Add optional chunked or local slot-to-hit assignment heads for interpretability
      and auxiliary losses.
- [ ] If the baseline is successful, generate a new dataset version containing a
      truly independent pre-assignment particle collection and corresponding target
      jets.
