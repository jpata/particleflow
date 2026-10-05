# CLD PF/hits comparison: heuristic PF, learned PF, elementwise hits, set hits

This study compares four particle-flow reconstructions on CLD, using the
8-10 September 2026 `cld_pf_hits_comparison` campaign under
`experiments/cld_pf_hits_comparison/`:

- **Heuristic PF** (`ycand`): the fixed, non-ML CLD particle-flow
  reconstruction, evaluated directly from the non-hits track/cluster TFDS
  test split. Not a trained model and has no training step.
- **Learned PF (tracks+clusters)**: an attention-based MLPF model trained on
  the non-hits track/cluster TFDS (`cld_edm_{ttbar,ww_fullhad,qq}_pf`),
  archived from `pf_seed12345_20260908_164627_589652`.
- **Elementwise (hits)**: the same backbone family trained directly on
  tracker/calorimeter hits (`cld_edm_{ttbar,ww_fullhad,qq}_hits`) with a
  per-hit elementwise particle-flow output, archived from
  `elementwise_hits_seed12345_20260908_164625_578183`.
- **Set (hits)**: the hit-based backbone with a 256-slot set-prediction
  decoder (input-conditioned queries, 0.4 local attention radius, cardinality
  and auxiliary matching losses), archived from
  `set_hits_seed12345_20260908_164625_682170`.

## Why this campaign is a cleaner comparison than earlier studies

All three trained runs share seed 12345, a global event batch of 512, a
40,000-step cosine schedule, and 8x NVIDIA H100 PCIe hardware, and **all three
completed the full 40,000 steps** ("Training completed" in each rank-zero
log). This is the first hit-vs-PF-input comparison in this series where every
run reaches the same, complete training length; earlier studies
(`20260905_hit_output_comparison`) compared runs at different steps or with
incomplete schedules.

## Comprehensive jet and particle metrics

This campaign trained after `mlpf/jet_utils.py` and
`mlpf/model/validation_metrics.py` were extended (commit `125a5a58`) to
record, at every validation step:

- comprehensive jet metrics (`mlpf.jet_utils.jet_matching_metrics`): angular
  precision/recall/F1/fake-rate from the one-to-one Hungarian jet assignment,
  plus a response-qualified precision/recall/F1 that additionally requires
  the matched jet's pT ratio to fall within 0.5 of unity, alongside the
  previously recorded median response, response IQR, and match fraction;
- comprehensive particle metrics
  (`mlpf.model.validation_metrics.compute_validation_particle_metrics`):
  scheme-independent Hungarian particle matching (efficiency, purity, F1,
  duplicate fraction), per-class efficiency/PID accuracy, and event-level
  closure (energy response, scalar-pT response, vector-pT closure, MET
  absolute error).

`eval_pf_baseline.py` in this study computes the identical two metric sets
for the heuristic PF baseline, calling the exact same
`jet_matching_metrics` and `compute_validation_particle_metrics` functions the
training pipeline uses, so the heuristic reference is on equal footing with
the three trained runs rather than reduced to a jet-only reference line as in
earlier studies.

## What is archived under `inputs/`

For each of `pf/`, `elementwise_hits/`, and `set_hits/`:

- the complete history JSON for all 8 checkpoints (steps 5,000-40,000);
- the rank-zero training log, copied from `train.log.0` (the launcher's own
  `train.log` is a short pre-spawn stub, not the full per-step log) and
  renamed to `train.log`;
- `tensorboard/train/` and `tensorboard/valid/`, copied separately from the
  run's `runs/train/` and `runs/valid/` writers;
- `train-config.yaml`, `hyperparameters.json`, `scenario-manifest.json`, and
  `particleflow_spec.yaml`;
- `plots_step_40000/`, the complete final-checkpoint plot directory for all
  three samples (ttbar, WW to 4q, qq).

Checkpoints, model pickle files, and the `batch*`/`preds_step_*` prediction
parquet files are intentionally excluded (multi-GB per run and not needed by
this notebook). Plot directories for intermediate checkpoints (5,000-35,000)
are also excluded; only the final, complete step-40,000 plots are archived.

`inputs/set_hits_diagnostics/per_layer.csv` is copied from
`notebooks/studies/20260911_set_hits_diagnostics/output/slots/per_layer.csv`.
It holds the per-decoder-layer particle metrics of the set (hits) model
(each layer's auxiliary-loss head read out with presence >= 0.5 and scored
with the validation matching on 1,000 ttbar test events) and feeds the
"Set decoder depth" slide. Regenerating it requires the GPU studies
documented in that follow-up study.

`inputs/heuristic_pf_baseline.json` holds the full output of
`eval_pf_baseline.py`, including per-sample jet metrics and per-sample and
pooled ("combined") particle metrics, so the notebook does not need to
recompute the heuristic reference or re-read the underlying TFDS.

## Regenerating the heuristic PF baseline

`eval_pf_baseline.py` reads the non-hits `_pf` TFDS (CLD 3.2.1, splits 1-10)
directly, independent of any trained MLPF model or of the campaign's own
`data_dir`. The `pf` training run was configured against
`/mnt/ceph/users/jpata/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds`, which is not
mounted on this machine; the script instead reads an equivalent local copy of
the same dataset version from `/mnt/work/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds`.
Regenerating the baseline therefore requires that path, or an equivalent
local copy, to be present. Run it with:

```bash
uv run python notebooks/studies/20260911_cld_pf_hits_comparison/eval_pf_baseline.py
```

By default it evaluates 52 events per split x 10 splits = 520 events per
sample, approximating (but not exactly matching) the 512-event test scope
used by the trained campaign. The full per-sample dict, plus a "combined"
entry pooling particle metrics across all three samples (for comparison
against the trained runs' single interleaved-validation particle-metric
value), is written to `inputs/heuristic_pf_baseline.json`. Recorded jet
numbers are reproduced in the script's own docstring.

## Slide narrative

The deck is organised as setup first, results second:

1. Question and the four formulations; inputs and targets; model
   architecture; loss functions (elementwise vs set, including the
   Hungarian matcher cost, cardinality and auxiliary terms, and the fixed
   task-weight calibration); training setup (LAMB, cosine schedule, batch,
   sampler, validation/test cadence); and how every metric family is
   computed and aggregated (per-event step, pooled dataset-level ratio vs
   mean over events, and which sample it uses).
2. Part 1, training behaviour: walltime and GPU memory, elementwise vs set
   loss curves, calibrated task weights.
3. Part 2, particle-level results on the 512-event validation mix:
   matching, multiplicity, step-40,000 table, per-class efficiency and PID,
   event closure.
4. Part 3, jet-level results on 512 test events per process: comprehensive
   jet metrics per process, final table, physics-validation plots.
5. Set decoder depth (per-layer read-out from the follow-up diagnostics
   study) and conclusions.

The setup slides are derived from `mlpf/model/losses.py`,
`mlpf/model/set_losses.py`, `mlpf/model/validation_metrics.py`,
`mlpf/jet_utils.py`, `mlpf/model/utils.py` (schedule), and the archived
`hyperparameters.json` / `train-config.yaml` of each run.

## Rendering the slides

From the repository root:

```bash
notebooks/studies/20260911_cld_pf_hits_comparison/render.sh
```

This produces an executed notebook, PNG/PDF/SVG figures under `output/`,
`output/hit_output_comparison.slides.html`, and
`output/hit_output_comparison.slides.pdf`. Notebook code is omitted from both
slide exports (tagged `hide-input`). The HTML presentation uses Reveal.js
from a public CDN when viewed. PDF rendering requires Google Chrome or
Chromium.

## Known comparison limitations

- Each configuration uses one seed and one training run; no repeated-seed
  variance estimate is available.
- The heuristic PF reference pools 520 events per sample (1,560 events total)
  independent of any trained model, while the three trained runs' particle
  metrics come from an interleaved 512-event validation mix across all three
  samples and their jet metrics come from 512 test events per sample. Counts
  are close but not identical.
- The "Learned PF" run's raw per-component training losses are on a
  different numeric scale than the two hit-based runs (different backbone,
  element population, and independently calibrated fixed task weights), so
  this study does not plot its raw losses alongside elementwise/set; only the
  physics-level (particle and jet) metrics are compared across all three
  trained runs and the heuristic reference.
- `gpu_type` in each run's `train-config.yaml` reads `l40`, a scheduling
  default; the actual assigned hardware, taken from each rank-zero log, is
  8x NVIDIA H100 PCIe for all three runs.
