# 20260907 campaign input archive

This directory contains immutable snapshots used by
`hit_output_comparison.ipynb`.

| Directory | Source experiment | Selected step | Role |
|---|---|---:|---|
| `elementwise_40k/` | `elementwise_seed12345_20260907_090315_620002` | 40,000 | Production elementwise endpoint |
| `set_query_origin_20k/` | `set_seed12345_20260907_090321_871639` | 20,000 | Production set endpoint with query-origin loss |
| `set_query_origin_smoke_2k/` | `set_seed12345_20260907_132200_683525` | 2,000 | Local code-path smoke test |

For each run, `history/` contains every history JSON available when archived.
`tensorboard/train/` and `tensorboard/valid/` keep the two writers separate.
`train.log` is the rank-zero log. The selected `plots_step_*` directory is
copied in full.

The production set job was configured for 40,000 steps. Its archived history
and plots end at the complete 20,000-step checkpoint; the log continues to
step 20,400 without a completion record. The smoke run uses only CLD split 10
and is not included in the production result plots.

Large derived artifacts are intentionally omitted: checkpoints,
`model_kwargs.pkl`, `batch0_epoch*.parquet`, `preds_step_*`, and plots from
unselected checkpoints.
