# Hit-based datasets

Hit-based inputs are a research workflow for CLD and CLIC. The track-and-cluster datasets in the [catalog](catalog.md) provide the default path; studies of raw tracker and calorimeter measurements use the hit workflow described here.

## What changes

| Property | Track/cluster (`*_pf`) | Hit (`*_hits`) |
|---|---|---|
| Input elements | Reconstructed tracks and calorimeter clusters | Tracker hits and calorimeter hits |
| TFDS task | `pixi run tfds` | `pixi run tfds_hit` |
| CLD model | `pyg-cld-v1` | `pyg-cld-hits-v1` |
| CLIC model | `pyg-clic-v1` | `pyg-clic-hits-v1` |
| Baseline candidates | Stored in `ycand` | `ycand` is a zero-filled placeholder |
| Operational status | Supported default | Research workflow |

Both representations are derived from the same validated Key4HEP Parquet files. The hit builder combines tracker and calorimeter hits into the input array `X`. The feature metadata identifies each element's type and geometry, and `ytarget` contains the corresponding particle target for each input row. Generator missing momentum and generator/target jets remain event-level fields.

## Event views

Raw detector hits and target particles for the same representative `ttbar`
event are shown in the transverse plane. Tracker hits are red, ECAL hits blue,
HCAL hits green, and muon-system hits orange.

### CLD

| Raw hits | Target particles |
|:---:|:---:|
| ![CLD tracker, calorimeter, and muon-system hits.](../../notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/inputs/event_displays/cld_event_5_hits_5x5cm.svg) | ![CLD target particles.](../../notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/inputs/event_displays/cld_event_5_targets_5x5cm.svg) |

### CLICdet

| Raw hits | Target particles |
|:---:|:---:|
| ![CLIC tracker, calorimeter, and muon-system hits.](../../notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/inputs/event_displays/clic_event_5_hits_5x5cm.svg) | ![CLIC target particles.](../../notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/inputs/event_displays/clic_event_5_targets_5x5cm.svg) |

The [catalog](catalog.md) lists the available hit datasets, versions, and model recipes.

## Production and validation

Follow [shared Key4HEP production](key4hep.md) through strict Parquet validation, then build the hit representation:

```bash
PROD=cld pixi run tfds_hit
```

Verify the exact TFDS name, configuration, and version. For example, after producing CLD `ttbar` configuration 1:

```bash
uv run python - <<'PY'
import tensorflow_datasets as tfds

builder = tfds.builder(
    "cld_edm_ttbar_hits/1:3.2.1",
    data_dir="/path/to/cld-workspace/tfds",
)
event = builder.as_data_source(split="train")[0]
print(builder.info.full_name, event["X"].shape, event["ytarget"].shape)
PY
```

Success requires the same number of input and target rows. The hit-geometry and hit-representation gates in the Parquet validation report establish the corresponding detector relationships.

## Resource expectations

Hit events contain many more elements than track/cluster events. This increases TFDS size, data-loading pressure, attention memory, padding waste, and training time. Hardware requirements therefore depend on the event-size distribution and model settings.

Before a large run:

1. inspect element-count distributions from representative Parquet files;
2. build and decode one TFDS configuration;
3. run a short training with the dedicated hit recipe;
4. measure peak host and accelerator memory;
5. scale workers, open-reader limits, padding, and batch accumulation from that measurement.

Start with the smaller `gpu_batch_multiplier` values in the checked-in hit recipe for the matching detector, then tune them from measured memory use.
