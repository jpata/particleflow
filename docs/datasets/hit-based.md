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

Both representations are derived from the same validated Key4HEP Parquet files. Reuse those files to build the hit TFDS.

The [catalog](catalog.md) is the single inventory of available hit datasets, versions, and compatible model recipes. The hit builder concatenates tracker and calorimeter hits into `X`, with the element type and geometry encoded in the feature metadata. Targets are aligned with those input elements. The builder retains events with usable inputs and targets. Generator missing momentum and generator/target jets remain event-level fields.

## Production and validation

Follow [CLD and CLIC production](key4hep.md) through strict Parquet validation, then build the hit representation:

```bash
PROD=cld pixi run tfds_hit
```

Verify the exact TFDS name, configuration, and version. For example, after downloading or producing CLD `ttbar` configuration 1:

```bash
uv run python -c 'import tensorflow_datasets as tfds; b = tfds.builder("cld_edm_ttbar_hits/1:3.2.1", data_dir="/path/to/cld-workspace/tfds"); e = b.as_data_source(split="train")[0]; print(b.info.full_name, e["X"].shape, e["ytarget"].shape)'
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

## Mixed representations

The repository can interleave dataset sources for research studies. PF-object and hit inputs have different feature semantics and memory behavior, so combining them requires advanced configuration: pin every source version, use an architecture/configuration that explicitly supports the input dimensions, and validate per-source sampling and losses. Establish scientific equivalence through an explicit comparison of the target definitions.
