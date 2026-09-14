# Hit-based datasets

Hit-based inputs are a research workflow for CLD and CLIC. Begin with the track-and-cluster datasets in the [catalog](catalog.md) unless raw tracker and calorimeter measurements are essential to the study.

## What changes

| Property | Track/cluster (`*_pf`) | Hit (`*_hits`) |
|---|---|---|
| Input elements | Reconstructed tracks and calorimeter clusters | Tracker hits and calorimeter hits |
| TFDS task | `pixi run tfds` | `pixi run tfds_hit` |
| CLD model | `pyg-cld-v1` | `pyg-cld-hits-v1` |
| CLIC model | `pyg-clic-v1` | `pyg-clic-hits-v1` |
| Baseline candidates | Stored in `ycand` | No hit-level baseline; `ycand` is zero-filled |
| Operational status | Supported default | Research workflow |

Both representations are derived from the same validated Key4HEP Parquet files. Do not run a second detector simulation merely to build hit TFDS.

The [catalog](catalog.md) is the single inventory of available hit datasets, versions, and compatible model recipes. The hit builder concatenates tracker and calorimeter hits into `X`, with the element type and geometry encoded in the feature metadata. Targets are aligned with those input elements. Events without usable inputs or targets are skipped. Generator missing momentum and generator/target jets remain event-level fields.

## Production and validation

Follow [CLD and CLIC production](key4hep.md) through strict Parquet validation, then build the hit representation:

```bash
PROD=cld pixi run tfds_hit
```

Verify the exact TFDS name, configuration, and version. For example, after downloading or producing CLD `ttbar` configuration 1:

```bash
uv run python -c 'import tensorflow_datasets as tfds; b = tfds.builder("cld_edm_ttbar_hits/1:3.2.1", data_dir="/path/to/cld-workspace/tfds"); e = b.as_data_source(split="train")[0]; print(b.info.full_name, e["X"].shape, e["ytarget"].shape)'
```

Success requires the same number of input and target rows. Also review the hit-geometry and hit-representation gates in the Parquet validation report; TFDS decoding alone cannot establish those relationships.

## Resource expectations

Hit events contain many more elements than track/cluster events. This increases TFDS size, data-loading pressure, attention memory, padding waste, and training time. The repository does not publish a universal hardware minimum because event sizes and model settings vary.

Before a large run:

1. inspect element-count distributions from representative Parquet files;
2. build and decode one TFDS configuration;
3. run a short training with the dedicated hit recipe;
4. measure peak host and accelerator memory;
5. scale workers, open-reader limits, padding, and batch accumulation from that measurement.

Do not reuse the much larger `gpu_batch_multiplier` values from the track/cluster recipes. The checked-in hit recipes deliberately use smaller accumulation settings and are the starting point for the matching detector.

## Mixed representations

The repository can interleave dataset sources for research studies, but PF-object and hit inputs have different feature semantics and memory behavior. Combining them is not a beginner path: pin every source version, use an architecture/configuration that explicitly supports the input dimensions, and validate per-source sampling and losses. A successful concatenation is not evidence that the target definitions are scientifically equivalent.
