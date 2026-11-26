# Dataset catalog

Use this page to choose a detector, input representation, dataset name, and compatible model recipe. The [download guide](download.md) covers published data, while the [generation guide](generate.md) covers custom data production.

## Choose a representation, then a detector and sample

Start with the input representation. Each detector page lists its physics samples, exact TFDS names, schema versions, and compatible model recipes. Pileup conditions distinguish samples within a detector.

| Input representation | Detector or dataset family | Typical use |
|---|---|---|
| [Tracks and clusters](track-cluster.md) | [CMS](cms.md), [CLD](cld.md), [CLICdet](clic.md), [ColliderML](colliderml.md), [MAIA](maia.md), [IDEA](idea.md), [ILD@FCC-ee](ild.md) | Reconstructed tracks and calorimeter clusters; the default starting point for CMS, CLD, and CLIC. |
| [Hits](hit-based.md) | [CLD](cld-hits.md), [CLICdet](clic-hits.md), [ColliderML](colliderml-hits.md) | Tracker and calorimeter measurements for dedicated hit-model research. |

The catalog includes maintained recipes and explicitly identified partial workflows. A builder's existence does not imply a trained model, physics validation, or public download. Detector pages identify those boundaries. Dataset schema, Python package, and model artifact versions are independent.

After selecting a sample, [download and verify](download.md) published data or follow [dataset generation](generate.md). Full productions and unpublished versions are available to Grid users through [WebDAV](webdav.md). The shared [Key4HEP production guide](key4hep.md) covers both CLD/CLIC representations.

See [detector event views](event-displays.md) for reusable visualizations of the
input representations and reference particles.

## Configuration partitions and event splits

CMS, CLD, CLIC, and ColliderML recipes use configuration partitions `1` through `10`; IDEA and ILD@FCC-ee use configuration `1`. A configuration partition contains its own `train` and `test` event splits: the number identifies the partition, while `train` or `test` identifies the event split. The standard ten-partition model recipes combine all ten partitions. Configuration `1` provides a one-partition subset for a small experiment; the configured training corpus spans all ten.

The on-disk layout is:

```text
<data root>/<dataset name>/<configuration>/<version>/
```

For example:

```text
data/tfds/tensorflow_datasets/cld/cld_edm_ttbar_pf/1/3.4.0/
```

Pass the detector-level directory containing the dataset-name directories as `--data-dir`; for this example it is `data/tfds/tensorflow_datasets/cld`.

## Published availability

The [Hub](https://huggingface.co/datasets/jpata/particleflow/tree/main/tensorflow_datasets) publishes configuration 1 of each dataset. As checked on 26 November 2026 ([revision `b024304b`](https://huggingface.co/datasets/jpata/particleflow/tree/b024304b5dd9e47e9d71d7e51ad77029e5002d8d/tensorflow_datasets)), these recipe versions are published:

| Representation | Detector | Published datasets | Version | Events (train / test) |
|---|---|---|---|---|
| Tracks and clusters | CLD | `cld_edm_{ttbar,ww_fullhad,qq,z_qq,z_tautau}_pf` | 3.4.0 | about 4,500 / 500 per sample |
| Tracks and clusters | CLICdet | `clic_edm_{ttbar,ww_fullhad,qq,z_qq,z_tautau}_pf` | 3.4.0 | about 4,500 / 500 per sample |
| Tracks and clusters | IDEA | `idea_edm_{ttbar,ww_fullhad,qq,z_qq,z_tautau}_pf` | 0.5.0 | about 8,000 / 2,000 per sample |
| Tracks and clusters | ILD@FCC-ee | `ild_edm_{ttbar,ww_fullhad,qq,z_qq,z_tautau}_pf` | 0.2.0 | about 45,000 / 5,000 per sample |

These come from the [Key4hep v1.4.0 production](key4hep-50k.md). For CLD and CLICdet, the production has ten configurations; only configuration 1 is on the Hub. Some samples have slightly fewer events than nominal, because postprocessing drops events it cannot convert.

Not yet published at the recipe version:

| Selection | Recipe version | Newest on the Hub |
|---|---|---|
| CLD and CLICdet hits | 3.3.0 | 3.2.2, for `ttbar`, `WW`, `qq` and CLD `ZZ` |
| ColliderML tracks and clusters | 1.5.0 | 1.4.0, for the no-pileup and pileup-200 ttbar samples; 1.5.0 drops out-of-acceptance truth particles |
| ColliderML hits | 1.3.0 | Not published |
| CMS, MAIA | | Not published |

The Hub also keeps older versions of every dataset, such as CLD and CLICdet 3.3.0 and IDEA 0.3.0, and CLD `ZZ` up to 3.3.0. The repository's `root/` and `parquet/` directories hold the reconstructed and postprocessed files of the v1.4.0 samples; `parquet/` also has the ColliderML ttbar samples. Use the version your recipe names, and do not rename a dataset directory to a different version.

The `tfds_mapping`, `tfds_hit_mapping`, and model dataset entries in [`particleflow_spec.yaml`](https://github.com/jpata/particleflow/blob/main/particleflow_spec.yaml) define what the current code expects. For scientific provenance and citations, use the [publication map](../science/publications.md).
