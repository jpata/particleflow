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
data/tfds/tensorflow_datasets/cld/cld_edm_ttbar_pf/1/3.3.0/
```

Pass the detector-level directory containing the dataset-name directories as `--data-dir`; for this example it is `data/tfds/tensorflow_datasets/cld`.

## Published availability

The [Hub](https://huggingface.co/datasets/jpata/particleflow/tree/main/tensorflow_datasets) publishes configuration 1 of each dataset. As checked on 25 November 2026, it has these versions:

| Representation | Detector | Published datasets | Version |
|---|---|---|---|
| Tracks and clusters | CLD | `cld_edm_{ttbar,ww_fullhad,qq,zz}_pf` | 3.3.0 |
| Tracks and clusters | CLICdet | `clic_edm_{ttbar,ww_fullhad,qq}_pf` | 3.3.0 |
| Tracks and clusters | IDEA | `idea_edm_{ttbar,ww_fullhad,qq}_pf` | 0.3.0 |
| Hits | CLD | `cld_edm_{ttbar,ww_fullhad,qq,zz}_hits` | 3.2.2 |
| Hits | CLICdet | `clic_edm_{ttbar,ww_fullhad,qq}_hits` | 3.2.2 |

The current recipes use the [v1.4.0 campaign](key4hep-50k.md) versions instead: CLD/CLIC track/cluster **3.4.0**, CLD/CLIC hits **3.3.0**, IDEA **0.5.0** (measured GGTF/GenFit tracks) and ILD **0.2.0**. The Hub entries above describe the previous campaigns.

Also not yet published at the recipe version:

- ColliderML tracks and clusters: the Hub has 1.4.0; recipes expect 1.5.0, which drops out-of-acceptance truth particles. ColliderML hits are not published.
- CMS and MAIA datasets are not on the Hub.

The Hub also holds older versions, and some v1.4.0-campaign datasets (for example CLD/CLIC 3.4.0 and ILD) may already be present; check the live tree. Use the version your recipe names; do not rename a dataset directory to a different version.

The `tfds_mapping`, `tfds_hit_mapping`, and model dataset entries in [`particleflow_spec.yaml`](https://github.com/jpata/particleflow/blob/main/particleflow_spec.yaml) define what the current code expects. For scientific provenance and citations, use the [publication map](../science/publications.md).
