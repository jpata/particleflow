# Dataset catalog

Use this page to choose a detector, input representation, dataset name, and compatible model recipe. The [download guide](download.md) covers published data, while the [generation guide](generate.md) covers custom data production.

## Choose a representation, then a detector and sample

Start with the input representation. Each detector page lists its physics samples, exact TFDS names, schema versions, and compatible model recipes. Pileup conditions distinguish samples within a detector.

| Input representation | Detector or dataset family | Typical use |
|---|---|---|
| [Tracks and clusters](track-cluster.md) | [CMS](cms.md), [CLD](cld.md), [CLICdet](clic.md), [ColliderML](colliderml.md), [MAIA](maia.md), [IDEA](idea.md) | Reconstructed tracks and calorimeter clusters; the default starting point for CMS, CLD, and CLIC. |
| [Hits](hit-based.md) | [CLD](cld-hits.md), [CLICdet](clic-hits.md), [ColliderML](colliderml-hits.md) | Tracker and calorimeter measurements for dedicated hit-model research. |

The catalog includes maintained recipes and explicitly identified partial workflows. A builder's existence does not imply a trained model, physics validation, or public download. Detector pages identify those boundaries. Dataset schema, Python package, and model artifact versions are independent.

After selecting a sample, [download and verify](download.md) published data or follow [dataset generation](generate.md). The shared [Key4HEP production guide](key4hep.md) covers both CLD/CLIC representations.

## Configuration partitions and event splits

CMS, CLD, CLIC, MAIA, and ColliderML recipes use configuration partitions `1` through `10`; IDEA uses configuration `1`. A configuration partition contains its own `train` and `test` event splits: the number identifies the partition, while `train` or `test` identifies the event split. The standard ten-partition model recipes combine all ten partitions. Configuration `1` provides a one-partition subset for a small experiment; the configured training corpus spans all ten.

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

As checked on **28 October 2026**, the [Hub snapshot](https://huggingface.co/datasets/jpata/particleflow/tree/d03df152669939dae26336f68de7c0af575bb12d/tensorflow_datasets) includes metadata, feature definitions, and all declared split shards for these current-recipe selections:

| Representation | Detector | Published datasets | Version | Configuration |
|---|---|---|---|---|
| Tracks and clusters | CLD | `cld_edm_{ttbar,ww_fullhad,qq,zz}_pf` | 3.3.0 | 1 |
| Tracks and clusters | CLICdet | `clic_edm_{ttbar,ww_fullhad,qq}_pf` | 3.3.0 | 1 |
| Tracks and clusters | ColliderML | `colliderml_ttbar_nopu_pf`, `colliderml_ttbar_pu200_pf` | 1.2.0 | 1 |
| Tracks and clusters | IDEA | `idea_edm_{ttbar,ww_fullhad,qq}_pf` | 0.2.0 | 1 |
| Hits | CLD | `cld_edm_{ttbar,ww_fullhad,qq}_hits` | 3.2.2 | 1 |
| Hits | CLICdet | `clic_edm_{ttbar,ww_fullhad,qq}_hits` | 3.2.2 | 1 |

The [download guide](download.md) supplies the CLD 3.3.0 inputs required by
[the training guide](../training/train.md). Only configuration `1` is published
for the selections above; use `--data_config 1` rather than the standard
ten-partition training selection. Produce additional partitions with
[the generation workflow](generate.md).

Treat the [live Hub tree](https://huggingface.co/datasets/jpata/particleflow/tree/main/tensorflow_datasets) as the source of truth for publication availability. The `tfds_mapping`, `tfds_hit_mapping`, and model dataset entries in [`particleflow_spec.yaml`](https://github.com/jpata/particleflow/blob/main/particleflow_spec.yaml) define what the current code expects. A usable dataset has a detector, name, configuration, and version that match in both contexts.

For scientific provenance and citations, use the [publication map](../science/publications.md) and the dataset record associated with the version you consume.
