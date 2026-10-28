# Dataset catalog

Use this page to choose a detector, input representation, dataset name, and compatible model recipe. The [download guide](download.md) covers published data, while the [generation guide](generate.md) covers custom data production.

## Choose an input representation

Track-and-cluster datasets (`*_pf`) are the supported default and contain substantially fewer input elements per event. Choose a hit dataset (`*_hits`) for studies that need tracker and calorimeter hits. Hit datasets use dedicated model recipes and require more memory; see [Hit-based datasets](hit-based.md).

## Current recipes

The table describes recipes in `particleflow_spec.yaml` as of 23 October 2026. Dataset versions identify the data schema and production. The Python package and model checkpoints have their own versions, so record all three in derived work.

| Detector | Input representation | Dataset names | Recipe version | Compatible model recipe |
|---|---|---|---|---|
| CMS Run 3, pileup | Tracks and calorimeter clusters | `cms_pf_ttbar`, `cms_pf_qcd`, `cms_pf_ztt` | 3.2.0 | `pyg-cms-v1` |
| CMS Run 3, no pileup | Tracks and calorimeter clusters | `cms_pf_ttbar_nopu`, `cms_pf_qcd_nopu`, `cms_pf_ztt_nopu` | 3.2.0 | `pyg-cms-v1` |
| CLD, 365 GeV | Tracks and calorimeter clusters | `cld_edm_ttbar_pf`, `cld_edm_ww_fullhad_pf`, `cld_edm_qq_pf`, `cld_edm_zz_pf` | 3.3.0 | `pyg-cld-v1` (defaults: `ttbar`, `WW`, `qq`) |
| CLD, 365 GeV | Tracker and calorimeter hits | `cld_edm_ttbar_hits`, `cld_edm_ww_fullhad_hits`, `cld_edm_qq_hits`, `cld_edm_zz_hits` | 3.2.2 | `pyg-cld-hits-v1` (defaults: `ttbar`, `WW`, `qq`) |
| CLICdet, 380 GeV | Tracks and calorimeter clusters | `clic_edm_ttbar_pf`, `clic_edm_ww_fullhad_pf`, `clic_edm_qq_pf` | 3.3.0 | `pyg-clic-v1` |
| CLICdet, 380 GeV | Tracker and calorimeter hits | `clic_edm_ttbar_hits`, `clic_edm_ww_fullhad_hits`, `clic_edm_qq_hits` | 3.2.2 | `pyg-clic-hits-v1` |
| ColliderML Release 1, no pileup | Tracks and spatial calorimeter clusters | `colliderml_ttbar_nopu_pf` | 1.2.0 | `pyg-colliderml-v1` |
| ColliderML Release 1, pileup ~200 | Tracks and spatial calorimeter clusters | `colliderml_ttbar_pu200_pf` | 1.2.0 | `pyg-colliderml-pu200-v1` |
| ColliderML Release 1, no pileup | Tracker and calorimeter hits | `colliderml_ttbar_hits` | 1.0.0 | `pyg-colliderml-hits-v1`, `pyg-colliderml-hits-set-v1` |
| MAIA | Tracks and calorimeter clusters | `maia_edm_ttbar_pf` | 1.1.0 | `pyg-maia-v1` |
| IDEA | Truth-seeded proxy tracks and calorimeter clusters | `idea_edm_{ttbar,ww_fullhad,qq}_pf` | 0.2.0 | `pyg-idea-pipeline-v1` |

CLD hit-set studies use `pyg-cld-hits-heptv2-set-v1`; see the
[current evidence and recommendation](../cld_hit_results.md). ColliderML also has
a pileup-200 hit builder, but no corresponding named training recipe in the
current specification.

IDEA is a pipeline-validation workflow with truth-seeded proxy tracks and an
oracle reference. Its model recipe still selects `0.2.0`, while the builders
default to `0.2.1` (calorimeter-cell shower widths). Align the recipe with the
actual rebuilt data before training. The historical `0.1.0` sample inventory is
recorded in the [local comparison runbook](../validation/detector-comparison.md).

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

| Detector | Published datasets | Version | Configuration |
|---|---|---|---|
| CLD | `cld_edm_{ttbar,ww_fullhad,qq,zz}_pf` | 3.3.0 | 1 |
| CLD | `cld_edm_{ttbar,ww_fullhad,qq}_hits` | 3.2.2 | 1 |
| CLICdet | `clic_edm_{ttbar,ww_fullhad,qq}_pf` | 3.3.0 | 1 |
| CLICdet | `clic_edm_{ttbar,ww_fullhad,qq}_hits` | 3.2.2 | 1 |
| ColliderML | `colliderml_ttbar_nopu_pf`, `colliderml_ttbar_pu200_pf` | 1.2.0 | 1 |
| IDEA | `idea_edm_{ttbar,ww_fullhad,qq}_pf` | 0.2.0 | 1 |

The [download guide](download.md) now supplies the CLD 3.3.0 inputs required by
[the training guide](../training/train.md). Only configuration `1` is published
for the selections above; use `--data_config 1` rather than the standard
ten-partition training selection. Produce additional partitions with
[the generation workflow](generate.md).

Treat the [live Hub tree](https://huggingface.co/datasets/jpata/particleflow/tree/main/tensorflow_datasets) as the source of truth for publication availability. The `tfds_mapping`, `tfds_hit_mapping`, and model dataset entries in [`particleflow_spec.yaml`](https://github.com/jpata/particleflow/blob/main/particleflow_spec.yaml) define what the current code expects. A usable dataset has a detector, name, configuration, and version that match in both contexts.

For detector-software and sample details, see [CMS](cms.md), [CLD](cld.md), or [CLICdet](clic.md). For scientific provenance and citations, use the [publication map](../science/publications.md) and the dataset record associated with the version you consume.
