# Dataset catalog

Use this page to select a dataset. For installation commands, see [Download a dataset](download.md); for producing new data, see [Generate a dataset](generate.md).

The tables describe the recipes on the current development branch. Dataset versions are schema and production versions; they are independent of the Python package version and model-checkpoint version. Always record all three in derived work.

## Current recipes

The canonical inventory is `productions.*.tfds_mapping`, `productions.*.tfds_hit_mapping`, and `models.*_datasets` in [`particleflow_spec.yaml`](https://github.com/jpata/particleflow/blob/main/particleflow_spec.yaml). The summary below should change with that file.

| Detector | Input representation | Dataset names | Recipe version | Compatible model recipe |
|---|---|---|---|---|
| CMS Run 3, pileup | Tracks and calorimeter clusters | `cms_pf_ttbar`, `cms_pf_qcd`, `cms_pf_ztt` | 3.2.0 | `pyg-cms-v1` |
| CMS Run 3, no pileup | Tracks and calorimeter clusters | `cms_pf_ttbar_nopu`, `cms_pf_qcd_nopu`, `cms_pf_ztt_nopu` | 3.2.0 | `pyg-cms-v1` |
| CLD, 365 GeV | Tracks and calorimeter clusters | `cld_edm_ttbar_pf`, `cld_edm_ww_fullhad_pf`, `cld_edm_qq_pf`, `cld_edm_zz_pf` | 3.2.1 | `pyg-cld-v1` (the default recipe excludes `zz`) |
| CLD, 365 GeV | Tracker and calorimeter hits | `cld_edm_ttbar_hits`, `cld_edm_ww_fullhad_hits`, `cld_edm_qq_hits`, `cld_edm_zz_hits` | 3.2.1 | `pyg-cld-hits-v1` (the default recipe excludes `zz`) |
| CLICdet, 380 GeV | Tracks and calorimeter clusters | `clic_edm_ttbar_pf`, `clic_edm_ww_fullhad_pf`, `clic_edm_qq_pf` | 3.2.1 | `pyg-clic-v1` |
| CLICdet, 380 GeV | Tracker and calorimeter hits | `clic_edm_ttbar_hits`, `clic_edm_ww_fullhad_hits`, `clic_edm_qq_hits` | 3.2.1 | `pyg-clic-hits-v1` |

IDEA `0.1.0` datasets also exist for pipeline validation. They use truth-seeded proxy tracks and an oracle reference, so they are not included in the supported training catalog. Their limitations are recorded in [Current capabilities](../science/capabilities.md).

## Configuration partitions and event splits

Each recipe defines configuration partitions `1` through `10`. A configuration partition contains its own `train` and `test` event splits; the configuration number is not an event-split name. The default model recipes combine all ten partitions. Downloading only configuration `1` is useful for a small experiment, but it is not the complete configured training corpus.

The on-disk layout is:

```text
<data root>/<dataset name>/<configuration>/<version>/
```

For example:

```text
data/tfds/tensorflow_datasets/cld/cld_edm_ttbar_pf/1/3.2.1/
```

Pass the detector-level directory containing the dataset-name directories as `--data-dir`; for this example it is `data/tfds/tensorflow_datasets/cld`.

## Published availability

As checked on 14 September 2026, the [public Hugging Face dataset repository](https://huggingface.co/datasets/jpata/particleflow/tree/main/tensorflow_datasets) contains configuration `1` of the CLD and CLIC datasets above at their current recipe versions. It also contains older versions and the experimental IDEA datasets. The CMS 3.2.0 TFDS directories are configured in the repository but are not present under the public Hub's `tensorflow_datasets/` tree.

Treat the live Hub tree as the source of truth for publication availability. Treat `particleflow_spec.yaml` as the source of truth for what the current code expects. A dataset is usable only when the requested detector, name, configuration, and version exist in both contexts.

## Which representation should I use?

Start with the track-and-cluster (`*_pf`) datasets unless the study specifically concerns lower-level detector inputs. They are the supported default path and have substantially smaller events. The hit datasets require the dedicated model recipes and more memory; see [Hit-based datasets](hit-based.md).

For detector-software and sample details, see [CMS data](cms.md) or [CLD and CLIC data](key4hep.md). For scientific provenance and citations, use the [publication map](../science/publications.md) and the dataset record associated with the version you consume.
