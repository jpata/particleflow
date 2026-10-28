# CLD tracks and clusters

CLD track/cluster datasets use Key4HEP/EDM4hep production at 365 GeV. The schema version is **3.3.0**, and the standard model recipe is `pyg-cld-v1`.

## Physics samples

| Physics sample | TFDS dataset | Training selection |
|---|---|---|
| Top-quark pairs | `cld_edm_ttbar_pf` | Included in default recipe |
| Fully hadronic WW | `cld_edm_ww_fullhad_pf` | Included in default recipe |
| Inclusive quark pairs | `cld_edm_qq_pf` | Included in default recipe |
| ZZ | `cld_edm_zz_pf` | Additional builder; add to a custom recipe |

The recipes configure partitions 1--10; the [verified public selection](catalog.md#published-availability) provides configuration 1. Use `--data_config 1` for the download-based training path. The default recipe combines `ttbar`, `WW`, and `qq`; their ordering and relative exposure are described in [Dataset mixing](../training/data-mixing.md).

## Event view

Tracks and calorimeter clusters are shown beside the target particles for a
representative `ttbar` event in the transverse plane.

| Tracks and clusters | Target particles |
|:---:|:---:|
| ![CLD reconstructed tracks and calorimeter clusters.](../../notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/inputs/event_displays/cld_event_5_pf_5x5cm.svg) | ![CLD target particles.](../../notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/inputs/event_displays/cld_event_5_targets_5x5cm.svg) |

Target labels use $e$, $\mu$, $\tau$, $\nu$, $\gamma$, $\pi$, and $K^0$.

## Production and related inputs

Use [shared Key4HEP production](key4hep.md) with `PROD=cld` and build this representation with `pixi run tfds`. For tracker and calorimeter measurements, use the separate [CLD hit datasets](cld-hits.md).
