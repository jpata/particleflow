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

These views show the same simulated `ttbar` event (index 0) at 365 GeV in the transverse plane. The particle panel shows the Pandora reference reconstruction.

| Tracks and clusters | Pandora reconstructed particles |
|:---:|:---:|
| ![CLD: tracks and clusters for event 0.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/cld/event_0_tracks_clusters.png) | ![CLD: pandora reconstructed particles for the same event.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/cld/event_0_particles.png) |

[All-component overlay](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/cld/event_0_overlay.png),
[vector input view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/cld/event_0_tracks_clusters.svg) and
[vector particle view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/cld/event_0_particles.svg) are available for reuse.
See [event-view conventions](event-displays.md) for particle sources, sampling and scale.

## Production and related inputs

Use [shared Key4HEP production](key4hep.md) with `PROD=cld` and build this representation with `pixi run tfds`. For tracker and calorimeter measurements, use the separate [CLD hit datasets](cld-hits.md).
