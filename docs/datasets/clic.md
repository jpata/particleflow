# CLICdet tracks and clusters

CLICdet track/cluster datasets use Key4HEP/EDM4hep production at 380 GeV. The schema version is **3.3.0**, and the standard model recipe is `pyg-clic-v1`.

## Physics samples

| Physics sample | TFDS dataset | Training selection |
|---|---|---|
| Top-quark pairs | `clic_edm_ttbar_pf` | Included in default recipe |
| Fully hadronic WW | `clic_edm_ww_fullhad_pf` | Included in default recipe |
| Inclusive quark pairs | `clic_edm_qq_pf` | Included in default recipe |

The recipes configure partitions 1--10; the [verified public selection](catalog.md#published-availability) provides configuration 1. Use `--data_config 1` for the download-based training path. The default recipe combines `ttbar`, `WW`, and `qq`; their ordering and relative exposure are described in [Dataset mixing](../training/data-mixing.md).

## Event view

These views show the same simulated `ttbar` event (index 0) at 380 GeV in the transverse plane. The particle panel shows the Pandora reference reconstruction.

| Tracks and clusters | Pandora reconstructed particles |
|:---:|:---:|
| ![CLIC: tracks and clusters for event 0.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/clic/event_0_tracks_clusters.png) | ![CLIC: pandora reconstructed particles for the same event.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/clic/event_0_particles.png) |

[All-component overlay](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/clic/event_0_overlay.png),
[vector input view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/clic/event_0_tracks_clusters.svg) and
[vector particle view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/clic/event_0_particles.svg) are available for reuse.
See [event-view conventions](event-displays.md) for particle sources, sampling and scale.

## Production and related inputs

Use [shared Key4HEP production](key4hep.md) with `PROD=clic` and build this representation with `pixi run tfds`. For tracker and calorimeter measurements, use the separate [CLICdet hit datasets](clic-hits.md).
