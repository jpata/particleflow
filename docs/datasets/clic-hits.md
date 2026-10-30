# CLICdet hits

CLICdet hit datasets contain tracker and calorimeter measurements from Key4HEP/EDM4hep production at 380 GeV. The schema version is **3.2.2**, and the standard hit recipe is `pyg-clic-hits-v1`. Hit reconstruction is a research workflow.

## Physics samples

| Physics sample | TFDS dataset | Training selection |
|---|---|---|
| Top-quark pairs | `clic_edm_ttbar_hits` | Included in default recipe |
| Fully hadronic WW | `clic_edm_ww_fullhad_hits` | Included in default recipe |
| Inclusive quark pairs | `clic_edm_qq_hits` | Included in default recipe |

Recipes configure partitions 1--10. The [verified Hub selection](catalog.md#published-availability) provides configuration 1 for `ttbar`, `WW`, and `qq`; check other selections separately.

## Event view

These views show the same simulated `ttbar` event (index 0) at 380 GeV in the transverse plane. The particle panel shows the Pandora reference reconstruction from the source event; the hit dataset stores its own training targets.

| Raw detector hits | Pandora reconstructed particles |
|:---:|:---:|
| ![CLIC: raw detector hits for event 0.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/clic/event_0_hits.png) | ![CLIC: pandora reconstructed particles for the same event.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/clic/event_0_particles.png) |

[All-component overlay](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/clic/event_0_overlay.png),
[vector input view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/clic/event_0_hits.svg) and
[vector particle view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/clic/event_0_particles.svg) are available for reuse.
See [event-view conventions](event-displays.md) for particle sources, sampling and scale.

## Production and related inputs

Follow [shared Key4HEP production](key4hep.md) through strict Parquet validation, then use `PROD=clic pixi run tfds_hit`. The [hit overview](hit-based.md) explains row targets, placeholder baseline candidates, decoding checks, and memory requirements. Use [CLICdet tracks and clusters](clic.md) for the reconstructed-object representation.
