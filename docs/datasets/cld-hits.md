# CLD hits

CLD hit datasets contain tracker and calorimeter measurements from Key4HEP/EDM4hep production at 365 GeV. The schema version is **3.2.2**, and the standard hit recipe is `pyg-cld-hits-v1`. Hit reconstruction is a research workflow.

## Physics samples

| Physics sample | TFDS dataset | Training selection |
|---|---|---|
| Top-quark pairs | `cld_edm_ttbar_hits` | Included in default recipe |
| Fully hadronic WW | `cld_edm_ww_fullhad_hits` | Included in default recipe |
| Inclusive quark pairs | `cld_edm_qq_hits` | Included in default recipe |
| ZZ | `cld_edm_zz_hits` | Additional builder; outside default recipe |

Recipes configure partitions 1--10. The [verified Hub selection](catalog.md#published-availability) provides configuration 1 for `ttbar`, `WW`, and `qq`; check other selections separately.

For CLD particle-set studies, the recommended research preset is `pyg-cld-hits-heptv2-set-v1`; see the [evidence and recommendation](../cld_hit_results.md).

## Event view

These views show the same simulated `ttbar` event (index 0) at 365 GeV in the transverse plane. The particle panel shows the Pandora reference reconstruction from the source event; the hit dataset stores its own training targets.

| Raw detector hits | Pandora reconstructed particles |
|:---:|:---:|
| ![CLD: raw detector hits for event 0.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/cld/event_0_hits.png) | ![CLD: pandora reconstructed particles for the same event.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/cld/event_0_particles.png) |

[All-component overlay](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/cld/event_0_overlay.png),
[vector input view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/cld/event_0_hits.svg) and
[vector particle view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/cld/event_0_particles.svg) are available for reuse.
See [event-view conventions](event-displays.md) for particle sources, sampling and scale.

## Production and related inputs

Follow [shared Key4HEP production](key4hep.md) through strict Parquet validation, then use `PROD=cld pixi run tfds_hit`. The [hit overview](hit-based.md) explains row targets, placeholder baseline candidates, decoding checks, and memory requirements. Use [CLD tracks and clusters](cld.md) for the reconstructed-object representation.
