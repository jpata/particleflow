# CLICdet hits

CLICdet hit datasets contain tracker and calorimeter measurements from Key4HEP/EDM4hep production at 380 GeV. Hits are part of the [general datasets](full.md) `clic_edm_<process>_full` (version **1.0.0**), together with tracks, clusters and per-element truth labels. The hit recipe is `pyg-clic-hits-heptv2-set-v1`. Hit reconstruction is a research workflow.

## Physics samples

| Physics sample | TFDS dataset | Training selection |
|---|---|---|
| Top-quark pairs | `clic_edm_ttbar_full` | Included in default recipe |
| Fully hadronic WW | `clic_edm_ww_fullhad_full` | Included in default recipe |
| Inclusive quark pairs | `clic_edm_qq_full` | Included in default recipe |
| Z→qq (91 GeV) | `clic_edm_z_qq_full` | Produced; outside default recipe |
| Z→tautau (91 GeV) | `clic_edm_z_tautau_full` | Produced; outside default recipe |

Recipes configure partitions 1--10. The general datasets are not yet on the Hub (see the [catalog](catalog.md#published-availability)); produce them with the [Key4hep v1.4.0 production](key4hep-50k.md).

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

Follow [shared Key4HEP production](key4hep.md) through strict Parquet validation, then use `PROD=clic pixi run tfds_full` to build the [general datasets](full.md) (`clic_edm_<process>_full`), which contain the raw hits together with tracks, clusters and truth labels. They replace the `clic_edm_<process>_hits` builders. The [hit overview](hit-based.md) explains row targets, placeholder baseline candidates, decoding checks, and memory requirements. Use [CLICdet tracks and clusters](clic.md) for the reconstructed-object representation.
