# CLD hits

CLD hit datasets contain tracker and calorimeter measurements from Key4HEP/EDM4hep production at 365 GeV. Hits are part of the [general datasets](full.md) `cld_edm_<process>_full` (version **1.0.0**), together with tracks, clusters and per-element truth labels. The hit recipes are `pyg-cld-hits-heptv2-set-v1` (recommended) and `pyg-cld-hits-heptv2-sector-set-v1`. Hit reconstruction is a research workflow.

## Physics samples

| Physics sample | TFDS dataset | Training selection |
|---|---|---|
| Top-quark pairs | `cld_edm_ttbar_full` | Included in default recipe |
| Fully hadronic WW | `cld_edm_ww_fullhad_full` | Included in default recipe |
| Inclusive quark pairs | `cld_edm_qq_full` | Included in default recipe |
| Z→qq (91 GeV) | `cld_edm_z_qq_full` | Produced; outside default recipe |
| Z→tautau (91 GeV) | `cld_edm_z_tautau_full` | Produced; outside default recipe |
| ZZ and the 240/365 GeV tau samples | `cld_edm_<process>_full` | Additional builders; not produced by default |

Recipes configure partitions 1--10. The general datasets are not yet on the Hub (see the [catalog](catalog.md#published-availability)); produce them with the [Key4hep v1.4.0 production](key4hep-50k.md).

See the [evidence and recommendation](../studies/cld_hit_results.md) for the HEPTv2 set model.

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

Follow [shared Key4HEP production](key4hep.md) through strict Parquet validation, then use `PROD=cld pixi run tfds_full` to build the [general datasets](full.md) (`cld_edm_<process>_full`), which contain the raw hits together with tracks, clusters and truth labels. They replace the `cld_edm_<process>_hits` builders. The [hit overview](hit-based.md) explains row targets, placeholder baseline candidates, decoding checks, and memory requirements. Use [CLD tracks and clusters](cld.md) for the reconstructed-object representation.
