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

Hits and target particles are shown for a representative `ttbar` event in the transverse plane. Tracker hits are red, ECAL hits blue, HCAL hits green, and muon-system hits orange.

| Raw hits | Target particles |
|:---:|:---:|
| ![CLD tracker, calorimeter, and muon-system hits.](../../notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/inputs/event_displays/cld_event_5_hits_5x5cm.svg) | ![CLD target particles.](../../notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/inputs/event_displays/cld_event_5_targets_5x5cm.svg) |

## Production and related inputs

Follow [shared Key4HEP production](key4hep.md) through strict Parquet validation, then use `PROD=cld pixi run tfds_hit`. The [hit overview](hit-based.md) explains row targets, placeholder baseline candidates, decoding checks, and memory requirements. Use [CLD tracks and clusters](cld.md) for the reconstructed-object representation.
