# ColliderML hits

ColliderML Release 1 hit datasets use raw tracker and calorimeter hits with one target row per input hit. Both builders use schema version **1.0.0**. Pileup membership is carried in the target particle's `ispu` feature.

## Physics samples

| Physics sample | Pileup | TFDS dataset | Model recipe |
|---|---|---|---|
| Top-quark pairs | None | `colliderml_ttbar_hits` | `pyg-colliderml-hits-v1`, `pyg-colliderml-hits-set-v1` |
| Top-quark pairs | Approximately 200 interactions | `colliderml_ttbar_pu200_hits` | Builder exists; no named training recipe |

The TFDS mappings configure partitions 1--10. These samples are not included in the [verified public selection](catalog.md#published-availability); check publication availability before downloading. A custom pileup-200 training recipe requires its own validation.

## Production and related inputs

`productions.colliderml.tfds_hit_mapping` in `particleflow_spec.yaml` selects the hit builders under `mlpf/heptfds/colliderml_hits/`. Use the ColliderML converter and source/workspace configuration with [dataset generation](generate.md); the CLD/CLIC Key4HEP commands do not define ColliderML conversion.

Inspect input sizes and measure memory with a short dedicated hit-model run before scaling. For the clustered representation and its weighted two-group recipe, see [ColliderML tracks and clusters](colliderml.md).
