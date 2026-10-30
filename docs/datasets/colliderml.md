# ColliderML tracks and clusters

ColliderML Release 1 supplies tracks and spatial calorimeter clusters for top-pair events. This is a research workflow with schema version **1.2.0**; it has no PF baseline in the local detector comparison.

## Physics samples

| Physics sample | Pileup | TFDS dataset | Model recipe |
|---|---|---|---|
| Top-quark pairs | None | `colliderml_ttbar_nopu_pf` | `pyg-colliderml-v1` |
| Top-quark pairs | Approximately 200 interactions | `colliderml_ttbar_pu200_pf` | `pyg-colliderml-pu200-v1` |

`pyg-colliderml-pu0pu200-groups-v1` combines these samples in separate groups with independent batch sizes and optional step weights. See the runnable [dataset-mixing example](../training/data-mixing.md).

Both samples have published configuration-1 data in the [catalog](catalog.md#published-availability); recipes configure partitions 1--10. Use `--data_config 1` for the published subset.

## Event view

These views show the same simulated no-pileup `ttbar` event (index 0) in the transverse plane. Inputs are ACTS tracks and spatial calorimeter clusters. The particle panel shows visible primary-leaf truth; allocated training targets are defined by postprocessing.

| Tracks and clusters | Primary-leaf truth proxy |
|:---:|:---:|
| ![ColliderML: tracks and clusters for event 0.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/colliderml/event_0_tracks_clusters.png) | ![ColliderML: primary-leaf truth proxy for the same event.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/colliderml/event_0_particles.png) |

[All-component overlay](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/colliderml/event_0_overlay.png),
[vector input view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/colliderml/event_0_tracks_clusters.svg) and
[vector particle view](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/colliderml/event_0_particles.svg) are available for reuse.
See [event-view conventions](event-displays.md) for particle sources, sampling and scale.

## Production and validation

`productions.colliderml` in `particleflow_spec.yaml` defines the source revision, download and workspace roots, postprocessing, and TFDS mappings. The converter supports radius-graph and CLUE clustering and surrogate-ancestor truth targets. Follow [dataset generation](generate.md) with the reviewed source and workspace paths, and use the shared Parquet validator with `--detector colliderml`.

The [detector-comparison runbook](../validation/detector-comparison.md) records the real-data training check and feature-validation limits. [ColliderML hit datasets](colliderml-hits.md) use separate names and recipes.
