# IDEA tracks and clusters

IDEA is a pipeline-validation workflow using truth-seeded proxy tracks and calorimeter clusters. Its oracle reference does not establish reconstructed-physics performance.

## Physics samples

| Physics sample | TFDS dataset | Recipe and published version |
|---|---|---|
| Top-quark pairs | `idea_edm_ttbar_pf` | 0.2.0 |
| Fully hadronic WW | `idea_edm_ww_fullhad_pf` | 0.2.0 |
| Inclusive quark pairs | `idea_edm_qq_pf` | 0.2.0 |

The `pyg-idea-pipeline-v1` recipe uses all three samples with configuration **1**. The [catalog](catalog.md#published-availability) records their public availability.

## Builder and recipe compatibility

The local builders default to **0.2.1**, which includes calorimeter-cell shower widths, while the model recipe selects **0.2.0**. Align the recipe with the actual rebuilt data before training. Do not relabel a dataset directory to bridge this difference.

For production, review `productions.idea` in `particleflow_spec.yaml` and the builders under `mlpf/heptfds/idea_pf_edm4hep/`, then follow [dataset generation](generate.md). The [detector-comparison runbook](../validation/detector-comparison.md) preserves the separate historical sample inventory and its limitations.
