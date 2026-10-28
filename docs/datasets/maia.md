# MAIA tracks and clusters

MAIA provides a track/cluster workflow for a muon-collider detector study. Conversion, TFDS building, and local CPU training are implemented, but data validation retains failed gates and incomplete detector metadata.

## Physics sample

| Physics sample | TFDS dataset | Schema version | Model recipe |
|---|---|---|---|
| Top-quark pairs | `maia_edm_ttbar_pf` | 1.1.0 | `pyg-maia-v1` |

The recipe configures partitions 1--10. Public availability must be checked independently; MAIA is not part of the [verified Hub selection](catalog.md#published-availability).

## Production and validation

Review `productions.maia` in `particleflow_spec.yaml` before following [dataset generation](generate.md). The builder is under `mlpf/heptfds/maia_pf_edm4hep/`. The [detector-comparison runbook](../validation/detector-comparison.md) records source assumptions, validation failures, and local smoke coverage. Passing a short training run does not resolve the detector-data limitations.
