# Model architectures

MLPF encodes detector elements, exchanges information between them, and predicts a particle collection. The current implementation follows this structure:

```text
input features and validity mask
  -> input encoding and optional hit features
  -> backbone layers
  -> elementwise heads or particle-set decoder
  -> particle presence, identity, and kinematics
```

## Encoding and information exchange

`input_encoding: joint` applies a common encoder to the input features. `split` selects an element-type-specific encoding, allowing tracks, clusters, or hit types to use different learned transformations. Optional hit features add geometry and local tracker/calorimeter summaries.

The backbone controls information exchange:

| `model.type` | Mechanism | Main computational considerations |
|---|---|---|
| `attention` | Multi-head self-attention across valid elements | Global pairwise attention; Flash execution packs valid elements, while dense math attention uses a padding mask |
| `gnnlsh` | Locality-sensitive hashing and message passing within bins, with configurable kernels and inter-bin attention | Bin size, hash counts, and inter-bin processing control work and connectivity |
| `hept` | Hash-based point attention with geometric information | Hash partitions and local attention control neighborhood size |
| `heptv2` | Geometry-aware hash buckets and local attention | Bucket size and hash settings control work; the native GPU path uses compiled FlexAttention |
| `litept` | The LitePT point-model adapter | Requires the external LitePT code and its dependencies; see [installation](../getting-started/installation.md) |

Attention is used by the standard CMS, CLD, and CLIC track/cluster recipes. Other backbones and hit models have research-workflow status. Use [capabilities](../science/capabilities.md), [ONNX validation](../validation/onnx.md), and [export evidence](../validation/export-evidence.md) for supported configurations and measured hardware behavior.

Several configuration choices act at different stages:

- `backbone.mode: shared` uses one backbone for classification and regression; `split` gives these tasks separate encoders and backbones.
- With a shared backbone, `task_queries` enables separate classification and regression attention readouts.
- Supported hit-attention configurations can specify tracker, calorimeter, and common layer counts to process subsystems before combining them.
- `learned_representation_mode` selects the last layer or concatenated layer representations where supported.

## Two output modes

| Property | `elementwise` | `set` |
|---|---|---|
| Prediction positions | One per input element | A configured number of particle slots |
| Supervision | Input-aligned target rows | Compact target particles matched to slots |
| Particle selection | Presence-head class | Configured presence-probability threshold |
| Readout | Classification and kinematic heads | Particle queries attending to backbone features |

Elementwise prediction learns which input positions represent particles. Set prediction starts from learned or input-conditioned queries. During training, Hungarian matching pairs slots with targets using configurable presence, PID, angular, and momentum costs. Unmatched slots receive no-object supervision. The matcher raises an error when an event has more targets than slots.

Optional hit grouping learns an assignment from each hit to a matched particle slot or background, using `particle_number` supervision. Tracker momentum readouts can use these groups. The CLD preset `pyg-cld-hits-heptv2-set-v1` combines HEPTv2, input-conditioned queries, global decoder attention, and supervised hit grouping. Its evidence and optional readout experiments are described in the [CLD hit results](../cld_hit_results.md).

## Learning objectives

The elementwise objective combines particle-presence classification, PID classification, and kinematic regression. PID and regression use rows with a particle target; presence also uses valid rows with no particle. Kinematic errors are weighted by the square root of target transverse momentum. Configuration selects fixed regression weights or calibrated task weights.

The set objective uses matched particles and no-object slots, with optional cardinality, intermediate-decoder, hit-grouping, and tracker-related terms. Jets and missing momentum provide downstream physics checks. The standard elementwise loss contains particle-level terms.

Implementation: [`ModelArchitectureConfig`](https://github.com/jpata/particleflow/blob/main/mlpf/conf.py), [`MLPF`](https://github.com/jpata/particleflow/blob/main/mlpf/model/mlpf.py), [`set_decoder.py`](https://github.com/jpata/particleflow/blob/main/mlpf/model/set_decoder.py), [`losses.py`](https://github.com/jpata/particleflow/blob/main/mlpf/model/losses.py), and [`set_losses.py`](https://github.com/jpata/particleflow/blob/main/mlpf/model/set_losses.py).

## Research context

The [2021 MLPF study](https://arxiv.org/abs/2101.08578) introduced full-event locality-sensitive graph-based reconstruction with a multi-task objective. The [2024 CLIC study](https://arxiv.org/abs/2309.06782) compared scalable graph and kernel-transformer models using tracks and calorimeter clusters. The [2025 transfer study](https://arxiv.org/abs/2503.00131) demonstrated CLIC-to-CLD fine-tuning, and the [2026 CMS study](https://arxiv.org/abs/2601.17554) evaluated MLPF within CMS software on simulation and collision data. The [publications page](../science/publications.md) links their archived code and results.

The local HEPT and HEPTv2 modules are based on the [HEPT tracking study](https://arxiv.org/abs/2510.07594) and [HEPTv2 paper](https://arxiv.org/abs/2606.20437). These papers study charged-particle tracking. Here, their point-encoder implementations are adapted to the MLPF feature layout and particle-reconstruction heads. Current particle-set and hit-grouping configurations are documented through repository tests and the CLD evidence page. Published performance numbers apply to the corresponding paper's detector, task, and software setup.
