# Limitations and development priorities

Use [Current capabilities](capabilities.md) for implemented workflows. This page collects remaining validation and research work for the v3.2.0 codebase; it is not a release history or a delivery schedule. Track active proposals in the [issue tracker](https://github.com/jpata/particleflow/issues).

## Data and detector coverage

| Area | Current boundary | Follow-up work |
|---|---|---|
| Published data | Current CLD, CLIC, and ColliderML selections on the Hub provide configuration 1. | Publish and validate additional partitions for full-corpus training. |
| IDEA | Truth-seeded proxy tracks and oracle candidates; published/model version 0.2.0 differs from builder default 0.2.1. | Align rebuilt data and recipes, then establish reconstruction-level validation. |
| MAIA | Conversion and local training exist, with failed validation gates and incomplete detector metadata. | Resolve data-quality and metadata limitations before expanding support. |
| ColliderML | Clustered and hit workflows exist; pileup-200 hits have no named training recipe. | Extend maintained recipes and detector-specific physics validation. |

Dataset versions and publication availability are maintained in the [catalog](../datasets/catalog.md).

## Models and physics validation

Raw-hit particle-set models, tracker readout, grouping losses, and sparse/sectorized decoding remain research workflows. Priorities are closing the track/cluster performance gap, measuring scaling on full events, and comparing options with matched datasets, training exposure, and metric definitions. The [CLD study](../cld_hit_results.md) records the evidence behind the current research preset.

Cross-detector target definitions, descendant-track ownership, pileup labeling, single-particle monitoring, and outlier studies need detector-specific validation. Additional model families and generative reconstruction remain research directions; an experimental decoder does not establish a complete generative workflow.

## Deployment and accelerator qualification

Elementwise attention and HEPTv2 exports have numerical and padding regression coverage. Remaining work includes a particle-set export contract, broader checkpoint and hardware qualification, and validated reduced-precision operating points. Optimized FP16 attention does not meet the retained trained-checkpoint FP32 tolerance. AMD results depend on the exact provider/compiler stack, and dynamic graphs can need CPU shape operations. See [Export qualification](../validation/export-evidence.md).

Single-node DDP and site-specific scenario submission are implemented. Multi-site throughput, microbatch sizing, memory use, and data exposure need measurement for each detector/model combination. Ray workflows require cluster-specific setup. CPU CI does not establish accelerator or distributed performance.

## Documentation and maintenance

Keep runnable guides aligned with configuration defaults, published datasets, and tested environments. Add troubleshooting and generated reference material where they simplify maintained workflows. Report evidence with its model, data, software stack, and validation scope, and keep experimental options separate from recommended defaults. Follow the [documentation contribution guide](../development/documentation.md).
