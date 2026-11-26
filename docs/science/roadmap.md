# Limitations and development priorities

Use [Current capabilities](capabilities.md) for implemented workflows. This page collects remaining validation and research work for the v3.3.0 codebase; it is not a release history or a delivery schedule. Track active proposals in the [issue tracker](https://github.com/jpata/particleflow/issues).

## Data and detector coverage

| Area | Current boundary | Follow-up work |
|---|---|---|
| Published data | The Hub provides configuration 1 for the current CLD, CLIC, and ColliderML selections; see the catalog for versions not yet published. | Publish the recipe versions and additional partitions for full-corpus training. |
| IDEA | GGTF/GenFit inputs and oracle candidates; builders and recipes use 0.5.0. | Validate tracking quality, truth matching and reconstruction-level physics performance. |
| MAIA | Conversion and local training exist, with failed validation gates and incomplete detector metadata. | Resolve data-quality and metadata limitations before expanding support. |
| ColliderML | Clustered and hit workflows exist for ttbar; the other Release-1 samples and pileup-200 hits have builders but no training recipe. | Extend maintained recipes and detector-specific physics validation. |

Dataset versions and publication availability are maintained in the [catalog](../datasets/catalog.md).

## Models and physics validation

Raw-hit particle-set models, tracker readout, grouping losses, and sparse/sectorized decoding remain research workflows. Priorities are closing the track/cluster performance gap, measuring scaling on full events, and comparing options with matched datasets, training exposure, and metric definitions. The [CLD study](../studies/cld_hit_results.md) records the evidence behind the current research preset.

Joint training across detectors and fine-tuning to new geometries (IDEA, CLD granularity scans) have scenarios but no published results yet.

Cross-detector target definitions, descendant-track ownership, pileup labeling, single-particle monitoring, and outlier studies need detector-specific validation. Additional model families and generative reconstruction remain research directions; an experimental decoder does not establish a complete generative workflow.

## Deployment and accelerator qualification

Elementwise attention and HEPTv2 exports have numerical and padding regression coverage. The standalone Key4HEP evaluator does not yet support input-normalized checkpoints, which now include the release checkpoints. Remaining work includes a particle-set export contract, broader checkpoint and hardware qualification, and validated reduced-precision operating points. Optimized FP16 attention does not meet the retained trained-checkpoint FP32 tolerance. AMD results depend on the exact provider/compiler stack, and dynamic graphs can need CPU shape operations. See [Export qualification](../validation/export-evidence.md).

Single-node DDP and site-specific scenario submission are implemented. Multi-site throughput, microbatch sizing, memory use, and data exposure need measurement for each detector/model combination. Ray workflows require cluster-specific setup. CPU CI does not establish accelerator or distributed performance.
