# Roadmap

This roadmap is a snapshot of public development work on 4 September 2026. Open issues and pull requests describe directions, not delivery commitments. Check the linked GitHub item for its current status.

## Active integration work

- [IDEA end-to-end integration](https://github.com/jpata/particleflow/pull/495) extends detector coverage.
- [Learned-query set prediction](https://github.com/jpata/particleflow/pull/497) explores a different way to construct output particles.
- [Safer, resumable Hugging Face uploads](https://github.com/jpata/particleflow/pull/499) improves dataset publication.

## Near-term correctness and validation

- [Batch and microbatch semantics](https://github.com/jpata/particleflow/issues/498) aims to make training configuration easier to compare across scenarios.
- [Target ownership for descendant tracks](https://github.com/jpata/particleflow/issues/496) addresses training-target semantics.
- [Simulation-aware evaluation](https://github.com/jpata/particleflow/issues/370) tracks improvements to jet and target comparisons.
- [CMS pileup labels](https://github.com/jpata/particleflow/issues/368) and [CMS ECAL cluster shapes](https://github.com/jpata/particleflow/issues/307) track missing detector information.
- [Single-particle monitoring](https://github.com/jpata/particleflow/issues/357) and [outlier studies](https://github.com/jpata/particleflow/issues/327) improve diagnosis of reconstruction failures.

## Longer-term research directions

- [Common cross-detector datasets](https://github.com/jpata/particleflow/issues/443)
- [Linear attention](https://github.com/jpata/particleflow/issues/435)
- [State-space models](https://github.com/jpata/particleflow/issues/281)
- [Set-conditional generation](https://github.com/jpata/particleflow/issues/247)
- [Interactive public model inference](https://github.com/jpata/particleflow/issues/241)

## Documentation delivery

The detailed documentation work is tracked in [issue #500](https://github.com/jpata/particleflow/issues/500). The intended order is:

1. Establish the site, onboarding, physics overview, capabilities, publications, and roadmap.
2. Document dataset download, generation, detector-specific processing, and data validation.
3. Document training, fine-tuning, distributed execution, physics validation, and ONNX validation.
4. Add generated reference material, troubleshooting, CI checks, and maintenance guidance.
