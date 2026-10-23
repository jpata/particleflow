# Roadmap

This snapshot was updated on **23 October 2026**, through merged PR #541.
The linked issues below were open when checked on that date. They describe
research directions and remaining work, not delivery commitments. Implemented
paths and their limitations are summarized in [Current capabilities](capabilities.md).

## Recently integrated

- [IDEA pipeline integration (#495)](https://github.com/jpata/particleflow/pull/495), with truth-seeded proxy tracks and an oracle reference for software validation.
- [Particle-set prediction (#497)](https://github.com/jpata/particleflow/pull/497) and [CLD raw-hit reconstruction, tracker readout, and diagnostics (#541)](https://github.com/jpata/particleflow/pull/541). Experimental options retain the evidence boundaries in the [CLD results summary](../cld_hit_results.md).
- [Hugging Face upload improvements (#499)](https://github.com/jpata/particleflow/pull/499) and [explicit ColliderML upload source directories (#519)](https://github.com/jpata/particleflow/pull/519).
- [ColliderML Release 1 integration (#509)](https://github.com/jpata/particleflow/pull/509), [CLUE clustering (#521)](https://github.com/jpata/particleflow/pull/521), and [surrogate-ancestor truth targets (#526)](https://github.com/jpata/particleflow/pull/526).
- [MAIA processing fixes (#512)](https://github.com/jpata/particleflow/pull/512), [IDEA cluster widths (#515)](https://github.com/jpata/particleflow/pull/515), and [muon-system cluster features (#517)](https://github.com/jpata/particleflow/pull/517).
- [Learning-rate finder (#528)](https://github.com/jpata/particleflow/pull/528), [resume fixes and weights-only loading (#529)](https://github.com/jpata/particleflow/pull/529), and [LUMI LAIF execution (#532)](https://github.com/jpata/particleflow/pull/532).
- [Matching-cost correction (#522)](https://github.com/jpata/particleflow/pull/522). Historical metrics may need common rescoring before comparison with current results.

Merged functionality is not automatically a recommended scientific default.
Current CLD hit-set recommendations distinguish completed ablations, partial
studies, and opt-in options.

## Open correctness and validation work

- [Scenario-global batches and task/dataset/GPU-specific microbatches (#498)](https://github.com/jpata/particleflow/issues/498).
- [Descendant-track ownership versus status-1 target representation (#496)](https://github.com/jpata/particleflow/issues/496). ColliderML surrogate targets are implemented, but this broader ownership issue remains open.
- [Simulation-aware evaluation (#370)](https://github.com/jpata/particleflow/issues/370).
- [CMS pileup labels (#368)](https://github.com/jpata/particleflow/issues/368) and [CMS ECAL cluster shapes (#307)](https://github.com/jpata/particleflow/issues/307).
- [Single-particle monitoring (#357)](https://github.com/jpata/particleflow/issues/357) and [CMS outlier studies (#327)](https://github.com/jpata/particleflow/issues/327). New hit-reconstruction diagnostics do not close these broader tasks.

## Open research directions

- [Common ColliderML/Collide-2V datasets (#443)](https://github.com/jpata/particleflow/issues/443); ColliderML integration is now part of the implemented foundation.
- [Linear attention (#435)](https://github.com/jpata/particleflow/issues/435).
- [State-space models (#281)](https://github.com/jpata/particleflow/issues/281).
- [Set-conditional generation (#247)](https://github.com/jpata/particleflow/issues/247); the merged particle-set decoder does not establish completion of this research direction.
- [Hugging Face model card and model execution (#241)](https://github.com/jpata/particleflow/issues/241).

## Documentation and maintenance

The documentation site, onboarding, dataset production/download guides, training,
resume, and validation guides are implemented. The broader
[documentation issue (#500)](https://github.com/jpata/particleflow/issues/500)
remains open; generated reference material and further troubleshooting and
maintenance guidance remain areas for follow-up.

The repository has strict documentation builds, unit and detector integration
CI, and a first-party complexity check. Periodic cleanup automation is a proposed
next step, not an installed workflow. Its initial priorities are keeping recipe
and publication versions distinct, recording actual validation coverage, and
reviewing which experimental interfaces remain maintained. Follow the
[documentation contribution guide](../development/documentation.md) when changing
commands, capability status, or scientific claims.
