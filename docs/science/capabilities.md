# Current capabilities

This page describes the development branch as of **23 October 2026**, through
PR #541. It summarizes implemented paths and their validation boundaries.
Checkpoint performance belongs to the corresponding validation results.

**Supported** denotes a maintained configuration, documented user path, and
software checks. **Research workflow** denotes experimental paths whose scientific
and platform validation depends on the study. **Partial** identifies explicit
limitations in inputs or validation. These labels do not establish physics
performance or coverage of every configuration combination.

## Detector and input support

| Detector | Collision setting | Inputs | Data version in model recipe | Status |
|---|---|---|---|---|
| CMS Run 3 | Proton-proton, with and without pileup | Tracks and calorimeter clusters | 3.2.0 | **Supported** |
| CLD | Electron-positron | Tracks and calorimeter clusters | 3.3.0 | **Supported** |
| CLICdet | Electron-positron | Tracks and calorimeter clusters | 3.3.0 | **Supported** |
| CLD | Electron-positron | Tracker and calorimeter hits | 3.2.2 | **Research workflow** |
| CLICdet | Electron-positron | Tracker and calorimeter hits | 3.2.2 | **Research workflow** |
| ColliderML Release 1 | Proton-proton ttbar, no pileup or pileup ~200 | Tracks and spatial calorimeter clusters | 1.2.0 | **Research workflow** |
| ColliderML Release 1 | Proton-proton ttbar, no pileup | Tracker and calorimeter hits | 1.0.0 | **Research workflow** |
| MAIA | Muon-collider detector study | Tracks and calorimeter clusters | 1.1.0 | **Partial**; conversion, TFDS, and local CPU training |
| IDEA | Electron-positron detector concept | Truth-seeded proxy tracks and calorimeter clusters | 0.2.0 | **Partial**; pipeline validation with an oracle reference |

Versions come from `particleflow_spec.yaml`; they are not a statement of public
download availability. As verified on 28 October 2026, the Hub provides
configuration-1 CLD and CLIC track/cluster 3.3.0 and hit 3.2.2 data for the
standard `ttbar`, `WW`, and `qq` recipes, plus ColliderML track/cluster 1.2.0
and IDEA 0.2.0. IDEA builders default to 0.2.1 while its model recipe selects
0.2.0. See the [catalog](../datasets/catalog.md) for the verified Hub snapshot,
publication boundaries, and this version mismatch.

ColliderML has radius-graph and CLUE calorimeter clustering, surrogate-ancestor
truth targets, conversion to MLPF Parquet, and clustered/hit TFDS builders.
A pileup-200 hit builder also exists, but has no dedicated named training recipe
in the current specification. It has no PF baseline in the local comparison.
MAIA validation retains known failures and incomplete detector metadata; IDEA
proxy tracks and oracle candidates cannot establish reconstructed-physics quality.

## End-to-end operations

The documented CMS, CLD, and CLIC track/cluster paths cover detector production,
postprocessing, TFDS creation, training, checkpoint loading, inference, and
ONNX export/comparison. CMS physics validation requires CMSSW and site inputs;
CLD/CLIC use the [Key4HEP validation workflow](../validation/key4hep.md).

[CI](https://github.com/jpata/particleflow/blob/main/.github/workflows/test.yml)
runs the unit suite separately from detector smoke jobs:

| Detector smoke job | Scope |
|---|---|
| CMS | Small data preparation, CPU training, checkpoint loading, ONNX validation |
| CLD / CLIC | Small ROOT conversion, Parquet checks, TFDS, two-step CPU training |
| ColliderML | Offline synthetic source conversion, report-mode validation, TFDS, two-step CPU training |

MAIA has a local smoke script and unit coverage, but is outside this integration
matrix. Real ColliderML training and the five-detector feature comparison are
separate local checks in the [comparison runbook](../validation/detector-comparison.md).
Report-mode validators can report failed gates without failing the script.
GPU-dependent tests can skip when no GPU is available; CPU CI does not establish
accelerator, distributed, or full physics validation.

## Model and training support

| Model family | Intended use | Status |
|---|---|---|
| Multi-head attention | Default CMS, CLD, and CLIC track/cluster recipes | **Supported** |
| GNNLSH | Scalable graph processing with locality-sensitive hashing | **Research workflow** |
| HEPT and HEPTv2 | Hash-based efficient particle transformers | **Research workflow** |
| LitePT | Sparse point-transformer studies | **Research workflow**; NVIDIA-oriented dependencies |
| Shared backbone and task-query readout | Cross-task model studies | **Research workflow** |
| Particle-set decoder and hit grouping | CLD/CLIC/ColliderML hit-set studies | **Research workflow** |

For CLD raw hits, the current recommended research preset is
`pyg-cld-hits-heptv2-set-v1`, with global decoder attention, supervised hard
hit grouping, and tracker-neighborhood features. Dedicated tracker momentum
readout and particle-balanced grouping loss remain opt-in. Sparse pooling,
sectorized attention, fitted readouts, and auxiliary losses remain experimental.
The [CLD evidence summary](../cld_hit_results.md) records completed and partial
ablations, historical dataset versions, and changed metric definitions.

Training supports full checkpoint resume, [weights-only fine-tuning](../training/resume-and-fine-tune.md),
and the `lr-find` learning-rate range test. Per-step history includes fractional
training epoch and unweighted total losses. Optional reconstruction diagnostics
and frozen-backbone probes support hit-model studies; see the [repository map](../map.md).

Single-node distributed PyTorch training and scenario-based seeded job submission
are implemented. [Training scenarios](../../configs/training/README.md) separate
scientific recipes from hardware profiles, including local, Flatiron, and LUMI
execution. LUMI execution uses LAIF containers and has a site smoke check.
Ray Train and Ray Tune remain research workflows requiring cluster-specific setup.

## Validation and compatibility boundaries

Use the [validation overview](../validation/overview.md) to distinguish dataset
integrity, model behavior, physics behavior, and deployment checks. A short smoke
run establishes integration, not convergence. Historical study results retain
their original code, dataset, checkpoint, and metric definitions.

- Hit-level events require substantially more memory and tuning than track/cluster inputs.
- A reproducible checkpoint includes its saved model configuration and a compatible dataset schema.
- Current Key4HEP track/cluster data include muon-system cluster features; older 3.2.1 data must not be treated as current 3.3.0 input by relabeling directories.
- Site-specific production and accelerator paths require their own environment checks.
- Published physics claims remain tied to the artifacts and setup in the corresponding [publication](publications.md).
