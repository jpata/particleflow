# Current capabilities

This page summarizes the codebase prepared for **v3.3.0**. Use it to choose a maintained workflow and understand its validation limits. Checkpoint performance belongs to the corresponding validation results.

**Supported** denotes a maintained configuration, documented user path, and
software checks. **Research workflow** denotes experimental paths whose scientific
and platform validation depends on the study. **Partial** identifies explicit
limitations in inputs or validation. These labels do not establish physics
performance or coverage of every configuration combination.

## Detector and input support

| Detector | Collision setting | Inputs | Data version in recipe | Status |
|---|---|---|---|---|
| CMS Run 3 | Proton-proton, with and without pileup | Tracks and calorimeter clusters | 3.2.0 (scenarios: 2.8.0) | **Supported** |
| CLD | Electron-positron, 365 GeV | Tracks and calorimeter clusters | 3.4.0 | **Supported** |
| CLICdet | Electron-positron, 380 GeV | Tracks and calorimeter clusters | 3.4.0 | **Supported** |
| CLD | Electron-positron | Tracker and calorimeter hits | 3.3.0 | **Research workflow** |
| CLICdet | Electron-positron | Tracker and calorimeter hits | 3.3.0 | **Research workflow** |
| ColliderML Release 1 | Proton-proton ttbar, no pileup or ~200 | Tracks and CLUE calorimeter clusters | 1.5.0 | **Research workflow** |
| ColliderML Release 1 | Proton-proton ttbar, no pileup | Tracker and calorimeter hits | 1.3.0 | **Research workflow** |
| MAIA | Muon-collider detector study | Tracks and calorimeter clusters | 1.2.0 | **Partial**: conversion, TFDS, local CPU training; failed validation gates |
| IDEA | Electron-positron detector concept | Fitted GGTF/GenFit tracks and dual-readout calorimeter clusters | 0.5.0 | **Partial**: pipeline validation with an oracle reference |
| ILD@FCC-ee | Electron-positron, 91 and 365 GeV | Fitted tracks and Pandora calorimeter clusters | 0.2.0 | **Experimental**: reconstruction not yet validated for physics |

Versions come from `particleflow_spec.yaml`. The [catalog](../datasets/catalog.md#published-availability) lists which of them are published on the Hub. The CLD, CLIC, IDEA and ILD versions come from the [Key4hep v1.4.0 production](../datasets/key4hep-50k.md); all four are published on the Hub. The older IDEA 0.3.0 data, also on the Hub, use truth-seeded proxy tracks and cannot replace 0.5.0.

CLD calorimeter granularity scans (six ECAL and six HCAL cell sizes) have their own productions (`cld_ecal_pitch_*`, `cld_hcal_pitch_*`) built from shared geometry; see the [granularity scan runbook](https://github.com/jpata/particleflow/blob/main/scripts/cld_ecal_readout/GRANULARITY_SCAN.md). ColliderML has builders for the other Release-1 samples, and a pileup-200 hit builder, without training recipes.

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
| ColliderML | Offline synthetic source conversion, Parquet checks (failures do not stop the job), TFDS, two-step CPU training |

MAIA has a local smoke script and unit coverage, but is outside this integration
matrix. Real ColliderML training and the five-detector feature comparison are
separate local checks in the [comparison runbook](../validation/detector-comparison.md).
The ColliderML and MAIA smoke scripts continue past failed Parquet checks, so a passing script does not mean the data passed validation.
GPU-dependent tests can skip when no GPU is available; CPU CI does not establish
accelerator, distributed, or full physics validation.

## Model and training support

| Model family | Intended use | Status |
|---|---|---|
| Multi-head attention | Default CMS, CLD, and CLIC track/cluster recipes | **Supported** |
| HEPTv2 | Hash-bucketed attention; release checkpoints for all detectors | **Research workflow** |
| GNNLSH, HEPT | Earlier scalable graph and hashing models | **Research workflow** |
| LitePT | Sparse point-transformer studies | **Research workflow**; NVIDIA-oriented dependencies |
| Particle-set decoder and hit grouping | CLD/CLIC/ColliderML hit-set studies | **Research workflow** |

For CLD raw hits, the recommended research preset is `pyg-cld-hits-heptv2-set-v1`; the [CLD hit-model evidence](../studies/cld_hit_results.md) records the ablations behind it.

Training features:

- [Input normalization](../training/train.md#input-normalization) of track/cluster features with per-dataset-version statistics, embedded in checkpoints and ONNX graphs.
- [Joint training on several detectors](../training/train.md#train-on-several-detectors) (CLD, CLIC, ColliderML) with an optional shared feature vocabulary (`model.feature_bag`). Research workflow.
- HEPTv2 can hash tracks at their calorimeter entry and Key4HEP clusters at their centroid (`track_hash_at_calo`, `cluster_hash_at_centroid`), so that a track and its cluster land in the same bucket. On by default in the HEPTv2 settings of the CMS, CLD, CLIC, ColliderML, and IDEA recipes.
- Full checkpoint resume, [weights-only fine-tuning](../training/resume-and-fine-tune.md), and the `lr-find` learning-rate range test.
- [Dataset mixing](../training/data-mixing.md) with per-group step weights, interleaved shards, and per-group losses.
- Single-node DDP. Ray Train and Ray Tune remain research workflows requiring cluster-specific setup.
- [Training scenarios](../training/scenarios.md) with hardware profiles for local, Flatiron, LUMI, Tallinn, and EmpireAI execution, including the fixed `release_checkpoints` recipe used for each release.
- Opt-in per-worker benchmark telemetry and [portable GPU benchmarks](../training/benchmark-portability.md).

## Attention and deployment

Native Flash attention packs valid elements automatically; dense math attention masks padded keys and supports CPU execution. HEPTv2 defaults to deterministic integer hashing with stable sorting. Both changes can alter predictions from older checkpoints and require held-out validation.

The elementwise ONNX exporter supports decomposed opset-20 attention and standard Attention-24 with valid key lengths. Exported outputs are zero on padding, and batch/sequence dimensions are dynamic. Set-decoder export requires a separate output contract and is not supported by this helper.

CPU export CI checks numerical and padding contracts for attention and HEPTv2. GPU qualification separately checks runtime placement, precision, and kernel execution. Retained trained-checkpoint evidence qualifies the portable FP32 matrix on CPU, NVIDIA L40S, and AMD W7900 in the recorded environments; optimized FP16 attention executes but fails the same numerical tolerance. These checks establish export behavior for those artifacts, not physics quality for all checkpoints. See [ONNX validation](../validation/onnx.md) and [export evidence](../validation/export-evidence.md).

## Validation and compatibility boundaries

Use the [validation overview](../validation/overview.md) to distinguish dataset
integrity, model behavior, physics behavior, and deployment checks. A short smoke
run establishes integration, not convergence. Historical study results retain
their original code, dataset, checkpoint, and metric definitions.

- Hit-level events require substantially more memory and tuning than track/cluster inputs.
- A reproducible checkpoint includes its saved model configuration and a compatible dataset schema.
- Dataset versions change features or targets; never relabel an older dataset directory to satisfy a recipe. For example, IDEA 0.5.0 uses fitted tracks where 0.3.0 used truth-seeded ones.
- The standalone Key4HEP evaluator does not yet pass dataset source IDs, so it cannot run checkpoints trained with input normalization (including the v3.3.0 release checkpoints). Use the pipeline `test` command for those.
- Site-specific production and accelerator paths require their own environment checks.
- Published physics claims remain tied to the artifacts and setup in the corresponding [publication](publications.md).
