# From simulation to validation

Each stage establishes the inputs for the next. Published datasets and checkpoints let users enter this workflow at training or evaluation.

| Stage | Transformation and responsible code | Check before proceeding |
|---|---|---|
| Simulation | Detector-specific generation under `mlpf/data/` produces ROOT detector signals and truth relationships | Expected collections, events, and production configuration |
| Postprocessing | CMS, Key4HEP, or ColliderML converters construct element features and assigned targets in pickle or Parquet | Schema, associations, closure, and geometry checks supported by the converter |
| Dataset construction | Builders under `mlpf/heptfds/` package a representation into versioned TFDS splits | Decode events; confirm names, versions, feature shapes, and split contents |
| Training input | `PFDataset` transforms momentum targets, batches variable-size events, and optionally constructs particle sets | Matching model recipe, feature schema, masks, and target counts |
| Training | `mlpf/pipeline.py train` optimizes the model and saves checkpoints and configuration | Finite losses, held-out validation, and complete experiment artifacts |
| Inference | The standalone Key4HEP evaluator or detector-specific validation workflow decodes predicted particles | Correct checkpoint/configuration pairing and readable output collections |
| Validation | Particle, jet, missing-momentum, and export tools compare outputs with reference quantities | Physics response and, for deployment, PyTorch/runtime numerical agreement |

The [input and target guide](inputs-targets-and-outputs.md) explains the data carried through these stages; the [architecture guide](model-architectures.md) explains the learned transformation.

For commands, follow [dataset generation](../datasets/generate.md), [dataset validation](../validation/dataset-validation.md), [training](../training/train.md), and [model validation](../validation/overview.md). Keep the checkpoint, saved model configuration, dataset version, and software revision together as described in the [artifact reference](../reference/artifacts-and-directories.md).
