# Current capabilities

This page describes the development branch as of 4 September 2026. It summarizes implemented paths, not the performance of any particular checkpoint.

## Detector and input support

| Detector | Collision setting | Inputs | Data version in the default recipe | Status |
|---|---|---|---|---|
| CMS Run 3 | Proton-proton, with and without pileup | Tracks and calorimeter clusters | 3.2.0 | **Supported** |
| CLD | Electron-positron | Tracks and calorimeter clusters | 3.2.1 | **Supported** |
| CLICdet | Electron-positron | Tracks and calorimeter clusters | 3.2.1 | **Supported** |
| CLD | Electron-positron | Tracker and calorimeter hits | 3.2.1 | **Research workflow** |
| CLICdet | Electron-positron | Tracker and calorimeter hits | 3.2.1 | **Research workflow** |
| MAIA | Muon-collider detector study | Postprocessing only | — | **Partial** |
| IDEA | Electron-positron detector concept | Work in progress | — | **Planned** |

The version numbers above come from the default recipes in `particleflow_spec.yaml`. Dataset, code, and checkpoint versions are separate and must be checked together.

## End-to-end operations

| Operation | CMS | CLD | CLIC |
|---|---:|---:|---:|
| Detector simulation configuration | Supported | Supported | Supported |
| Detector-specific postprocessing | Supported | Supported | Supported |
| TFDS creation | Supported | Supported | Supported |
| Short local software test | Supported | Supported | Supported |
| Standard PyTorch training | Supported | Supported | Supported |
| Multi-GPU training | Supported | Supported | Supported |
| Checkpoint loading and fine-tuning | Supported | Supported | Supported |
| Standalone ROOT evaluation | CMS-specific workflow | Supported | Supported |
| Physics validation | CMSSW/site dependent | Supported | Supported |
| ONNX export and comparison | Supported | Supported | Supported |

“Supported” means that code and configuration exist in this repository. Large-scale simulation and physics validation still require the relevant CMS or Key4HEP software environment, computing resources, and detector knowledge.

## Model support

The standard detector recipes currently use an attention model with separate input encodings for different detector-element types. The shared MLPF interface also contains several advanced backbones:

| Model family | Intended use | Status |
|---|---|---|
| Multi-head attention | Default CMS, CLD, and CLIC recipes | **Supported** |
| GNNLSH | Scalable graph processing with locality-sensitive hashing | **Research workflow** |
| HEPT and HEPTv2 | Hash-based efficient particle transformers | **Research workflow** |
| LitePT | Sparse point-transformer studies | **Research workflow**; Nvidia-oriented dependencies |
| Shared backbone and task-query readout | Cross-task model studies | **Research workflow** |

The repository supports single-node distributed PyTorch training. Ray Train and Ray Tune paths are available for distributed training and hyperparameter searches, but they require cluster-specific setup and are treated as research workflows.

## Validation support

The repository can check four different questions:

1. **Dataset integrity:** Are the schemas, detector relationships, target assignments, and energy sums sensible?
2. **Model behavior:** Does the loss converge, and are particle-level predictions well formed?
3. **Physics behavior:** How do particles, jets, and missing momentum compare with reference targets and rule-based reconstruction?
4. **Deployment behavior:** Does an ONNX export agree numerically with PyTorch, and what runtime and memory does it use?

A local smoke test covers only a small part of the first two questions. Published physics claims require the larger validation setup described in the corresponding [paper](publications.md).

## Known boundaries

- Site configurations are not equally mature. Tallinn is the reference full-production setup; LXPlus and local execution need environment-specific review.
- Hit-level events are substantially larger than track/cluster events and require more memory and tuning.
- CMS validation depends on CMSSW and can require experiment data, calibrations, and site services.
- A checkpoint is only meaningful with its saved model configuration and a compatible dataset schema.
- Open research branches and roadmap items are not part of the supported interface until they are merged and documented.
