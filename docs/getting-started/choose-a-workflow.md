# Choose a workflow

The shortest route depends on the artifact you already have.

| Goal | What you need | Recommended route |
|---|---|---|
| Check the installation | A clean checkout | Run one [quickstart](quickstart.md) smoke test |
| Understand the method | Nothing | Read [Particle flow and MLPF](../concepts/particle-flow-and-mlpf.md) |
| Evaluate a published model | A compatible checkpoint and ROOT file | Download the model and use the detector-specific standalone evaluator |
| Train a model | A prepared TFDS dataset | Download one dataset split, then run `mlpf ... train` |
| Reproduce a dataset | Generator configuration and detector software | Run simulation, postprocessing, validation, then TFDS conversion |
| Validate CMS reconstruction | CMSSW output | Use the CMS validation workflow |
| Validate CLD or CLIC reconstruction | EDM4hep ROOT plus a checkpoint | Use the Key4HEP standalone evaluator |
| Compare deployment formats | A checkpoint, `model_kwargs.pkl`, and TFDS | Run PyTorch-to-ONNX numerical and timing validation |

## Three levels of use

### 1. Use published artifacts

This is the recommended starting point. Download a dataset or checkpoint and avoid detector simulation. It is the quickest route to model evaluation and training experiments.

### 2. Run a local smoke workflow

The scripts in `scripts/local_test_*.sh` download two small input files, build a small dataset, and run a short CPU training. They check that the software components fit together. They do not establish physics performance.

### 3. Produce data at scale

The Pixi and Snakemake workflow covers simulation, postprocessing, TFDS creation, training, and validation. A full campaign can use many batch jobs and substantial storage. It requires a reviewed site configuration and detector software access.

## Detector choice

- Choose **CMS** for proton-proton collisions in the CMS Run-3 software and detector environment.
- Choose **CLD** for electron-positron studies with the CLD detector concept and Key4HEP/EDM4hep data.
- Choose **CLIC** for electron-positron studies with CLICdet and Key4HEP/EDM4hep data.

CLD and CLIC support both reconstructed track/cluster inputs and lower-level tracker/calorimeter hit inputs. The hit workflows have much larger events and are best treated as research workflows. See [Current capabilities](../science/capabilities.md).
