# Machine-learned particle flow

MLPF reconstructs the particles produced in a collision from detector measurements. It uses a neural network to combine information from tracking detectors and calorimeters and produces a list of particle candidates for the full event. The main workflows cover the CMS detector and the CLD and CLIC future-collider detector concepts. Research workflows add the open ColliderML proton-proton dataset, the IDEA and MAIA detector concepts, and lower-level hit inputs.

This documentation describes the codebase prepared for **v3.3.0**: supported workflows, configuration, and validation limits. Dataset schemas and model artifacts have independent versions. Published physics results remain tied to their original code, data, detector setup, and checkpoints.

## Where should I start?

| If you want to... | Start here |
|---|---|
| Understand particle flow and MLPF | [Particle flow and MLPF](concepts/particle-flow-and-mlpf.md) |
| Check that the repository works | [Quickstart](getting-started/quickstart.md) |
| Choose between evaluation, training, and data production | [Choose a workflow](getting-started/choose-a-workflow.md) |
| Download a prepared training dataset | [Dataset download](datasets/download.md) |
| Produce a dataset from simulation | [Dataset generation](datasets/generate.md) |
| Train or continue a model | [Train a model](training/train.md) |
| Balance training samples and monitor their losses | [Dataset mixing](training/data-mixing.md) |
| Run a reproducible training campaign on a cluster | [Training scenarios](training/scenarios.md) |
| Evaluate a published CLD or CLIC model | [Key4HEP evaluation](validation/key4hep.md) |
| Choose the right validation check | [Validation overview](validation/overview.md) |
| Export and compare an ONNX model | [ONNX validation](validation/onnx.md) |
| See which detectors and features are implemented | [Current capabilities](science/capabilities.md) |
| Read the scientific results | [Publications](science/publications.md) |
| Understand remaining limits and priorities | [Development priorities](science/roadmap.md) |
| Find a code component | [Repository map](map.md) |
| Read the detailed hit-model studies | [Research notes](studies/index.md) |

## The workflow at a glance

```text
detector simulation
        |
        v
tracks, calorimeter clusters, or detector hits
        |
        v
detector-specific postprocessing and data checks
        |
        v
versioned TensorFlow Dataset (TFDS)
        |
        v
MLPF training and checkpoint
        |
        v
particle predictions
        |
        v
particle, jet, missing-momentum, and deployment validation
```

The full workflow is available for CMS, CLD, and CLIC. Published datasets and model checkpoints from the [MLPF Hugging Face repositories](https://huggingface.co/jpata) provide the shortest starting path. Each release publishes reference checkpoints trained with the same recipe, for example under `cld/clusters/v3.3.0/` or `cms/v3.3.0/` in the model repository.

For a first hands-on use, download the published CLD dataset and follow the [short training path](training/train.md). For an existing checkpoint, start with the [standalone Key4HEP evaluator](validation/key4hep.md).

## Support levels

The documentation uses four labels:

- **Supported**: there is a maintained configuration and a tested path in this repository.
- **Research workflow**: the implementation is usable, but it needs more detector, model, or computing knowledge.
- **Partial**: an identified subset of the end-to-end workflow is implemented.
- **Planned**: the work describes a proposed or developing future capability.

See [Current capabilities](science/capabilities.md) for the detailed matrix.

## Project links

- [Source code](https://github.com/jpata/particleflow)
- [Datasets](https://huggingface.co/datasets/jpata/particleflow)
- [Models](https://huggingface.co/jpata/particleflow)
- [Issues](https://github.com/jpata/particleflow/issues)
- [Discussions](https://github.com/jpata/particleflow/discussions)
