# Machine-learned particle flow

MLPF reconstructs the particles produced in a collision from detector measurements. It uses a neural network to combine information from tracking detectors and calorimeters and produces a list of particle candidates for the full event. The project supports studies with the CMS detector and with the CLD and CLIC future-detector concepts.

This documentation describes the current development version of the repository. A result reported in a paper belongs to the code, data, detector setup, and model used in that study; it is not automatically a performance claim for the current branch.

## Where should I start?

| If you want to... | Start here |
|---|---|
| Understand particle flow and MLPF | [Particle flow and MLPF](concepts/particle-flow-and-mlpf.md) |
| Check that the repository works | [Quickstart](getting-started/quickstart.md) |
| Choose between evaluation, training, and data production | [Choose a workflow](getting-started/choose-a-workflow.md) |
| See which detectors and features are implemented | [Current capabilities](science/capabilities.md) |
| Read the scientific results | [Publications](science/publications.md) |
| Understand planned work | [Roadmap](science/roadmap.md) |
| Find a code component | [Repository map](map.md) |

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

The full workflow is available for CMS, CLD, and CLIC. New users normally do not need to generate detector simulation: published datasets and model checkpoints are available from the [MLPF Hugging Face repositories](https://huggingface.co/jpata).

## Support levels

The documentation uses four labels:

- **Supported**: there is a maintained configuration and a tested path in this repository.
- **Research workflow**: the implementation is usable, but it needs more detector, model, or computing knowledge.
- **Partial**: only part of the end-to-end workflow is implemented.
- **Planned**: the work is proposed or under development and is not a current capability.

See [Current capabilities](science/capabilities.md) for the detailed matrix.

## Project links

- [Source code](https://github.com/jpata/particleflow)
- [Datasets](https://huggingface.co/datasets/jpata/particleflow)
- [Models](https://huggingface.co/jpata/particleflow)
- [Issues](https://github.com/jpata/particleflow/issues)
- [Discussions](https://github.com/jpata/particleflow/discussions)
- [Documentation plan](https://github.com/jpata/particleflow/issues/500)
