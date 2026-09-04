# Particle flow and MLPF

## What particle flow reconstructs

A collision produces particles that pass through a detector. The detector does not record the particles directly. It records signals such as:

- curved tracks left by charged particles in a tracking detector;
- energy deposits in electromagnetic and hadronic calorimeters; and
- signals in dedicated systems, such as muon detectors.

Particle-flow reconstruction combines these signals into one list of reconstructed particles. Each particle has a type, such as charged hadron, neutral hadron, photon, electron, or muon, and an estimated momentum and energy.

The difficult part is deciding which detector signals came from the same particle and which came from nearby particles. The number of signals and particles also changes from event to event. Good particle reconstruction matters because jets, missing transverse momentum, and many later analysis quantities are built from this particle list. This role of particle flow is described in the [first MLPF paper](https://doi.org/10.1140/epjc/s10052-021-09158-w) and in the [CMS implementation study](https://doi.org/10.1088/1742-6596/2438/1/012100).

## What MLPF changes

Traditional particle-flow algorithms use a sequence of detector-specific rules. MLPF learns the reconstruction from simulated examples.

For each event, the model receives a variable-length set of detector elements. Depending on the detector and dataset, these elements can be tracks and calorimeter clusters or lower-level tracker and calorimeter hits. The network lets elements exchange information, then predicts particle identity and kinematics. Outputs classified as “no particle” are removed from the final list.

Training is supervised: simulated particles provide targets for the detector signals. The loss combines several tasks, including deciding whether a particle is present, identifying its type, and estimating its kinematics. The exact target definition is detector-specific and is part of the data-processing pipeline.

The original study used a graph neural network and showed that full-event reconstruction can be learned as one task. Later work explored scalable graph and transformer models with full detector simulation. See the [2021 method paper](https://doi.org/10.1140/epjc/s10052-021-09158-w) and the [2024 future-collider study](https://doi.org/10.1038/s42005-024-01599-5).

## Why use machine learning here?

MLPF is intended to learn correlations across the whole event while running efficiently on parallel hardware. The scientific case has developed in several steps:

- The first study established the end-to-end learning problem in simulated high-pileup collisions.
- The CMS studies placed MLPF inside the experiment software and evaluated particles, jets, missing momentum, and runtime.
- The full-simulation CLIC study compared scalable model families and accelerator platforms.
- The CLD study showed that a model trained for one detector can be fine-tuned for another detector with less training data.

These findings are specific to their published setups. The [publications page](../science/publications.md) summarizes each result and links the matching archived code and data where available.

## From detector signals to physics quantities

```text
tracks + calorimeter information
              |
              v
       MLPF particle list
          /         \
         v           v
       jets     missing momentum
```

A model can have a low training loss and still produce biased physics quantities. Validation therefore happens at several levels:

1. Check that detector elements and training targets are internally consistent.
2. Check particle identification and momentum response.
3. Reconstruct jets and missing momentum and compare them with reference particles.
4. Check exported models against PyTorch and measure runtime and memory.

## Terms used in this documentation

**Detector element**
: One input item, such as a track, calorimeter cluster, or detector hit.

**Target particle**
: The simulated particle assigned to the detector inputs for supervised training. This is a reconstructable target, not necessarily every particle produced by the event generator.

**Particle candidate**
: One particle predicted by MLPF or reconstructed by a reference particle-flow algorithm.

**Jet**
: A group of nearby reconstructed particles, usually associated with a quark or gluon produced in the collision.

**Missing transverse momentum**
: The momentum imbalance in the plane transverse to the beam. It is inferred from the reconstructed visible particles.

**Pileup**
: Additional proton-proton collisions recorded with the collision of interest. Pileup is important for the CMS workflow and is not present in the same form in the electron-positron samples used here.
