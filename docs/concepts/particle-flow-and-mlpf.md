# Particle flow and MLPF

## What particle flow reconstructs

A collision produces particles that pass through a detector. The detector does not record the particles directly. It records signals such as:

- curved tracks left by charged particles in a tracking detector;
- energy deposits in electromagnetic and hadronic calorimeters; and
- signals in dedicated systems, such as muon detectors.

Particle-flow reconstruction combines these signals into one list of reconstructed particles. Each particle has a type, such as charged hadron, neutral hadron, photon, electron, or muon, and an estimated momentum and energy.

The difficult part is deciding which detector signals came from the same particle and which came from nearby particles. The number of signals and particles also changes from event to event. Good particle reconstruction matters because jets, missing transverse momentum, and many later analysis quantities are built from this particle list.

## Where particle flow came from

Particle flow did not begin as one standard algorithm with a single name. A widely cited early implementation was the “energy-flow” reconstruction used by ALEPH at the LEP electron-positron collider. It combined tracking and calorimeter measurements to build charged and neutral reconstructed objects. The approach and its detector performance are described in the [1995 ALEPH detector paper](https://doi.org/10.1016/0168-9002(95)00138-7).

The idea was developed further for future linear colliders. A [2001 study by Brient and Videau](https://arxiv.org/abs/hep-ex/0202004) argued for measuring charged particles, photons, and neutral hadrons with the detector subsystem best suited to each one. This motivated highly granular calorimeters that can separate nearby particle showers. [PandoraPFA](https://doi.org/10.1016/j.nima.2009.09.009) later provided a detailed algorithm and a systematic study of this approach for linear-collider detectors.

CMS adapted particle flow to the more crowded environment of a proton-proton collider. Its [particle-flow reconstruction paper](https://doi.org/10.1088/1748-0221/12/10/P10003) explains how tracks, calorimeter clusters, and muon information are linked into a global event description. These earlier algorithms provide the physics starting point for MLPF.

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

## Related machine-learning approaches

MLPF is one of several ways to formulate learned particle reconstruction. The approaches below solve closely related problems, but they differ in their detector inputs and in how they decide how many particles to produce.

- [Object condensation](https://doi.org/10.1140/epjc/s10052-020-08461-2) learns a space in which detector inputs belonging to the same object collect around a representative point. It does not require a fixed number or ordering of output objects. A later [high-granularity calorimeter study](https://arxiv.org/abs/2106.01832) used object condensation with a graph neural network to cluster detector hits and predict particle-shower properties in one model.
- [HGPflow](https://doi.org/10.1140/epjc/s10052-023-11677-7) treats detector inputs as nodes and reconstructed particles as hyperedges that can collect information from several nodes. Its learned incidence matrix describes how detector measurements contribute to particles. The first study focused on particles inside individual jets; a [later full-event study](https://doi.org/10.1140/epjc/s10052-025-14443-z) applied the method to proton-proton and electron-positron collisions.
- [HitPF](https://arxiv.org/abs/2603.04084) works directly from charged-particle tracks and calorimeter and muon hits, without a separate hand-designed calorimeter clustering stage. It combines a geometric-algebra transformer with object-condensation-based clustering, followed by particle identification and energy regression.

These references provide context for the wider ML reconstruction field. They do not imply that the complete HGPflow, HitPF, or object-condensation models are implemented as options in this repository.

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
