# Publications

The papers below show how MLPF developed from a proof of concept into full detector studies. They are also the primary sources for physics claims in this documentation.

Published numbers should be read together with the detector simulation, dataset, target definition, architecture, and code release used in that paper. Current `main` contains newer data schemas and model options and is not an exact reproduction of every historical setup.

## 2021: first end-to-end MLPF study

[MLPF: Efficient machine-learned particle-flow reconstruction using graph neural networks](https://doi.org/10.1140/epjc/s10052-021-09158-w) introduced full-event particle-flow reconstruction as a supervised, multi-task graph-learning problem. The study used simulated top-pair events with high pileup and reported improved physics response over its rule-based benchmark together with scalable computation.

- [Archived code](https://zenodo.org/records/4559587)
- [Dataset](https://doi.org/10.5281/zenodo.4559324)

## 2021 workshop / 2023 proceedings: CMS implementation

[Machine Learning for Particle Flow Reconstruction at CMS](https://doi.org/10.1088/1742-6596/2438/1/012100) described an MLPF implementation based on CMS tracks and calorimeter clusters. It connected the model to CMS jet and missing-transverse-momentum reconstruction and reported approximately linear runtime and memory scaling with input size.

- [arXiv version](https://arxiv.org/abs/2203.00330)
- [CMS public result from the initial study](https://cds.cern.ch/record/2792320)

## 2022: updated CMS performance

[CMS-DP-2022-061](https://cds.cern.ch/record/2842375) presented an updated CMS MLPF study. It is a public detector-performance note rather than a separate software release. Use it as the source for the plots and setup reported there.

## 2024: scalable models with full CLIC simulation

[Improved particle-flow event reconstruction with scalable neural networks for current and future particle detectors](https://doi.org/10.1038/s42005-024-01599-5) studied electron-positron collisions with full CLICdet simulation. It compared a graph neural network with a kernel-based transformer and avoided model operations that scale quadratically with event size. The best graph model improved jet transverse-momentum resolution by up to 50% relative to the rule-based baseline in that study. The work also tested portability across Nvidia, AMD, and Habana hardware.

- [arXiv version](https://arxiv.org/abs/2309.06782)
- [Archived code](https://zenodo.org/records/10928968)
- [Results](https://doi.org/10.5281/zenodo.10567397)

## 2025: fine-tuning from CLIC to CLD

[Fine-tuning machine-learned particle-flow reconstruction for new detector geometries in future colliders](https://doi.org/10.1103/PhysRevD.111.092015) started with a model trained on CLICdet and fine-tuned it on CLD simulation. In that setup, fine-tuning reached the same performance with about ten times fewer CLD training samples than training from scratch. The fine-tuned model reached event-level performance comparable to the rule-based reconstruction with 100,000 CLD events; training from scratch required at least one million events for a similar result.

- [arXiv version](https://arxiv.org/abs/2503.00131)
- [Archived code](https://zenodo.org/records/14930299)

## 2026: full CMS Run-3 result

[Full event interpretation with machine-learning-based particle-flow reconstruction in the CMS detector](https://doi.org/10.1140/epjc/s10052-026-15754-5) reported MLPF integrated in CMS software and evaluated it in simulation and collision data. In simulated top-pair events under 2023–2024 Run-3 conditions, the paper reports a 10–20% improvement in jet energy resolution for jets with transverse momentum from 30 to 100 GeV. It also reports a median MLPF inference time of 20 ms per event on an Nvidia L4 GPU, compared with about 110 ms for standard CMS particle-flow reconstruction in the stated timing setup.

- [arXiv version](https://arxiv.org/abs/2601.17554)
- [CMS public result](https://cds.cern.ch/record/2937578)
- [Archived code](https://zenodo.org/records/15573658)

## How to cite MLPF

- Cite the paper that matches the detector, method, or transfer-learning result you use.
- Cite a [versioned code archive](https://zenodo.org/search?q=parent.id%3A4452541&f=allversions%3Atrue&l=list&p=1&s=10&sort=version) when reproducibility depends on a specific release.
- Cite the dataset record and its paper when using published data.
- Record the MLPF git commit, dataset version, model configuration, and checkpoint version in derived work.
