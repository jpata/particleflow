# Validation overview

MLPF validation builds evidence in four layers, from software and data integrity through scientific and deployment performance. Each layer answers one clear question.

| Layer | Question | Typical evidence | Minimum time scale |
|---|---|---|---|
| Dataset integrity | Are files readable and are detector-to-particle relationships internally sensible? | Schema gates, counts, energy closure, assignment and geometry checks | Minutes to hours |
| Model behavior | Does training remain finite and do predictions have valid shapes and values? | Training/validation losses, checkpoint reload, particle-level distributions | Minutes to days |
| Physics behavior | Are reconstructed particles, jets, and missing momentum suitable for the intended analysis? | Response, resolution, efficiency, multiplicity, jet and MET comparisons | Hours to campaign scale |
| Deployment behavior | Does an exported model agree with PyTorch and fit runtime and memory constraints? | Numerical differences, failure counts, latency and memory summaries | Minutes to hours |

The time ranges are qualitative and depend on sample size, detector, storage, and hardware.

## 1. Dataset integrity

Run this before training on newly produced data. The common Parquet validator checks that arrays load, schemas and row relationships agree, energies close within defined gates, target assignments are sensible, and hit geometry is consistent when present. See [Dataset validation](dataset-validation.md).

Passing these checks establishes that the produced representation satisfies the tested invariants. Scientific review of the detector simulation, target definitions, and selection choices provides the next layer of evidence.

## 2. Model behavior

A short training checks loading, tensor shapes, forward and backward passes, and checkpoint writing. Longer training should monitor total and task losses on both training and validation data. Reload at least one checkpoint and run inference on held-out events.

A decreasing loss establishes optimization progress. Particle species, multiplicity, kinematics, and failure/outlier populations establish whether that progress produces well-formed reconstruction behavior.

## 3. Physics behavior

Physics validation asks whether the reconstructed event is useful for measurements. Compare MLPF with the reconstructable target and, where available, the detector's rule-based particle-flow algorithm. Study particles before jets so jet-level changes can be traced to reconstruction behavior.

Use [Key4HEP validation](key4hep.md) for CLD and CLIC. CMS validation requires the [CMSSW workflow](cms.md), including corrections and data-quality inputs. A new checkpoint receives its performance claims from its own validation; each publication's claims remain tied to that publication's code, data, model, and detector setup.

## 4. Deployment behavior

Exporting to ONNX changes the execution implementation. Compare outputs numerically on real, variable-size events before measuring speed. Then record latency, throughput, peak memory, precision, event-size distribution, warm-up policy, and hardware. See [ONNX validation](onnx.md).

## Minimum gate before sharing an artifact

For a dataset, publish schema/provenance metadata and a passing integrity report on representative shards. For a checkpoint, keep its configuration and dataset version, demonstrate reload and held-out inference, provide particle-level validation, and explicitly list completed and pending physics and deployment checks.
