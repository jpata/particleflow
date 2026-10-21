# Linear probes of hit instance information

Run four independent probes: tracker/raw, tracker/embedding,
calorimeter/raw, and calorimeter/embedding. The default frozen checkpoint is
`cld_pf_heptv2_hit_grouping_50k/hit_heptv2_grouped_seed12345_20261009_164038_591219/checkpoints/checkpoint-50000.pth`.
This is the completed grouped model with three-process test particle F1 0.7262
and mean jet-response IQR 0.1181 in the October comparison. The newer tracking
ablation campaign only has 5k intermediate validation results at implementation time.

## What is being tested

`particle_number` is an event-local instance ID, not a semantic class. Training
a classifier with particle IDs as fixed classes would be ill-defined across
events. Instead, for two hits with representations h_i and h_j, predict

```
P(same particle | i, j) = sigmoid(w · standardize(abs(h_i - h_j)) + b).
```

This is a linear logistic probe on fixed, symmetric pair differences. It is not
a linear classifier on individual hits, nor a learned nonlinear clustering
network. Standardization is fitted on probe-training pairs only. Raw inputs use
all original X columns, without truth features or learned feature engineering;
embeddings are the full `encode_backbone` representation immediately before the
set decoder. The frozen encoder sees the entire event, including both detectors
and background, even though probes are trained separately for each detector.
The raw probe has less context and fewer parameters than the embedding probe;
this comparison tests accessible information, not equal model capacity.

A common graph uses the union of 16 nearest neighbors in xyz and 16 in
(eta, sin(phi), cos(phi)), within each detector. It is built from raw features
without particle IDs, so candidate edges are identical for the raw and embedding
probes. Classify edges, threshold their scores, and take connected components
as instances. No true particle count is supplied. Singleton components are kept.
The graph avoids a quadratic all-pairs tensor, but can miss long-range tracker
links, and connected components can merge instances through one false edge.

**Scope:** only hits with positive particle_number are evaluated. This uses an
oracle foreground mask and excludes background rejection from the task. No
representative-hit/PID filter is applied: related hits with zero PID but positive
particle_number remain included. Both approaches get the same foreground hits.
Calorimeter labels are hard particle ownership, not fractional energy ownership.

## Event splits and fitting

The default dataset is `cld_edm_ttbar_hits/1:3.2.1`, TFDS `test` split. This split
was not used for backbone gradient updates, but may have been used for its
validation/model selection. Results are a representation diagnostic, not an
independent final physics benchmark.

Shuffle the selected event range with a fixed seed, then assign disjoint events
to probe training, validation, and testing. IDs never link across events.
Subsample at most 8192 candidate edges/event for training, use the same edge
sample for both representations, and balance positive/negative loss weights per
event. Ten epochs of averaged SGD fit the logistic classifiers. The fixed L2
coefficient is 1e-4; no hyperparameters are selected using probe-test events.
Choose the edge threshold separately per probe by mean validation adjusted Rand
index (ARI). Test edge AP/AUC use all candidate edges with their natural class
balance, not the balanced training sample.

Report equal-event means and individual event measurements for:

- ARI: permutation-invariant agreement of full instance partitions.
- B-cubed precision/recall/F1: hit-weighted purity/completeness within each event.
- Instance F1 at strict IoU > 0.5: uniquely matched predicted/truth instances.
- Edge average precision and ROC AUC, plus candidate positive fraction.
- Oracle-graph ARI: keep exactly same-particle candidate edges. This diagnoses
  missing graph connectivity independently of the learned classifier.
- Predicted and true instance counts and hit counts.

Thresholded connected components and oracle ARI must be read alongside edge
metrics. A good edge probe can still yield bad partitions through false bridges.

## Local execution

From the repository root, with GPU access (outside the Codex sandbox):

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 uv run python scripts/studies/probe_hit_instances.py \
  --output experiments/hit_instance_probe_50k \
  --train-events 64 --valid-events 16 --test-events 32
```

The initial local run uses 32/8/16 events. All extraction is batch size one,
in inference mode with bfloat16 CUDA autocast; CPU logistic fitting uses cached
float32 representations. A fresh output directory is required. Outputs contain
checkpoint SHA256, CLI arguments, explicit event IDs, per-event feature/edge
caches, fitted scalers/classifiers/thresholds, and `results.json`. The script
strictly loads the saved model state, checks finite features, and never updates
the backbone. Use `--dataset`, `--offset`, `--neighbors`, `--epochs`, and
`--pairs-per-event` for follow-up studies, retaining independent event splits.

## Initial local result (2026-10-12)

Completed outside the sandbox on the local RTX 5060 Ti, using 32 training,
8 validation, and 16 test ttbar events, seed 12345, ten probe epochs.
Output: `experiments/hit_instance_probe_50k_initial/` (including `comparison.png`).

| Detector | Input | Edge AP | Instance ARI | B-cubed F1 | Instance F1, IoU > 0.5 |
|---|---|---:|---:|---:|---:|
| Tracker | Raw | 0.3566 | 0.1112 | 0.2911 | 0.0143 |
| Tracker | Embedding | 0.6875 | 0.2576 | 0.4067 | 0.0719 |
| Calorimeter | Raw | 0.9682 | 0.5379 | 0.7305 | 0.1434 |
| Calorimeter | Embedding | 0.9740 | 0.6407 | 0.7742 | 0.1742 |

Both detectors selected threshold 0.70 for raw features and 0.95 for embeddings.
Candidate-edge same-particle prevalence is 17.8% for tracker and 89.1% for
calorimeter; compare AP within each detector, not across detectors.
Oracle-graph ARI is 0.8204 for tracker and 0.9767 for calorimeter.

Frozen embeddings improve instance ARI by 0.1464 (tracker) and 0.1028
(calorimeter). However, the component decoder remains highly fragmented:
embedding probes predict 354 tracker instances versus 61 truth instances/event,
and 762 calorimeter instances versus 93 truth instances/event. These counts
include singleton fragments. The low strict-IoU instance F1 shows that the
embeddings are informative but this simple probe is not yet an effective
standalone reconstruction algorithm. A larger event sample, finer validation
threshold search, and graph/clustering ablations are appropriate follow-ups.

Reproduce the initial run in a fresh output directory:

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 uv run python scripts/studies/probe_hit_instances.py \
  --output experiments/hit_instance_probe_50k_repeat \
  --train-events 32 --valid-events 8 --test-events 16 --epochs 10
```

## Why calorimeter edge AP is high

`PYTHONPATH=. .venv/bin/python scripts/studies/analyze_calorimeter_probe.py`
recomputes a diagnostic on the same 16 held-out probe events, without training
or threshold retuning. Results are in
`experiments/hit_instance_probe_50k_initial/calorimeter_edge_audit.json`.

The mean same-particle fraction is 2.86% among all unordered foreground hit pairs,
but 89.06% among candidate edges. Spatial-only candidates are 91.41% positive,
direction-only graph candidates 89.47%, and edges present in both graphs 94.87%.
Here spatial-only refers to constructing just the spatial graph, rather than
excluding its overlap with the direction graph.

| Ranking | Edge AP | ROC AUC | AP reweighted to 50% positives |
|---|---:|---:|---:|
| Negative spatial distance, no training | 0.9619 | 0.7993 | 0.7818 |
| Negative directional distance, no training | 0.9533 | 0.7620 | 0.7419 |
| Raw-feature probe | 0.9682 | 0.8300 | 0.8198 |
| Embedding probe | 0.9740 | 0.8527 | 0.8450 |

A constant-score predictor has AP equal to prevalence, 0.8906. Thus the high AP
partly reflects the candidate construction and dense spatial shower structure.
The learned probes add discrimination beyond geometry, with embeddings improving
on raw features, but AP close to one does not establish near-perfect boundaries.

At the validation-selected clustering thresholds, raw/embedding precision is
95.97%/97.69%, recall is 76.12%/57.97%, and false-positive rate among
*different-particle edges* is 25.16%/10.41%. Mean false links per event are
3943/1655. These are separate operating points, not a matched-recall comparison.
False links can merge particles, while discarded true links can fragment them.

For a diagnostic subset where both endpoints have at least one different-particle
candidate neighbor, prevalence falls to 74.07%. AP becomes 0.9030/0.9123 and
balanced AP 0.7826/0.7999. The subset uses truth only for evaluation, not inference.
All numbers are equal-event means. This remains conditional on the oracle
foreground mask and hard particle-number labels.
