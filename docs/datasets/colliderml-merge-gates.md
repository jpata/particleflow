# Quantitative comparison of ColliderML merge gates

This follows the [target-merging audit](colliderml-target-merging.md), using the
same full 100-event ttbar PU200 shard, CLUE PU200 clustering, and
`majority_energy` exclusive assignment. All 276,321 original merge proposals are
evaluated against their **original exclusive host**. Accepted four-vectors are
summed again; rejected particles are explicitly counted as lost target energy.
There is no rematching, alternate-host search, sequential change of host, or
change to production behavior.

## Definitions

For host `a`, source `b` and their selected cluster `c`, let `w_ai` be particle
`a`'s calibrated contribution to hit cell `i`, and `D_ac` its total deposit in
cluster `c`. The compared families are:

- **Both contained:** each particle deposits at least 50% of its total attributed
  calorimeter energy in `c`. This tests shared-cluster containment, not spatial
  overlap inside that cluster.
- **Hit-energy overlap:** `O = sum_i min(w_ai/D_ac, w_bi/D_bc)` over hit cells
  in `c`. It is 0 for disjoint cells and 1 for identical normalized deposited
  energy distributions. Absolute particle energy does not set this score.
- **Shared support:** the smaller of the two deposited-energy fractions on cells
  receiving contributions from both particles. Unlike `O`, it ignores differing
  energy shapes on those common cells.
- **Centroid proximity:** either Euclidean distance in mm or angular ΔR between
  the two particles' deposited-energy centroids **within `c`**. Positions are
  measured in the calorimeter, not at the production vertex. Vertex proximity
  would be a poor discriminator in this high-pileup sample.
- **Momentum proximity:** ΔR between the original truth momentum directions.
  This directly constrains the direction of the composite; it is distinct from
  proximity of bent or backscattered particles' calorimeter deposits.

The script also evaluates both particles contributing at least 10% of the whole
cluster energy, overlap-or-centroid proximity, and overlap/centroid gates combined
with momentum proximity. These additional results are in `comparison.csv`.

## Comparison on a fixed target population

The main direction indicator is a change from the original host greater than
ΔR = 0.3. Its denominator is always the **same 4,856 targets that had final
pT ≥5 GeV under unconditional merging**, even when a gate subsequently lowers
their pT. This avoids mistaking removal from the high-pT selection for a cure.
The energy denominator is the original full target energy, 5,394,281 GeV across
the shard; original allocation drops are unchanged and excluded from that total.

| Accept merge when | Proposed merges retained | Shifted targets, fixed cohort | Rejected total target energy | Rejected hard-scatter target energy |
| --- | ---: | ---: | ---: | ---: |
| Always: current behavior | 100.00% | 417 (8.59%) | 0 | 0 |
| Both contained ≥50% | 61.72% | 226 (4.65%) | 2.17% | 0.589% |
| Hit-energy overlap ≥10% | 2.25% | 31 (0.64%) | 4.99% | 1.395% |
| Hit-energy overlap ≥25% | 0.58% | 11 (0.23%) | 5.20% | 1.468% |
| Shared-hit support ≥25% for both particles | 1.17% | 15 (0.31%) | 5.08% | 1.415% |
| Centroid ΔR <0.1 | 60.77% | 161 (3.32%) | 3.75% | 1.013% |
| Centroid ΔR <0.2 | 77.13% | 227 (4.67%) | 2.24% | 0.575% |
| Centroid distance <200 mm | 73.19% | 267 (5.50%) | 2.34% | 0.680% |
| Momentum ΔR <0.3 | 34.87% | 0 (0%) | 3.61% | 0.986% |
| Centroid ΔR <0.1 AND momentum ΔR <0.3 | 28.55% | 0 (0%) | 4.65% | 1.273% |
| Never merge: reference | 0 | 0 | 5.27% | 1.496% |

Hard-scatter energy is identified constituent by constituent from the original
`ispu`, so each percentage in that column uses **hard-scatter target energy** as
its own denominator. Rejected pileup energy is not charged to the hard scatter.

Whole-event bootstrap intervals from 2,000 paired resamples support the observed
reductions. For example, the fixed-cohort shifted rate is 7.72–9.49% for current
behavior, 4.01–5.33% for containment, 2.78–3.86% for centroid ΔR <0.1, and
0.42–0.89% for hit overlap ≥10%. Intervals characterize this shard's event
variation, not a separate validation set or detector/systematic uncertainty.
The zero momentum-gate result is largely expected by construction and does not
prove all remaining targets are physically correct.

## What these results mean

**Sharing a substantial part of a cluster is not enough.** Requiring both
particles to put half their deposit in it only reduces the large-shift count
from 417 to 226. A broad cluster can contain both showers almost completely.

**Exact shared-cell overlap is too restrictive as a sole rule.** At a 25%
threshold it rejects 99.42% of all proposed merges and retains only 0.80% of
pairs whose original momentum directions differ by <0.1. It approaches the
never-merge reference rather than selecting a useful unresolved-composite
population. Distinct particles can hit different cells even when their showers
are nearby. A neighborhood-smoothed overlap could be tested later, but was not
evaluated here.

**Deposit geometry improves matters, but does not control the truth direction.**
Centroid ΔR <0.1 retains 95.69% of pairs already close in momentum ΔR <0.1, while
reducing the large-shift count by 61.4%. It still leaves 161 shifted targets in
the original cohort. All 161 end below 5 GeV, so an evaluation restricted to the
gate's own resulting high-pT targets would misleadingly report zero failures.
The resulting high-pT target count falls from 4,856 to 3,552.

Threshold choice matters: centroid ΔR <0.3 retains 87.67% of merges and loses only
1.21% of total energy, but still leaves 254 shifted targets in the fixed cohort.
A distance cut of 100/200/300 mm leaves 145/267/349 shifted targets while losing
3.86/2.34/1.33% of energy. There is no detector-independent optimal distance
threshold established by this study.

**Momentum proximity is the most effective tested guard for this particular
pathology.** At ΔR <0.3 it removes all >0.3 direction changes here while keeping
all originally <0.1 momentum-separated pairs. It rejects 194,792 GeV of energy,
including 0.986% of the hard-scatter target energy. The accepted target set still
has 3,983 targets above 5 GeV. Adding a strict centroid cut costs another
1.04 percentage points of total energy without improving the large-shift count
in this sample. Requiring hit overlap OR centroid ΔR <0.1 changes very little
relative to centroid proximity alone: 163 shifted targets versus 161.

These are not free fixes. For the momentum ΔR <0.3 rule, rejected contributions
are 4.33% of the original scalar sum of particle pT. Their net transverse vector
per event has median magnitude **9.27 GeV** and 90th percentile **17.88 GeV**.
Those numbers measure the change to truth target momentum balance, not model
MET resolution. A production change must decide explicitly whether rejected
particles are dropped with accounting, represented separately, or assigned to
another compatible input.

## Alignment to the measured cluster and an example

Controlling motion from the host does not automatically fix a poor original
host-to-cluster match. On the same fixed cohort, 452 targets are misaligned from
their cluster by >0.3 under the current rule. This becomes 413 with the momentum
gate, but **562 with centroid ΔR <0.1**: removing an absorbed momentum can undo a
fortuitous alignment correction. These changes cannot be assessed using the
host-shift indicator alone.

For event 14, host 7822, the concrete example from the initial audit:

| Gate | Sources accepted /59 | Resulting pT [GeV] | Host direction shift | Cluster ΔR |
| --- | ---: | ---: | ---: | ---: |
| Current unconditional | 59 | 7.54 | 0.714 | 0.675 |
| Both contained ≥50% | 50 | 7.22 | 0.489 | 0.511 |
| Hit overlap ≥10% | 2 | 2.28 | 0.083 | 0.169 |
| Centroid ΔR <0.1 | 6 | 3.08 | 0.063 | 0.182 |
| Centroid distance <200 mm | 14 | 2.85 | 0.504 | 0.533 |
| Momentum ΔR <0.3 | 13 | 5.41 | 0.034 | 0.163 |

Here momentum compatibility preserves a high-pT, well-aligned composite rather
than merely suppressing it below the selection threshold.

## Recommended next step and reproduction

The [jet-response study](colliderml-merge-jet-response.md) compares current,
momentum ΔR <0.3 and never-merge targets on eleven PU200 shards, with truth-jet
matching efficiency, common matched cohorts and explicit pileup reference views.

Prototype an opt-in **original-host momentum ΔR <0.3** merge gate with explicit
rejected-particle accounting. Keep the existing behavior selectable and compare
jet/MET targets and downstream learning on an independent shard. Record
centroid separation and overlap as diagnostics; the tested strict centroid/overlap
AND cuts do not justify their extra energy cost for direction control. This does
not resolve the earlier audit's very forward, weakly measured original hosts.
No production target-building code was changed in this study.

```bash
uv run python3 -m scripts.studies.compare_colliderml_merge_gates \
  --source /tmp/colliderml-original-pu200 \
  --audit experiments/colliderml_merging_pu200_clue \
  --output experiments/colliderml_merge_gates --workers 4

# Reuse stored hit-overlap extraction and regenerate all comparisons:
uv run python3 -m scripts.studies.compare_colliderml_merge_gates --compare-only
```

Artifacts are in `experiments/colliderml_merge_gates/`: `overlaps.parquet`,
`pair_metrics.parquet`, `comparison.csv`, `events.csv`, `bootstrap.csv`,
`examples.csv`, `inputs.json`, and `tradeoff.png`. Input provenance includes the
original shard hashes and the comparison script hash. The unconditional replay
reproduces the audit's target pT and shifts within floating-point tolerance;
every rule checks accepted plus rejected energy against the original total.
The overlap extraction checks all original source pT values and both particles'
cluster deposits against the saved lineage. Four numerical tests check the
normalization and interpretation of the overlap coefficient.
The combined overlap, existing merge and ColliderML postprocessing checks passed
**25 tests**; formatting, lint and whitespace checks also passed.
