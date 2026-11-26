# ColliderML target-merging audit

This study audits the target allocator on the 100 events of ttbar PU200 shard
`train-00000-of-01000.parquet` from `CERN/ColliderML-Release-1`. It reruns truth
construction, spatial clustering and the actual shared allocator; it does not
infer merges from already-written target directions. Production code is unchanged.

## Configurations

There are two different defaults in this repository:

- `mlpf/data/colliderml/postprocessing.py` and the production specification's empty
  `postprocessing.args` use `bfs_merge`, radii 25/90 mm, fragment fraction 0.25.
- The Flatiron conversion scripts explicitly select `--algorithm clue
  --clue-preset pu200`. Those settings are evaluated separately on the same shard.
- An additional first-20-event control uses `bfs` at 25/90 mm, disabling only the
  fragment merge. This is a diagnostic control, not the older PU200 BFS production
  recipe, which used tighter radii 16.25/36 mm.

All runs use the current `majority_energy` input assignment and current truth
selection, including surrogate ancestors. The production settings of an existing
dataset must be established separately; its name alone does not identify these
clustering parameters.

## Results

**Unconditional merging produces substantial artificial single-particle targets,
including with the production-script CLUE PU200 preset.** This is a target-definition
problem rather than an arithmetic failure of four-vector addition.

Both full-shard runs start from the same 1,464,130 visible truth particles. All
151,173 track-hosted targets are identical between configurations; mergers go
into cluster-hosted targets.

| Measurement, 100 events | CLUE PU200 | CLI `bfs_merge` |
| --- | ---: | ---: |
| Final targets | 1,187,651 | 910,104 |
| Absorbed particles | 276,321 | 553,868 |
| Hosts receiving merges | 124,521 | 17,927 |
| Absorbed truth energy / total final target energy | 5.27% | 23.91% |
| Final targets with pT ≥ 5 GeV | 4,856 | 4,535 |
| Of these, shifted from host by ΔR > 0.3 | 417 (8.59%) | 1,093 (24.10%) |
| Of these, >0.3 from every ≥10%-energy constituent | 306 (6.30%) | 826 (18.21%) |
| Cluster targets with pT ≥ 5 GeV | 2,828 | 2,507 |
| Of these, shifted from host by ΔR > 0.3 | 417 (14.75%) | 1,093 (43.60%) |
| Merge pairs with deposit centroids >300 mm apart | 45,666 (16.53%) | 396,549 (71.60%) |

A stricter indicator directly addressing spatially separated particles requires
both a deposit-centroid separation >300 mm from an absorbed particle and a final
direction shift >0.3. For CLUE it identifies **3,902 targets overall (0.329%)**,
including **409 of 4,856 targets with pT ≥ 5 GeV (8.42%; event-bootstrap 95% interval
7.57–9.31%)**. Those 409 targets occur in 94/100 events and constitute 14.46% of
the high-pT cluster targets. A centroid-angular-separation cut >0.3 instead of the
300 mm cut also identifies 409 high-pT targets; the two selected sets need not be
identical. These are operational pathology indicators, not an independently
validated binary fake-particle classifier.

CLUE has 179,977 host/source pairs separated by momentum ΔR >0.3; 39,770 of them
also have deposit centroids >300 mm apart. Of all 276,321 pairs, 89,389 have a 3D
momentum opening angle >0.3 radians. The difference reinforces why momentum ΔR,
physical opening angle and calorimeter geometry should be audited separately.

The large pre-existing cluster-direction tail is a separate issue. For CLUE,
140 high-pT cluster targets change from aligned to misaligned at ΔR = 0.3, while
147 change in the opposite direction. Consequently, simply counting final
cluster misalignment almost hides the merging problem: 459/2,828 are misaligned
before and 452/2,828 after, despite 417 individual high-pT targets moving by >0.3.
The population here is selected by **final** pT for both measurements.

### Concrete example

In **event 14, original truth host 7822, cluster 15**, CLUE merges 59 particles into
a charged pion host (ultimately neutralized as a cluster target):

- Host pT: **2.27 → 7.54 GeV**; energy: **31.09 → 189.40 GeV**.
- Host-to-sum direction change: **ΔR = 0.714**.
- Cluster alignment: **ΔR = 0.197 → 0.675**.
- Every absorbed constituent has pT ≤0.789 GeV; their deposit centroids are as
  much as **1,051 mm** from the host's centroid within this same cluster.
- The summed target is >0.3 from every constituent carrying ≥10% of its energy.

This example combines spatially spread deposition, a newly high-pT composite,
and degraded alignment to the measured object. It does not rely on the extreme
forward-host cases.

Those extreme cases deserve a separate truth-selection review. For example,
event 13 host 8441 is already a 7,000 GeV forward proton with pT 0.175 GeV,
η=11.29, no track link and only 0.585 GeV of attributed calibrated calorimeter
deposit. It is a regular leaf target (`generatorStatus=1`), not a promoted
surrogate. The charged absolute visibility threshold admits it; the energy-based
assignment then favors retaining it. Merging changes its pT to 9.45 GeV, but the
unlearnable energy/geometry association predates merging. The release lacks a
generator-status field in its particle table, so this audit does not claim that
it is an incoming beam proton. Raw records are saved in `host_truth_examples.json`.

### Controlled clustering comparison

On the same first 20 events:

| Setting | Absorbed particles | pT ≥5 GeV targets shifted by >0.3 | Deposit-centroid pairs >300 mm |
| --- | ---: | ---: | ---: |
| CLI `bfs_merge` | 117,392 | 216 / 906 | 85,298 |
| CLUE PU200 | 58,685 | 95 / 987 | 9,663 |
| BFS without fragment merging, radii 25/90 mm | 16,510 | 0 / 697 | 1 |

The BFS control shows strong sensitivity to cluster extent and absorption. It
does not establish BFS as the best production clusterer: increased shower
fragmentation, target multiplicity, energy response and reconstruction quality
also need evaluation. Counts of merged hosts alone are misleading: the worst
configuration concentrates enormous numbers of sources into fewer hosts.

## Assessment and next experiment

The follow-up [quantitative merge-gate comparison](colliderml-merge-gates.md)
tests shared-cell overlap, shared-cluster containment and deposit/momentum
proximity on all 100 CLUE events, with fixed-cohort rates and energy accounting.

Four-vector merging is defensible for a deliberately defined, nearby unresolved
composite. A shared cluster by itself is insufficient evidence of that condition
at PU200. The current unconditional rule should not be treated as reliable
single-particle truth, and the CLI `bfs_merge` default should not be used for PU200
without a clustering review.

A next controlled target-definition experiment should require geometric and
directional compatibility, explicitly count rejected energy, and separately audit
very forward charged particles with tiny visible deposits and no usable track.
For reference, rejecting CLUE merges with original host/source ΔR >0.3 would
remove **194,792 GeV across this shard, or 3.61% of final target energy**. That is
an energy-accounting cost, not a demonstrated performance improvement; dropping
these contributions silently would replace one target problem with another.
For raw-hit training, independently representing visible particles on their hit
groups is also worth testing instead of imposing the cluster-view one-target-per-
input bottleneck.

No production behavior was changed. All event-level energy and momentum checks
passed; maximum absolute energy residual was 0.00142 GeV for CLUE and 0.0148 GeV
for `bfs_merge`, consistent with float32 accumulation on high-energy events.
The existing merge and ColliderML postprocessing suites passed **21 tests**.

## What the allocator does

After exclusive track/cluster assignment, an unassigned particle chooses the
cluster receiving its largest calibrated deposit. Its **entire truth four-vector**
is added to that cluster's exclusive target. There is no angular compatibility
test, distance test, minimum shared fraction or type/charge compatibility test.
The allocator retains the host's type and charge; its pileup label becomes the
energy-weighted pileup fraction. The converter subsequently maps cluster-hosted
targets to neutral classes and zeroes their cluster-view charge. A particle
without an eligible cluster host is dropped.

The resulting sum is a valid composite four-vector and conserves momentum and
energy, up to floating-point rounding. This alone does not make it a meaningful
single-particle training target. It can have a different direction, invariant mass
and particle content from the host, while retaining the host's identity. Inclusive
hit ownership is also filtered to retained particles rather than transferring all
absorbed particles' hit links into the host. This matters especially when the
merged kinematics are used in the raw-hit target view.

## Measurements

`targets.parquet` stores every retained host's original and final kinematics,
cluster alignment, multiplicity, invariant mass fraction and momentum coherence.
`merge_pairs.parquet` records each original host/source index pair, original
kinematics, deposited energy and deposited-energy centroid separation inside the
shared cluster. Original indices refer to the visibility-selected truth table,
not the raw release particle IDs or the compacted final target axis.

The principal operational indicators are:

- Direction change: final target versus original host, with thresholds
  ΔR = 0.1, 0.3, 0.5 and 1.0. This directly isolates the effect of merging.
- Composite direction: final target more than ΔR = 0.3 from **every constituent
  carrying at least 10% of its final energy**. Groups with no such constituent
  are excluded from this indicator and counted separately.
- Newly misaligned cluster targets: original host within ΔR = 0.3 of its cluster,
  final target outside. The converse measures resolved misalignment. Comparing
  only final cluster alignment would conflate matching errors and merge effects.
- Momentum and geometry separation: original source/host momentum ΔR and 3D
  opening angle, plus the distance between their deposit centroids in the shared
  cluster. Momentum separation alone does not establish spatially separated
  showers: charged particles bend, and low-pT particles can curl or backscatter.
- A new pT ≥ 5 GeV composite with every constituent below 5 GeV. This is not
  automatically a fake target: nearby particles can legitimately form a composite.
  A separate flag also requires a direction change greater than ΔR = 0.3.

Rates use final target pT. All-target and cluster-only denominators are reported
separately. `diagnostics.csv` includes 95% percentile bootstrap intervals from
2,000 resamples of whole events, preserving correlations within each event. These
describe variation within this shard, not systematic uncertainty or representativity
across the dataset. Zero observed failures do not imply a zero population rate.

An angular-veto estimate rejects each absorbed particle farther than a threshold
from its **original host** and sums the removed truth energy. It does not rerun
matching or establish a replacement algorithm; rejecting particles loses target
energy unless another representation is provided.

## Reproduction and artifacts

The four source files already available in `/tmp/colliderml-original-pu200/` were
verified byte-for-byte against SHA256 LFS object IDs in saved Hugging Face API
listings. `inputs.json` records source sizes, hashes, repository paths, code commit
and clustering/assignment settings. The live Hub page was unavailable from this
session; source identity was checked against those saved listings.

```bash
uv run python3 -m scripts.studies.audit_colliderml_merging \
  --source /tmp/colliderml-original-pu200 \
  --output experiments/colliderml_merging_pu200_clue \
  --algorithm clue --clue-preset pu200 --workers 4 --manifest-dir /tmp

uv run python3 -m scripts.studies.audit_colliderml_merging \
  --source /tmp/colliderml-original-pu200 \
  --output experiments/colliderml_merging_pu200 --workers 4 --manifest-dir /tmp

uv run python3 -m scripts.studies.audit_colliderml_merging \
  --source /tmp/colliderml-original-pu200 \
  --output experiments/colliderml_merging_pu200_bfs20 \
  --algorithm bfs --num-events 20 --workers 4
```

The source directory contains `particles.parquet`, `tracks.parquet`,
`calo_hits.parquet` and `tracker_hits.parquet`. The optional manifest directory
contains `colliderml-pu200-{table}-files.json` saved API listings; omit that option
when those listings are unavailable. The reader asserts event-ID alignment.
Each event also checks the recovered exclusive assignment and independently
reconstructs summed momentum and energy from the actual merge lineage.

Pass `--summarize-only` with the same source/output arguments to regenerate plots,
threshold tables, bootstrap diagnostics and examples from stored tables.
`merging_diagnostics.png` compares direction distributions and pre/post alignment;
`example_lineages.png` shows original and summed directions for two high-pT
examples. Large numerical tables and plots are local artifacts under `experiments/`.
