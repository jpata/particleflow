# ColliderML target assignment: jet fidelity and event energy balance

The two objectives need not conflict. Repartitioning a fixed particle set using
exact four-vector sums preserves total energy and momentum automatically, while
jet fidelity depends strongly on the partition. The current visibility and
allocator cuts introduce a separate energy deficit that repartitioning cannot fix.

I compared eight target definitions on the cached **1,000 PU0 and 1,000 PU200
events** from the [updated-chain audit](colliderml-updated-chain.md). The reference
is held fixed: accepted measurable truth at `f490a6e8`, including pileup and
excluding neutrinos, after the |η|>3.5 cut. These results evaluate target
definitions, not trained-model reconstruction or detector-compatible assignments.
Production code is unchanged.

Plots: [jet distributions](../../experiments/colliderml_target_alternatives/jet_distributions.png)
and [conservation/multiplicity comparison](../../experiments/colliderml_target_alternatives/tradeoffs.png).

## Energy and momentum are different constraints

Let original truth four-vectors be `P_i`, and define target outputs by

`T_j = Σ_i A_ji P_i`, with `A_ji ≥ 0` and `Σ_j A_ji = 1` for every retained source.

Then `Σ_j T_j = Σ_i P_i`. This holds for hard assignments, merges, and fractional
assignments. It conserves total energy, all three momentum components, and MET.
It does **not** guarantee the same jets, scalar `H_T=Σ|p_T|`, particle counts,
classes or masses. Scalar energy conservation alone is weaker still: assigning
the correct energy along a wrong reconstructed-object direction changes MET.

An extreme example makes the capacity issue explicit. Two opposite 100 GeV
particles assigned to one cluster have total `(E,px,py,pz)=(200,0,0,0)`.
One exact composite preserves event energy and MET but removes both jets.
Assigning 200 GeV along either particle's direction instead preserves scalar
energy but invents 200 GeV of transverse imbalance. One output cannot represent
both directions. A second output, cluster split, or another compatible object is
required; energy rescaling cannot solve the missing angular degree of freedom.

## Quantitative comparisons

Jets use anti-kt R=0.4, truth pT≥20 GeV, |η|<2.5, target seed pT≥3 GeV, and
one-to-one matching ΔR<0.1. Width is `(q84−q16)/2`. Independent matches are used
below; common and paired comparisons are also cached. Multiplicity is the mean
event-wise ratio to the current number of targets, **not** the ratio to available
input objects or the configured model query budget.

| PU200 target definition | Match efficiency | Median jet pT response | Jet σ68 | Median total-E response | MET vector RMS | Target-count ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current merges | 49.09% | 0.94557 | 0.07885 | 0.95719 | 11.808 GeV | 1.000 |
| Separate represented particles | 81.95% | 0.98674 | 0.03849 | 0.95719 | 11.808 GeV | 1.231 |
| Split groups to pairwise ΔR≤0.3 | 75.14% | 0.98374 | 0.04757 | 0.95719 | 11.808 GeV | 1.118 |
| Split groups to pairwise ΔR≤0.1 | 78.91% | 0.98529 | 0.04259 | 0.95719 | 11.808 GeV | 1.155 |
| Split at truth-jet boundaries, oracle | 77.76% | 0.98514 | 0.04260 | 0.95719 | 11.808 GeV | 1.120 |
| Local 0.1 split + weak residual cells | 87.99% | 1.00000 | 0.01331 | 1 | ≈0* | 1.210 |
| All truth in 0.1×≈0.1 angular cells | 38.85% | 0.99783 | 0.09448 | 1 | ≈0* | 0.366 |
| All accepted truth particles, identity | 100% | 1 | 0 | 1 | 0* | 1.294 |

`*` Full-truth closure is imposed in the target definition using generator
information. It is not a claim that a learned model reconstructs perfect MET.

For PU0, current σ68=0.001158, matching=99.97%, median total-E response=0.99494,
and MET vector RMS=3.268 GeV. Local 0.1 splitting reaches 100% matching and zero
central σ68 with 3.96% more targets, preserving the same energy/MET. A zero
central width still allows rare tails. Its paired mean absolute jet-response
error improves by 0.001917, 95% event-bootstrap interval
[0.001730, 0.002125].

### What the controls actually do

Local splits operate **inside each existing allocator group**. Starting from
the most energetic source, they add particles to a subgroup only if their
momentum ΔR from **every** subgroup member is below the cut. Otherwise they
create another subgroup. Thus a chain of nearby particles cannot bridge widely
separated endpoints. Original particles are never discarded; every subgroup
receives the exact sum of its constituents. These subgroups may share the same
reconstructed input and therefore require more output slots or splitting inputs.
Truth momenta are used during label construction; no deployment algorithm or
input-compatible regrouping was tested here.

The truth-jet split separates each original group by its constituents' original
truth jet (stored jets pT≥3 GeV), keeping unclustered soft particles separately.
It is a diagnostic of jet-compatible granularity, not a proven optimum or a
detector-based algorithm. The similar performance of local 0.1 splitting is
encouraging without tying labels explicitly to a chosen truth-jet algorithm.

The local-plus-residual control adds all accepted truth particles excluded by
visibility/allocation into η–φ cells of width 0.1 (φ width is adjusted to tile the
circle). Each cell stores its **full four-vector sum**, including any composite
mass and its own momentum direction, rather than a massless energy at a fixed
cell center. It adds about 640 residual outputs per PU200 event beyond the local
split: mean counts are 11,558 current, 13,363 local split, and 14,004 with residuals.
It is a resolved-particle plus soft-flow representation, not exclusively a set
of physical particle identities.

### The improvement survives fixing the matched population

On 30,437 central PU200 truth jets matched by both current and local 0.1 targets,
width falls from 0.07223 to 0.03781. The paired change in mean absolute response
error is **−0.04825**, with 95% event-bootstrap interval [−0.04975, −0.04680].
For local 0.1 plus residuals, 31,831 common jets give widths 0.07492→0.00893
and mean-absolute-error change −0.08137 [−0.08289, −0.07989].

The reference-pT-weighted absolute error, assigning a full-pT penalty to unmatched
truth jets, falls from 0.4800 for current to 0.1964 for local 0.1, 0.1707 for
separate represented particles and 0.1060 for local-plus-residual. This diagnostic
does not additionally penalize unmatched candidate jets; a final optimization
objective should include both missing and extra jets.

### The global deficit is upstream of merging

All five represented-particle partitions have identical event four-vector sums.
In PU200 they retain a median 95.72% of reference total energy, with mean missing
energy **1,325 GeV/event**, MET vector residual RMS 11.808 GeV, and longitudinal
momentum residual RMS 139.35 GeV. Changing merges cannot recover these missing
particles. PU0 has a smaller median energy deficit of 0.51%.

Scalar H_T is not invariant: the PU200 median H_T response improves from
0.94635 for current merges to 0.97004 for local 0.1 and 0.97005 for separate
represented particles. The improvement comes from avoiding cancellation inside
large-angle composites. Local-plus-residual reaches 0.999991.

Conversely, the all-truth 0.1 flow grid has exact event closure but **worse** jet
width and matching than current merging in PU200. On 17,720 common jets its mean
absolute response error is worse by +0.00323 [0.00079, 0.00546]. Coarse angular
aggregation can alter anti-kt catchment boundaries in dense pileup even when
every cell is small relative to R=0.4. Perfect global conservation and a median
near 1 are insufficient validation of a target definition.

## Alternatives for assigning particles to reconstructed objects

### 1. Keep a canonical particle set and separate its associations

Store independent unmerged truth targets and a sparse many-to-many matrix of
track/cluster/hit associations. An input cluster can support multiple target
particles, and a particle can use multiple inputs. Use query/set outputs or
multiple neutral slots per cluster. Tracks with reliable majority ownership can
remain unique charged anchors.

This removes the need to turn a collision of labels on one input row into a
collision of four-vectors. It is the cleanest option when particle identities
matter. The represented-atom control quantifies the attainable label fidelity
before restoring weak particles: about 23.1% more PU200 targets than current.

The repository already has a set decoder, but `PFDataset.py` currently builds
`ytarget_set` from the nonzero exclusive rows of the existing `ytarget` array.
Those rows are already merged. Switching to set output alone does not undo
merging. Data must contain a separate canonical target table; query capacity
must be adequate, and full-event PU200 may require local/bucketed queries.

For a hits-based model, a smaller alternative is to assign unmerged particles
directly to distinct supporting hits instead of inheriting track/cluster merges.
There may be enough hit rows even when there are too few cluster rows. This needs
a unique-hit matching/capacity audit; the cache comparison does not establish
that every source can obtain a distinct hit. Tracker-hit anchors with zero
input energy also need absolute momentum regression or an appropriate track
anchor rather than a log(target/input-energy) target.

### 2. Minimal change: local conservative merges with overflow

Keep existing matching, split incompatible merge groups with a bounded-diameter
rule, and emit each subgroup separately. A 0.1 threshold is a promising starting
point here: nearly halves jet width without affecting global four-vector
balance, with 15.5% more PU200 targets. A 0.3 cut costs 11.8% more outputs but
gives a somewhat broader response.

Before adding a slot, try rerouting a particle to an unused compatible input
that it actually contributed to. Improve matching costs using spatial
compatibility and deposit evidence. Current `majority_energy` already performs
maximum-weight bipartite cluster matching; replacing it with another greedy
energy ordering is not a new solution. Maximum-cardinality-first or geometrically
constrained rematching are distinct candidates, but were **not tested** by the
cached partition study because the needed input-object/link geometry was not
retained in these caches.

If all compatible objects are occupied, use an overflow slot or split the input.
Vetoing the merge and dropping the source loses energy and repeats the previous
MET problem. A reassignment to an unrelated object can preserve global energy
while generating an unpredictable label.

Input splitting can be performed by a truth-blind hit subclusterer, for example
using track extrapolations and neutral-shower seeds, followed by rebuilding truth
links and rerunning assignment. This preserves one output per reconstructed
primitive while increasing input granularity. The measured local target splits
show its potential benefit, but do not demonstrate that a particular detector
subclusterer can find those splits.

### 3. Fractional sharing / transport

For represented generator particles, use source-normalized weights `A_ji` as
above, restricted to detector-supported edges and local momentum compatibility.
Fractional routing can reduce a hard winner-takes-all assignment and distribute
shared-cluster information across output particles. Track-owned particles should
be protected against double counting their calorimeter deposits.

An established related formulation is HGPflow's many-to-many incidence matrix,
which partitions each cluster's deposited energy among output particles. Its
normalization conserves the **deposit budget** at that stage; generator energy
and four-momentum are a separate calibration/label constraint, particularly
after track substitution or learned energy corrections.
See [HGPflow, section 2.1](https://link.springer.com/article/10.1140/epjc/s10052-025-14443-z).

Fractional weights do not, by themselves, solve the one-output/two-directions
problem if all contributions are still summed into one output per cluster.
Multiple output directions or compatible alternative hosts remain necessary.

### 4. Resolved particles plus a soft component

Treat well-supported particles individually, and predict an additional local
energy-flow component for particles that fail the resolved visibility/ownership
cuts. The local-plus-residual control demonstrates a target representation that
improves jets **and** restores full accepted-truth energy/momentum, using 21.0%
more outputs than current—fewer than the separate represented-particle control.

Whether the residual is predictable from detector data remains untested. Weak
calorimeter contributions and unreconstructed low-pT particles carry uncertainty;
an exactly closed truth target is not an exactly predictable observation. Keep
the soft component explicit, train/calibrate it separately, and measure its
effect on jets, H_T, energy and MET on independent processes and pileup levels.
Do not call these aggregate cells charged hadrons or photons with artificial
particle identities.

### 5. Reweight fixed reconstructed directions

One can fit nonnegative object energies while holding directions fixed, using
jet and event constraints with a penalty for departing from detector evidence.
For massless fixed-direction objects, total energy and momentum are linear in
their energies. A constrained least-squares fit is simple, but exact feasibility
depends on the available directions and allowed weights; local/track constraints
can make it impossible. Global constraints alone leave many wrong jet solutions.

This is a possible fixed-capacity fallback, not the first recommendation. A
global energy rescale can align total energy or a jet median while preserving
the wrong topology and changing MET. A post-hoc scalar correction cannot restore
two separated objects represented by only one direction.

## A practical optimization formulation

First choose the truth scope and hold it fixed across all candidates. If the
reference includes particles below resolved visibility, either represent them
or explicitly accept a balance deficit. The acceptance convention must not be
changed as a way of lowering the objective.

For an allowed assignment/partition, use

`L = L_jets + λ_E (ΔΣE / E_scale)^2 + λ_T ||ΔΣp_T||^2 / pT_scale^2 + λ_z (ΔΣpz / pz_scale)^2 + λ_N N_outputs`,

with detector-support, track-ownership and angular-diameter constraints.
`L_jets` should include matched jet-vector errors and penalties for both unmatched
truth jets and extra target jets, rather than optimizing the median of matched
responses alone. Validate more than one jet radius if the target representation
is intended to be analysis-independent.

For a fixed set represented through exact four-vector sums, the event terms are
constant. Impose conservation as a **hard constraint** and optimize jet fidelity
versus output count; varying MET weights cannot improve a partition's already
fixed event sum. If energy-only direction projections, drops or calibration
changes are allowed, the event terms become active and must be measured.

A simple target-building strategy is to start from separate particles, allow
only local detector-compatible coalescences, and accept a coalescence when it
fits an output budget with a sufficiently small increase in reclustered jet
error. Keep weak residuals separately. This is an approximate constrained
partitioning algorithm, not a guarantee of a global optimum. A pure truth-jet
objective without locality/support constraints could create labels that cancel
selection errors but are hard to infer from detector inputs.

My preferred next implementation is **canonical unmerged targets plus sparse
input associations**, with **local conservative grouping and overflow** as the
smaller intermediate change. Add the soft component as a separate training
experiment to address the upstream balance deficit. If exactly one output per
existing object is a hard requirement, first audit unused compatible inputs and
matching capacity; the cached controls do not prove these splits fit that budget.

## Reproduction and validation

```bash
UV_CACHE_DIR=/tmp/particleflow-uv-cache MPLCONFIGDIR=/tmp/matplotlib-colliderml \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python3 -m scripts.studies.colliderml_target_alternatives --workers 8
UV_CACHE_DIR=/tmp/particleflow-uv-cache MPLCONFIGDIR=/tmp/matplotlib-colliderml \
  OPENBLAS_NUM_THREADS=1 uv run --no-sync python3 -m scripts.studies.plot_colliderml_target_alternatives
```

The input cache's extraction fingerprint is checked. All 2,000 event IDs are
unique within sample, every scheme covers the same events, and every target
partition asserts four-vector closure. The baseline is the current exact-group
control, consistently reclustered in double precision for all alternatives;
the previous audit separately verified that serialization effects are negligible.
Three tests cover pairwise rather than transitive angular cuts, φ wrapping,
exclusion of dropped sources and full-four-vector cell aggregation.

Artifacts under `experiments/colliderml_target_alternatives/` include
`tradeoffs.csv`, `jet_summary.csv` (pT, |p| and energy responses),
`balance_summary.csv`, `paired_comparison.csv`, `reference_jet_error.csv`,
per-event matches/balances, source/dependency fingerprints, and code snapshots.
The conservation checks do not certify input-link feasibility, PID/charge
quality, model capacity or trained-model performance. No production target
definition or model was modified.
