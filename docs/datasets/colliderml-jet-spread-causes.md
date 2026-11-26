# Why ColliderML target jets differ from truth jets: PU0 and PU200

This is the pre-acceptance-update study. The [updated-branch jet/MET chain
audit](colliderml-updated-chain.md) reruns 1,000 events per sample after the
|η|>3.5 truth cut, including final output rows and a fixed-reference comparison.

The response spread comes from particle selection and changes to jet membership.
Target momenta are generator four-vectors or their exact sums: no detector momentum
smearing is added. PU0 is already close to a delta function. PU200 inclusive jets
are affected both by visibility cuts and by merging across truth-jet boundaries.
Production target-building code is unchanged.

Plots: [stage comparison](../../experiments/colliderml_jet_spread_causes/stage_responses.png),
[constituent-flow examples](../../experiments/colliderml_jet_spread_causes/example_jet_flows.png),
[granularity controls](../../experiments/colliderml_jet_spread_causes/granularity_controls.png).

## Samples and definitions

We processed 1,000 distinct ttbar PU0 events (shard 0, event IDs 0–999) and
1,100 distinct ttbar PU200 events (shards 0–10, IDs 0–1099), with eight CPU
processes outside the sandbox. Both samples use ColliderML revision
`64c3d2f112df3d5d20979d22da7cfdff13e10c4b`.
The four PU0 source-file SHA256s were checked against the pinned Hugging Face
LFS metadata; the previous PU200 download manifest is retained in
`experiments/colliderml_jet_spread_causes/source_provenance.json`.

PU0 uses production CLUE `pu0`; PU200 uses production CLUE `pu200` and cached
current target jets from the [previous response study](colliderml-merge-jet-response.md).
These are independent event samples, with different pileup, jet spectra and
clustering presets. Within-sample ablations isolate processing effects; the
between-sample comparison does not isolate pileup density from the preset.

Jets use anti-kt R=0.4 and E-scheme recombination. The main selection is truth
pT ≥20 GeV, |η| <2.5, angular matching ΔR <0.1, and target jets pT ≥3 GeV.
The reference is the converter's **measurable pre-visibility truth** (excluding
neutrinos and including its surrogate ancestors), rather than an unrestricted
collection of all stable generator particles. Inclusive targets are compared to
inclusive truth, including pileup. The output also covers forward jets, pT ≥50
GeV, matching ΔR <0.2, and hard-scatter views.

Four stages use the same truth reference:

1. **Truth identity:** recluster exactly the reference particles.
2. **Visible atoms:** apply the existing visibility cuts; keep particles separate.
3. **Represented atoms:** additionally apply actual allocator drops, but retain
   every surviving source separately, including sources that current processing merges.
4. **Current:** construct the current merged targets.

Represented atoms differ from the previous "never merge" rule, which discarded
unassigned merge sources. This control retains their momentum and may require
more target slots than the current one-target-per-input representation permits.

## How much spread appears at each stage?

Response is pT(stage jet)/pT(truth jet); σ68=(q84−q16)/2. These values use
each stage's own matched jets; matching efficiencies include all references.

| Stage | PU0 median | PU0 σ68 | PU0 match efficiency | PU200 median | PU200 σ68 | PU200 match efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Truth identity | 1 | 0 | 100% | 1 | 0 | 100% |
| Visible atoms | 1 | 0 | 100% | 0.98654 | 0.03857 | 81.82% |
| Represented atoms | 1 | 0 | 100% | 0.98651 | 0.03868 | 81.76% |
| Current merged targets | 1 | 0.00119 | 99.97% | 0.94433 | 0.07815 | 49.95% |

There are 5,918 PU0 and 75,706 PU200 reference jets. A zero central width does
not imply no tails: 92.26% of PU0 represented-atom responses are within 0.01%
of unity, versus 67.60% after current merging. Only 0.68% of matched PU0 current
jets differ by more than 5%; the corresponding PU200 fraction is 63.80%.

To avoid changing the matched population, compare the 5,916 PU0 and 34,446 PU200
truth jets matched at **all four stages**. The following ordered increments are
changes in mean absolute response error, with 95% event-bootstrap intervals:

| Processing step | PU0 increment | PU200 increment |
| --- | ---: | ---: |
| Visibility | 0.000881 [0.000722, 0.001059] | 0.045490 [0.044767, 0.046204] |
| Allocator drops | 0.000032 [0, 0.000097] | 0.000113 [0.000033, 0.000188] |
| Merging | 0.002132 [0.001900, 0.002405] | 0.052511 [0.051205, 0.053878] |

These are ordered error increments, not independent variance components. On
the common PU200 cohort, visibility/represented/current σ68 values are
0.03621/0.03637/0.07260. Thus the result persists after controlling matching
selection. Original allocator drops are a very small effect here.

## Visibility and pileup

Keeping original truth-jet membership fixed, rather than reclustering the
remaining atoms, gives PU200 visible response σ68=0.02924 and mean response
0.97199. Selection alone removes about 2.8% of the projected transverse momentum
on average. Reclustering changes catchment boundaries and further broadens it.

Of particles removed by visibility within fiducial truth jets (pT ≥20 GeV,
|η| <4.5), **99.05% are pileup**. Charged hadrons account for **79.83% of the
lost projected jet pT**. Lost charged hadrons average 0.320 GeV scalar pT.
This explains why inclusive PU200 jets respond differently from PU0, even before
merging. Identity remains an exact delta in both samples: pileup by itself does
not create a spread when clustering identical constituent sets.

An oracle projection using only each target's original hard-scatter contribution
is almost a delta even in PU200: current median=1, σ68=0.000158, mean absolute
error=0.001437, and 6,613/6,614 reference jets matched. This is a truth-based
diagnostic, not a result achievable by a learned pileup-removal algorithm.

## Merging redistributes momentum between jets

Exact four-vector summation preserves event momentum, but not its partition
among jets. Merging spatially separated sources replaces their individual
directions with a composite direction. Cross-boundary groups carry momentum
into another jet; even within-jet premerges can change anti-kt's clustering
sequence. Composite masses also change rapidity, which enters the jet distance.

Detailed lineage is available for all 1,000 PU0 events and the first **100 PU200
events**. Among merged groups, **5.50% in PU0 versus 46.24% in PU200** contain
material contributions from multiple truth jets. Material means at least two
stored truth jets (pT ≥3 GeV) each contribute ≥10% of the group's scalar
constituent pT. Softer constituents enter the denominator but do not define
additional stored truth jets.

For every detailed matched jet we verify the exact identity

`Ptarget − Ptruth = −Pvisibility −Pallocator −Poutflow +Pinflow`.

Outflow consists of represented original truth-jet particles ending outside its
matched target jet; inflow consists of represented particles originally outside
that truth jet. Projecting transverse vectors onto the truth-jet direction gives

`response − 1 = −visibility_parallel −allocator_parallel −outflow_parallel +inflow_parallel +axis_term`,

where `axis_term = (|pTtarget| − pTtarget·truth_unit)/pTtruth`.
On the common detailed PU200 cohort (3,119 jets), mean contributions are:

| Term | Mean normalized contribution |
| --- | ---: |
| Visibility | −0.028396 |
| Allocator loss | −0.000114 |
| Outflow | −0.111888 |
| Inflow | +0.084973 |
| Axis term | +0.000446 |
| Resulting mean response | 0.945021 |

Migration has substantial cancelling inflow and outflow, with residual
event-by-event fluctuations. The geometric match agrees with the target jet
having maximum shared original scalar pT in **99.23%** of these current matches;
the broadening is principally changed contents, rather than a wrong angular
partner. This statistic applies to matched jets, not unmatched references.

Two examples, both with the best shared-constituent match:

| PU200 event / truth jet | Truth pT | Target pT | Visibility loss | Outflow | Inflow |
| --- | ---: | ---: | ---: | ---: | ---: |
| 74 / 28 | 77.29 GeV | 20.83 GeV | 0.46% | 72.68% | 0% |
| 40 / 70 | 50.25 GeV | 78.66 GeV | 0.17% | 11.51% | 68.05% |

Loss/outflow/inflow are transverse projections normalized by original truth pT;
the plotted axis term completes the exact response accounting.

## Controls on target granularity

On the 100 detailed PU200 events (6,883 references), splitting current groups
by original truth-jet origin recovers most of the atomic-particle agreement:

| Representation | Match efficiency | Median | σ68 | Mean absolute error |
| --- | ---: | ---: | ---: | ---: |
| Represented atoms | 82.71% | 0.98697 | 0.03762 | 0.04885 |
| Current | 49.47% | 0.94475 | 0.07724 | 0.10564 |
| Current, composite mass removed | 63.40% | 0.94760 | 0.06749 | 0.09379 |
| Current, split by truth jet | 79.37% | 0.98455 | 0.04345 | 0.05405 |

The split control conserves the total four-vector exactly and preserves MET.
It uses truth knowledge and adds targets, so it is an oracle diagnostic rather
than a deployable rule under the current output capacity. Particles outside
stored truth jets are kept separately. Remaining spread reflects visibility
and residual changes in jet clustering from allowed premerges.

The mass control resets only composite energies to |p|, retaining their
three-momenta. It changes total energy and is **not a proposed correction**.
Its partial improvement shows that mass/rapidity contributes, but cannot
explain all the merging effect. In PU0 it slightly worsens σ68.

The practical implication is to preserve source momentum while restricting
merges that move appreciable momentum across jet boundaries. Simply vetoing
sources loses momentum and worsens MET, as the [MET study](colliderml-merge-met.md)
shows. A target representation allowing multiple particles per input or a
momentum-conserving split is better motivated by these controls.

## Reproduction and validation

Extraction: `scripts/studies/colliderml_jet_spread_causes.py`.
Aggregation/plots: `scripts/studies/plot_colliderml_jet_spread_causes.py`.
Tests: `tests/test_colliderml_jet_spread_causes.py` (four passed).

```bash
UV_CACHE_DIR=/tmp/particleflow-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python3 -m scripts.studies.colliderml_jet_spread_causes --workers 8
UV_CACHE_DIR=/tmp/particleflow-uv-cache MPLCONFIGDIR=/tmp/matplotlib-colliderml \
  uv run --no-sync python3 -m scripts.studies.plot_colliderml_jet_spread_causes
```

PU200 uses the preceding response and lineage caches. Its visibility mapping,
represented energy, drops, MET and particle counts are checked against previous
accounting; detailed recomputed jets agree with cached jets. PU0 reruns the
actual allocator. There were no zero-pT truth atoms or visibility-map fallbacks.
Both four-vector flow closure and the scalar response identity are asserted,
as is momentum conservation in the oracle split.

Results and reusable particle/jet caches are under
`experiments/colliderml_jet_spread_causes/`; `stage_summary.csv`,
`paired_stage_changes.csv`, `flow_summary_common.csv`,
`visibility_losses_by_species.csv`, `merged_group_summary.csv`, and
`oracle_control_summary.csv` provide the tabulated results. The response plot
uses common matched cohorts; the main table above uses independent matches.
`flow_summary_common.csv` controls the detailed cohort; the independent flow
summary has different event coverage between stages and should not be used
for paired stage attribution.

An empty-view serialization failure required normalizing output index/bool
dtypes. Eighteen completed jobs use the original extraction hash and one uses
the corrected hash; `inputs.json` explicitly records this physics-equivalent
reuse, and each job retains its actual hash. Both executed script revisions
(`extraction_initial.py`, `extraction_repaired.py`) and `validation.json`
preserve provenance; the working scripts were subsequently formatted.
Production code was not changed.
