# ColliderML jet and MET response after the acceptance update

This reruns the postprocessing-chain study at commit `f490a6e8`, on **1,000 PU0
and 1,000 PU200 ttbar events**. All current truth construction, CLUE clustering,
allocation, merging, and final target-row construction are recomputed through
the actual production converter. Production code is unchanged.

The relevant main update is `7c8e8889`: particles with momentum |η|>3.5 are
removed from target candidates **and the generator reference**, including
out-of-acceptance surrogate ancestors. Their deposits remain in the detector
inputs but are not attributed to those removed particles. This changes both
the target construction and the denominator of a target/truth comparison.

Results directory: `experiments/colliderml_updated_chain_f490a6e8/`.

Plots:

- [Jet response at each step, independent matches](../../experiments/colliderml_updated_chain_f490a6e8/jet_stages_independent.png)
- [Jet response on common matched truth jets](../../experiments/colliderml_updated_chain_f490a6e8/jet_stages_common.png)
- [Response and matching versus truth jet pT](../../experiments/colliderml_updated_chain_f490a6e8/jet_response_vs_pt.png)
- [Response and matching versus η, including the previous branch](../../experiments/colliderml_updated_chain_f490a6e8/jet_response_vs_eta.png)
- [Inclusive target MET versus inclusive truth MET](../../experiments/colliderml_updated_chain_f490a6e8/met_inclusive.png)
- [Oracle hard-scatter MET](../../experiments/colliderml_updated_chain_f490a6e8/met_oracle_hardscatter.png)
- [Inclusive targets versus hard-scatter MET reference](../../experiments/colliderml_updated_chain_f490a6e8/met_inclusive_vs_hardscatter.png)
- [Acceptance update with the previous reference held fixed](../../experiments/colliderml_updated_chain_f490a6e8/acceptance_and_previous_branch.png)
- [Jet constituent-flow breakdown](../../experiments/colliderml_updated_chain_f490a6e8/jet_flow_breakdown.png)
- [Individual low/high response examples](../../experiments/colliderml_updated_chain_f490a6e8/example_jet_flows.png)

## Definitions and controls

The main truth reference is the updated converter's **accepted measurable
truth**, before visibility and allocation: particles with any attributed
calorimeter deposit or reconstructed-track hit share, minus neutrinos, with
the leaf/pi0-decay/surrogate rules in `truth.py`. It is not an unrestricted
collection of all stable generator particles. Inclusive truth contains pileup.
Hard-scatter-only references select `ispu==0`.

The same source revision as the previous study is used:
`CERN/ColliderML-Release-1@64c3d2f112df3d5d20979d22da7cfdff13e10c4b`.
PU0 uses source shard 0 (1,000 events); PU200 uses shards 0–9 (100 each).
Event IDs are 0–999 in each sample. Production CLUE `pu0` and `pu200` presets
are used respectively, with `majority_energy` assignment. Thus comparisons
between PU0 and PU200 include different pileup, spectra, and clustering presets;
within-sample stage comparisons isolate the processing effects.

| Stage | Particle collection |
| --- | --- |
| Truth | Accepted measurable truth, clustered identically to itself |
| Visible | Existing visibility cuts applied; particles remain separate |
| Represented | Actual allocator drops applied; all surviving sources remain separate |
| Exact merged | Sources grouped by the actual allocator's merge lineage, with exact four-vector sums |
| Final clustered targets | Actual exclusive track/cluster target rows and the converter's stored target jets |
| Final hits-view targets | Actual exclusive tracker/calo hit target rows, reclustered with the production jet recipe |

Visibility requires ownership of at least 20% of a reconstructed track's hits,
or calorimeter deposit/particle energy >10%, or (charged particles only) deposit
>0.5 GeV. The later allocator requires >50% track ownership to host a track;
particles with no usable track/cluster host can be dropped. Represented atoms
retain would-be merge sources; they differ from the earlier "never merge"
control, which discarded those sources, and can require additional target slots.

Jets use anti-kt R=0.4, E-scheme, and target pT≥3 GeV. Primary reference jets
have pT≥20 GeV, |η|<2.5 and one-to-one matching ΔR<0.1. Tables also cover
pT≥50/100/200 GeV, forward jets, matching ΔR<0.2, full |p| and energy response.
Response=pT(target)/pT(truth); σ68=(q84−q16)/2. Independent matches retain an
explicit unmatched row per reference; common matches require the same truth jet
to be matched at all six stages.

MET is the magnitude of the negative vector sum of **all** particles at a stage;
the jet cuts are not applied. Raw magnitude/residual distributions and vector
RMS use every event. MET response ratios use truth MET≥20 GeV to avoid unstable
division by a small reference. Vector RMS means sqrt(mean(ΔMETx²+ΔMETy²)).

Three views are kept distinct: inclusive/inclusive; an oracle projection of
each target's original hard-scatter contribution versus hard-scatter truth; and
inclusive targets versus hard-scatter truth. The oracle uses exact constituent
knowledge and is not a deployable pileup-removal result. Its final stages
describe grouping and hit retention, without assigning float32 rounding to
individual truth origins inside a mixed target.

A second, **frozen pre-acceptance reference** comes from the previous study's
generator-atom cache. Fresh raw-event calculations with only the acceptance
switch disabled reproduce its features exactly in four regression events.
Comparisons against this reference separate changed targets from a changed
truth denominator. Current target jets and lineage are never taken from an old
cache. Previous-branch jet curves use only the same 1,000 event IDs per sample.

## Quantitative results

| Stage | PU0 median | PU0 σ68 | PU0 match efficiency | PU200 median | PU200 σ68 | PU200 match efficiency |
| --- | --- | --- | --- | --- | --- | --- |
| Accepted truth | 1.00000 | 0.00000 | 100.00% | 1.00000 | 0.00000 | 100.00% |
| After visibility | 1.00000 | 0.00000 | 100.00% | 0.98679 | 0.03838 | 82.03% |
| Represented, separate | 1.00000 | 0.00000 | 100.00% | 0.98674 | 0.03849 | 81.95% |
| Exact merged groups | 1.00000 | 0.00116 | 99.97% | 0.94557 | 0.07885 | 49.09% |
| Final clustered targets | 1.00000 | 0.00116 | 99.97% | 0.94557 | 0.07885 | 49.09% |
| Final hits-view targets | 1.00000 | 0.00116 | 99.97% | 0.94557 | 0.07885 | 49.09% |

These are independent matches of central inclusive truth jets. The following MET results use all 1,000 events per sample; median ratios require truth MET≥20 GeV.

| Stage | PU0 median MET ratio | PU0 magnitude bias (GeV) | PU0 vector RMS (GeV) | PU200 median MET ratio | PU200 magnitude bias (GeV) | PU200 vector RMS (GeV) |
| --- | --- | --- | --- | --- | --- | --- |
| Accepted truth | 1.00000 | 0.000 | 0.000 | 1.00000 | 0.000 | 0.000 |
| After visibility | 1.00000 | 0.305 | 3.268 | 0.99953 | 0.140 | 11.804 |
| Represented, separate | 1.00000 | 0.305 | 3.268 | 0.99981 | 0.157 | 11.808 |
| Exact merged groups | 1.00000 | 0.305 | 3.268 | 0.99981 | 0.157 | 11.808 |
| Final clustered targets | 1.00000 | 0.305 | 3.268 | 0.99981 | 0.157 | 11.808 |
| Final hits-view targets | 1.00000 | 0.305 | 3.268 | 0.99981 | 0.157 | 11.808 |

### The jet bias persists after the acceptance update

PU0 remains close to a delta function: the final median is 1 and σ68=0.00116.
Its zero visible-stage σ68 reflects a point mass at unity, with rare tails:
mean absolute visible response error is 0.000882 rather than zero.
PU200 has two large changes: visibility shifts the median to 0.98679 and
broadens σ68 to 0.03838; merging shifts the final median to 0.94557 and broadens
σ68 to 0.07885. Allocator drops and output writing add very little. Only 49.09%
of central PU200 reference jets have a final angular match, versus 99.97% in PU0;
the response distribution must be read alongside that efficiency.

This is not just a change in which jets match. On the **30,946 PU200 truth jets
matched at every stage**, visible/represented/final medians are
0.98290/0.98281/0.94698 and widths are 0.03623/0.03640/0.07348.
Ordered increments in mean absolute response error, with 95% paired
event-bootstrap intervals, are:

| Step | PU200 increase in mean absolute response error |
| --- | ---: |
| Visibility | 0.04525 [0.04448, 0.04602] |
| Allocator drops | 0.000104 [0.000037, 0.000169] |
| Merging | 0.05337 [0.05190, 0.05493] |
| Exact groups to final output | 0.000000013 [−0.0000068, 0.0000078] |
| Clustered to hits-view targets | 0 |

With original truth-jet membership held fixed, PU200 visible response has
mean 0.97206 and σ68=0.02923. Visibility therefore removes about 2.8% of jet
pT on average before allowing jet boundaries to change. Of particles lost
to visibility inside central truth jets, **99.01% are pileup**; charged hadrons
carry **81.05% of the lost projected pT**. This is a set-selection bias relative
to the measurable pre-visibility reference.

For the common PU200 cohort, exact constituent flow gives the following
**mean** response decomposition:

| Contribution | Signed response change |
| --- | ---: |
| Visibility | −0.028932 |
| Allocator drops | −0.000109 |
| Outflow from the original truth jet | −0.113439 |
| Inflow from outside that truth jet | +0.085038 |
| Transverse-axis term | +0.000438 |
| Mean final exact-group response | 0.942996 |

Thus visibility and net migration each lower mean response by roughly 2.9
percentage points. The much larger, partially cancelling inflow/outflow
also fluctuates between jets and broadens both tails. These means should
not be interpreted as an additive decomposition of the median or σ68.
Geometric matching agrees with the target jet having maximum shared original
scalar pT in **99.03%** of common exact-group matches, so selecting the wrong
angular partner is not the main explanation for their spread.

Across all events, 45.67% of PU200 merged groups versus 5.56% in PU0 have material
contributions from multiple truth jets. Material means at least two stored truth
jets (pT≥3 GeV), each carrying ≥10% of the group's scalar constituent pT; soft
particles outside stored jets enter the denominator but do not define extra jets.
There are 1,214,234 PU200 and 10,171 PU0 merged groups in this comparison.

Examples from the updated branch, using exact merged groups and best-overlap
matches, show that the effect remains after the forward cut:

| PU200 event / truth jet | Truth pT | Target pT | Visibility loss | Outflow | Inflow |
| --- | ---: | ---: | ---: | ---: | ---: |
| 478 / 61 | 65.03 GeV | 17.32 GeV | 2.12% | 71.24% | 0% |
| 907 / 69 | 51.95 GeV | 99.00 GeV | 0.55% | 15.91% | 106.65% |

The percentages are projected transverse contributions normalized by truth pT.
The small axis term completes exact closure. The low-response example belongs
to the independent matched cohort, not the all-stage common cohort.

### Much of the broad inclusive distribution concerns pileup

Of the 68,957 central PU200 truth jets, 56,602 have at most 5% hard-scatter scalar
constituent pT. Their final median response is 0.93743 and σ68=0.09045.
Jets with more than 50% hard-scatter content have median 0.97622 and σ68=0.02274
(1,600/1,630 matched). These content selections are truth diagnostics.

The PU200 oracle hard-scatter projection is almost a delta: median 1,
σ68=0.000146, and 5,985/5,986 jets matched. This isolates the original
hard-scatter contribution inside targets; it does not demonstrate practical
pileup mitigation. The actual inclusive response also improves with truth pT:

| Minimum central truth pT | PU200 final median | σ68 | Matched / references |
| --- | ---: | ---: | ---: |
| 20 GeV | 0.94557 | 0.07885 | 33,851 / 68,957 |
| 50 GeV | 0.94639 | 0.06322 | 27,839 / 51,070 |
| 100 GeV | 0.95897 | 0.04130 | 8,975 / 11,735 |
| 200 GeV | 0.97921 | 0.01823 | 784 / 791 |

### MET smearing is already present before merging

The PU200 median MET ratio is **0.99981**, rather than the jet median 0.94557.
Mean magnitude bias is only +0.157 GeV, while vector RMS is 11.808 GeV.
Removing momentum does not necessarily reduce the magnitude of a vector sum:
it can increase or decrease cancellations in the event. A nearly unbiased
magnitude therefore does not imply an accurate MET vector.

Visibility alone gives vector RMS 11.804 GeV; allocator drops change this to
11.808 GeV, and exact merging leaves it unchanged. For PU0 the corresponding
RMS is 3.268 GeV. On the accepted-reference PU200 comparison, vector MSE is
139.432 GeV²: visibility contributes 139.326 GeV², allocator changes 0.221 GeV²,
and cross terms contribute −0.115 GeV². They cannot be added as independent RMS
contributions. Oracle hard-scatter-only PU200 MET has vector RMS 3.481 GeV.

No targets are lost from the final hits-view representation in these 2,000
events. The maximum event MET-vector shift from exact groups to serialized
target rows is 0.000328 GeV, and exact merge closure is better than
1.8×10⁻¹² GeV. Independent double reclustering changes 49/92,458 checked PU200
jets by >0.01% in pT or fails ΔR<0.001 matching (18 central jets); PU0 has zero
such changes among 6,502 checked jets. These effects are far too small to explain
the ensemble broadening.

### Comparing with the previous branch requires a fixed reference

The following central PU200 jet results use the same 1,000 events. Independent
matches are shown; reference populations and matching rates are explicit.

| Targets / reference | References | Match efficiency | Median | σ68 |
| --- | ---: | ---: | ---: | ---: |
| Previous / previous | 68,793 | 50.05% | 0.94423 | 0.07781 |
| Updated / previous | 68,793 | 49.30% | 0.94548 | 0.07876 |
| Updated / updated | 68,957 | 49.09% | 0.94557 | 0.07885 |

Holding both the previous reference **and matched population** fixed gives
29,546 common central jets. Their median changes from 0.94370 to 0.94506 and
σ68 from 0.06846 to 0.06921. The change in mean absolute error is
−0.00053, 95% interval [−0.00115, +0.00010]. This does not show a substantial
resolution improvement in central jets.

Forward jets remain problematic. For updated truth jets with 3≤|η|<3.5, the
independent final median is 0.67252, σ68=0.28949, and matching efficiency=30.10%.
On the 1,968 jets matched before and after the update **against the same previous
reference**, medians change from 0.76475 to 0.73318 and widths from 0.27921 to
0.27231. Mean absolute error changes by +0.00644 with interval
[−0.00019, +0.01256]. The matching population changes substantially, so an
independent-median shift alone should not be read as a paired effect.

The MET reference change is much larger:

| Sample | Previous targets / previous truth vector RMS | Updated targets / previous truth vector RMS | Updated targets / updated truth vector RMS |
| --- | ---: | ---: | ---: |
| PU0 | 3.709 GeV | 4.488 GeV | 3.268 GeV |
| PU200 | 16.732 GeV | 22.533 GeV | 11.808 GeV |

The accepted truth MET vector itself differs from the previous reference by
19.639 GeV RMS in PU200. Consequently, the smaller updated-target/updated-truth
residual is conditional on the new acceptance definition. It does **not** mean
that targets better preserve the unchanged pre-cut generator-event momentum:
against that fixed old reference the residual increases.

The broad jet response remains principally a combination of visibility selection
and momentum migration between jets. The acceptance cut changes the reference
and removes forward candidates; it does not solve the granularity problem.
Preserving event momentum is necessary for MET fidelity, but is insufficient
for jet fidelity when spatially separated particles are merged.



## Why jets can smear when MET does not

Targets use generator four-vectors or their sums, without a stochastic detector
momentum-resolution term. Detector evidence affects admission, ownership, and
grouping. A changed particle set can therefore lose momentum; a changed grouping
can preserve total momentum while redistributing it between jets.

For every matched exact-group jet, constituent lineage verifies

`Ptarget−Ptruth = −Pvisibility−Pallocator−Poutflow+Pinflow`.

Outflow consists of represented original constituents of the truth jet that end
up outside its matched target jet. Inflow consists of represented particles
originally outside that truth jet. Transverse projection onto the truth-jet axis
gives the exact scalar identity

`response−1 = −visibility_parallel−allocator_parallel−outflow_parallel+inflow_parallel+axis_term`.

The axis term converts the projected target momentum to its magnitude. These
terms add to each jet's response change and to its **mean**, but their medians
and widths are not additive. Paired event-bootstrap intervals report ordered
changes in mean absolute response error, rather than a variance decomposition.
`jet_flow_summary.csv` includes both common/independent cohorts and low/high tails.

By contrast, the sum of exact merged four-vectors equals the sum of represented
atoms. MET is unchanged by this grouping. `met_mse_components.csv` separates
acceptance, visibility, allocator, serialization and hits-view vector changes,
retaining their cross terms instead of adding RMS values in quadrature.

## Precision and output representation

The reference/constituent accounting uses exact double-precision Cartesian
vectors reconstructed from truth-table features. The final clustered target-jet
distribution uses the converter's **actual stored `targetjet`**, not an
independent approximation. In every event, the audit reproduces those stored
jets exactly using the original float32 target arithmetic and input order.

An initial validation assumption that double-precision reclustering must agree
jet by jet was false: small numerical perturbations can occasionally change
discrete anti-kt boundaries. The audit separately counts final production jets
that differ from double reclustering by >0.01% in pT or fail matching ΔR<0.001.
The exact-group versus final stage comparison includes these numerical effects.
Their event-level MET effect is measured directly from target rows.

## Reproduction and artifacts

```bash
UV_CACHE_DIR=/tmp/particleflow-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python3 -m scripts.studies.colliderml_updated_chain --workers 8
UV_CACHE_DIR=/tmp/particleflow-uv-cache MPLCONFIGDIR=/tmp/matplotlib-colliderml \
  uv run --no-sync python3 -m scripts.studies.plot_colliderml_updated_chain
```

The extraction calls `_event_record_one` once per event, observing its truth and
allocator results. It reconstructs exact groups from the observed lineage and
checks momentum conservation, target counts, feature alignment, exclusive PN
uniqueness, production jet reproduction and MET agreement. Large hit arrays are
discarded after the diagnostics; reusable generator-atom/group and jet/MET tables
are cached per job. Final CSVs provide distributions' statistics and attribution.

`inputs.json`, `run_summary.json`, `source_provenance.json`,
`reference_regression_checks.json`, `validation.json` and the `code/` snapshot
record inputs, exact event coverage and dependency hashes. All final jobs use
the same extraction fingerprint. The last two PU200 shards are split into
25-event jobs to keep eight CPUs occupied; `schedule_remaining.py` records this
execution scheduling. This does not change the extraction function or physics.

Checks: 21 tests passed across the new audit, acceptance-cut tests, and production
postprocessing tests. Independent accounting verifies per-jet scalar closure and
event-level exact-merge MET conservation. Ruff and diff checks are also run.

Earlier context: [pre-update jet decomposition](colliderml-jet-spread-causes.md),
[merge-rule comparison](colliderml-merge-jet-response.md), and
[merge-rule MET comparison](colliderml-merge-met.md).
