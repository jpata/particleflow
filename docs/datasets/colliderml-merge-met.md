# ColliderML merge rules: event-level MET

These measurements precede the |η|>3.5 truth-acceptance update. The
[updated-branch jet/MET chain audit](colliderml-updated-chain.md) reruns the
current chain on 1,000 PU0 and 1,000 PU200 events and separates reference changes.

This extends the [1,100-event jet-response study](colliderml-merge-jet-response.md)
to event-level missing transverse momentum, using the same PU200 events,
CLUE PU200 clustering, exclusive assignment and three target definitions.
Production code is unchanged.

## Magnitude distributions on all 1,100 events

For inclusive targets versus pileup-inclusive measurable truth:

| Merge rule | Mean signed magnitude residual [GeV] | σ68 of magnitude residual [GeV] | Mean absolute magnitude residual [GeV] |
| --- | ---: | ---: | ---: |
| Current unconditional | +2.27 | 11.28 | 9.55 |
| Truth-momentum ΔR <0.3 | +2.39 | 13.66 | 11.19 |
| Never merge | +2.15 | 13.37 | 11.19 |

The truth MET median is 76.60 GeV. The target medians are 80.09, 79.04 and
78.75 GeV, respectively. The overall magnitude distributions overlap strongly,
while the residual distributions expose the degradation from rejecting merges.
The [magnitude distribution and residual tails](../../experiments/colliderml_merge_met/met_magnitudes_inclusive.png)
are computed directly from the exact event sums stored by the jet study.

The gain in jet response therefore comes with an event-level momentum cost.
Four-vector merging preserves the event momentum budget while redistributing
it among target objects. Dropping rejected sources changes the cancellation of
transverse momenta; individual events can move upward or downward in MET.

## Exact-vector results on all 1,100 events

All 1,100 events passed agreement checks against the jet study's stored MET
magnitudes, target energies, accepted merge counts and rejected-vector norms.
All shard-0 host/source identities also matched the earlier particle audit.
There were **zero production-allocator fallbacks**. Every event appears under
every rule and reference view, without an event-quality or matching selection.

### Inclusive target versus inclusive truth MET

| Merge rule | Mean vector-error norm [GeV] | RMS vector error [GeV] | 90th percentile vector error [GeV] | Events with vector error >20 GeV |
| --- | ---: | ---: | ---: | ---: |
| Current unconditional | 14.66 | 16.72 | 25.76 | 21.91% |
| Truth-momentum ΔR <0.3 | 17.76 | 20.25 | 30.67 | 37.64% |
| Never merge | 17.86 | 20.24 | 30.96 | 36.18% |

The gate increases the mean vector error by **3.10 GeV**, with paired
event-bootstrap 95% interval **[2.67, 3.53] GeV**. Never merging increases it by
**3.20 [2.78, 3.62] GeV**. The gate worsens this indicator in 729/1,100 events;
never merging worsens it in 738/1,100. Both degradation estimates are well resolved
statistically in this sample.

The corresponding paired increases in mean absolute magnitude error are
**1.64 [1.24, 2.07] GeV** for the gate and **1.65 [1.26, 2.05] GeV** for never
merging. Thus the effect appears in both magnitude and direction-sensitive
metrics. Correlations of target and truth magnitudes remain high
(0.9777 /0.9691 /0.9688), so comparing distribution shapes or correlations alone
would understate the degradation.

For events with truth MET ≥20 GeV, the median absolute MET direction error
increases from **0.0873 rad** to **0.1105 rad** and **0.1162 rad**. The signed
parallel mean response moves from 1.0103 to 1.0015 and 1.0019, showing that a
near-unity average scale does not establish good event-level resolution.

Plots: [magnitude and vector residual distributions](../../experiments/colliderml_merge_met/met_distributions.png),
[event-by-event target versus truth MET](../../experiments/colliderml_merge_met/met_target_vs_truth.png),
and [response and vector error versus truth MET](../../experiments/colliderml_merge_met/met_response_vs_truth.png).

### Where the extra error comes from

For residual `e = C - M_truth` and rejected transverse vector `R`, the exact
sample mean-square identity is

```text
mean(|M_gate - M_truth|²) = mean(|e|²) + mean(|R|²) + 2 mean(e dot R).
```

The inclusive gate result, in GeV², is

```text
410.236 = 279.649 + 129.458 + 1.128.
```

The cross term is small; almost all the increase comes from the new rejected
momentum variance. The rejected vector has RMS norm **11.38 GeV** and median
norm **9.18 GeV**. This is an effect of removing momentum, rather than an
error in the four-vector addition. Never merging has rejected-vector mean square
139.090 GeV² and cross term −9.225 GeV², giving a similar final MET error despite
rejecting more total energy.

### Hard-scatter reference and oracle pileup separation

Against hard-scatter-only truth, inclusive target MET contains a large pileup
contribution. The hard-scatter truth median is **36.45 GeV**, while inclusive
target medians are about 79–80 GeV:

| Merge rule | Inclusive targets / HS truth: magnitude MAE [GeV] | Inclusive targets / HS truth: vector RMS [GeV] | Oracle HS magnitude MAE [GeV] | Oracle HS vector RMS [GeV] |
| --- | ---: | ---: | ---: | ---: |
| Current unconditional | 47.43 | 74.80 | 1.390 | 3.843 |
| Truth-momentum ΔR <0.3 | 47.80 | 75.10 | 1.607 | 4.026 |
| Never merge | 47.58 | 74.59 | 1.732 | 4.199 |

The inclusive-target/HS-reference differences between rules are small compared
with pileup, and paired mean-error changes have confidence intervals containing
zero. This view does not provide a clean ranking of merge rules.

Oracle hard-scatter separation shows a smaller but statistically resolved
signal-momentum cost. The gate increases mean absolute HS MET magnitude error by
**0.217 [0.182, 0.252] GeV**; never merging increases it by
**0.342 [0.298, 0.386] GeV**. Mean HS vector-error increases are
**0.294 [0.257, 0.332] GeV** and **0.484 [0.437, 0.531] GeV**, respectively.
Oracle results use exact constituent truth identities and do not model a learned
pileup-removal algorithm or weighting by the merged fractional `ispu` label.

### Assessment

**Current unconditional merging preserves MET better; the angular gate improves
particle/jet locality at an explicit event-momentum cost.** Neither the jet
comparison nor the MET comparison alone is a sufficient target-quality test.

For a production target change, rejecting sources should not be treated as a
free correction. Splitting problematic clusters or representing rejected
particles separately would preserve their momentum while allowing local truth
objects; the present fixed one-target-per-input comparison does neither.
This study does not choose a production change or claim downstream model MET
performance. Production code remains unchanged.

## Definition and conservation

MET is the vector `M = -sum_i (px_i, py_i)` over **all selected particles**.
There are no jet pT cuts, jet matching requirements, particle η cuts or MET cuts
on the event distributions. The conventional MET magnitude is `|M|`.

Let `C` be the current target MET vector and `R` the sum of the original transverse
momenta of sources rejected by a gate. Then

```text
M_gate = C + R
M_gate - M_truth = (C - M_truth) + R
```

Merging all sources by full four-vector addition cannot change `C`, regardless
of how implausible an individual resulting target is. Its difference from the
pre-visibility truth reference comes from the truth visibility selection and
original allocator drops. A MET comparison is therefore complementary to the
particle and jet locality diagnostics.

The original allocator drops, unchanged by this comparison, carry a mean total
energy of 0.96 GeV/event and an RMS of 1.48 GeV/event. These differ from sources
newly rejected by a merge gate. All rejection accounting remains explicit.

## References and measurements

The three views follow the jet study:

- **Inclusive:** all target particles against measurable truth including pileup,
  corresponding to the truth selection behind `genmet_withpu`.
- **Hard-scatter reference:** all target particles against hard-scatter-only
  measurable truth, corresponding to `genmet`. This includes pileup contamination
  in the target and does not isolate the merge effect.
- **Oracle hard scatter:** retain only the exact original hard-scatter
  constituent contributions to accepted target groups and compare to hard-scatter
  truth. This is a truth diagnostic, not a practical pileup-removal algorithm.

The truth set is pre-visibility measurable truth, including the converter's
surrogate ancestors. It is not unrestricted generator-level neutrino MET.

Measurements include signed magnitude residual, absolute magnitude error,
x/y component biases and widths, full vector-error norm, and truth-parallel and
perpendicular residuals. Equal MET magnitudes can conceal a large vector error:
MET vectors `(30,40)` and `(0,50)` GeV both have magnitude 50 GeV but differ by
31.62 GeV in the transverse plane.

Response ratios and Δφ are evaluated only when truth MET ≥20 GeV to avoid unstable
ratios near zero. The signed parallel response is `M_target dot M_truth / |M_truth|²`.
All events remain in the absolute-error and magnitude distributions.
Paired uncertainty estimates resample whole events 3,000 times; no jet-based
event selection or matched-jet cohort is used.

## Exact-vector extraction and validation

The jet study saved exact MET magnitudes and rejected-vector norms, but its jet
collections stop at 3 GeV. Those truncated jets are not summed to approximate MET.
Instead, eight CPU workers outside the sandbox reconstruct truth and exclusive
matching from the same pinned source shards. A sparse largest-deposit host lookup
reproduces the allocator's merge lineage, with production-allocator fallback if a
float32 accumulation tie differs.

For every event and rule, the extracted vector magnitude, accepted merge count,
rejected-vector norm and target energy must reproduce the saved jet-study
accounting. Shard 0 additionally checks original host/source identities against
the earlier particle audit. Current merging also checks exact transverse-vector
conservation independently of grouping.

```bash
# Immediate magnitude plots from the completed jet-study accounting:
uv run python3 -m scripts.studies.plot_colliderml_merge_met --stored-magnitudes

# Exact x/y vectors and hard-scatter views:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python3 -m scripts.studies.colliderml_merge_met --workers 8

uv run python3 -m scripts.studies.plot_colliderml_merge_met
```

Outputs are in `experiments/colliderml_merge_met/`. Per-event components and
metrics are retained in `events.parquet`; summaries and paired bootstrap
intervals are in `met_summary.csv`, `bootstrap.csv` and `paired_comparison.csv`.
`met_mse_decomposition.csv` records the checked mean-square identity and
rejected-vector contribution for every rule/reference combination.
Plots include `met_distributions.png`, `met_target_vs_truth.png` and
`met_response_vs_truth.png`. Four new tests check directional errors, low-MET
ratio guards, merge-host tie-breaking and absence of alternate-host searches.
