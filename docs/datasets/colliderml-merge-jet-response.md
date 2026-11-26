# ColliderML merge rules: target-jet response

These measurements precede the |η|>3.5 truth-acceptance update. See the
[updated-branch jet/MET chain audit](colliderml-updated-chain.md) for the rerun
of current postprocessing on 1,000 PU0 and 1,000 PU200 events.

This extends the [particle-level merge-gate comparison](colliderml-merge-gates.md)
to jets. It compares current unconditional merging, an original-host truth-momentum
ΔR <0.3 merge gate, and never merging. Production target-building code is unchanged.

The [PU0/PU200 causal decomposition](colliderml-jet-spread-causes.md) subsequently
isolates visibility cuts, allocator drops and merge-induced constituent migration,
including a momentum-preserving unmerged control and explicit jet examples.

## Results: 1,100 distinct events

All eleven shards completed, with 100 aligned events each and event IDs 0–1099.
All reconstruction, energy-accounting and code-hash checks passed. The results
favor the momentum gate as a compromise: it improves inclusive jet matching and
response tails substantially, while never merging gives a larger downward bias.
Rejecting merges does incur a small, measurable loss of hard-scatter momentum.

Plots: [pT response distributions](../../experiments/colliderml_merge_jet_response/response_distributions.png),
[full-momentum response distributions](../../experiments/colliderml_merge_jet_response/momentum_response_distributions.png),
[forward pT response](../../experiments/colliderml_merge_jet_response/response_distributions_forward.png),
and [response versus truth pT](../../experiments/colliderml_merge_jet_response/response_vs_pt.png).

### Inclusive target jets versus inclusive truth jets

The primary central selection is truth pT ≥20 GeV, |η| <2.5 and matching
ΔR <0.1. There are **75,706 reference jets**. The following response statistics
use each rule's own angular matches; efficiencies include unmatched references.

| Rule | Matched /75,706 | Matching efficiency | Median pT response | σ68 of pT response | Mean absolute pT response error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current unconditional | 37,814 | 49.95% | 0.9443 | 0.07815 | 0.10719 |
| Momentum ΔR <0.3 | 52,942 | 69.93% | 0.9430 | 0.05635 | 0.08927 |
| Never merge | 53,759 | 71.01% | 0.9344 | 0.06158 | 0.09561 |

Event-bootstrap 95% intervals for matching efficiency are 49.49–50.43%,
69.45–70.43% and 70.51–71.47%, respectively. The angular gate's matching gain is
not obtained by throwing away unmatched reference jets: all three use the same
75,706 reference-jet denominator.

The **31,351 truth jets matched under all three rules** give the paired response
comparison below. Full momentum means `|p| = sqrt(px² + py² + pz²)`, not energy.

| Rule | Median pT response | σ68(pT response) | Median full-momentum response | σ68(full-momentum response) |
| --- | ---: | ---: | ---: | ---: |
| Current unconditional | 0.9467 | 0.06739 | 0.9499 | 0.08324 |
| Momentum ΔR <0.3 | 0.9405 | 0.04762 | 0.9365 | 0.04885 |
| Never merge | 0.9306 | 0.05379 | 0.9276 | 0.05409 |

On this common cohort, the gate narrows the pT width by **29.3%** and the full
momentum width by **41.3%**. Its pT median shifts downward by 0.0062, and its
full-momentum median by 0.0134. Thus it improves the spread and tails rather
than making every jet unbiased. The mean absolute pT response error changes by
**−0.01117**, with paired event-bootstrap 95% interval **[−0.01232, −0.01008]**.
Only 45.4% of individual common jets improve in absolute error; many others lose
a little energy, while the reduction of large errors improves the overall mean.

Never merging changes the same mean absolute error by −0.00240, a much smaller
improvement. For reference pT ≥50 GeV, the paired changes are **−0.00508** for
the momentum gate and **+0.00427** for never merging; their 95% intervals exclude
zero. The never-merge downward bias outweighs its tail improvement for these
harder common jets.

The candidate side also improves. Among central target jets with pT ≥20 GeV,
50.56% of current candidates match a fiducial inclusive truth jet, compared with
68.93% for the gate and 70.44% for never merging. This is threshold-dependent
candidate-side matching, not a detector-independent fake-jet classification.

### Forward jets and matching sensitivity

For 2.5 ≤ |η| <4.5, pT ≥20 GeV, there are **31,945 inclusive truth jets**:

| Rule | Matching efficiency | Median pT response of own matches | σ68 of pT response | Fraction of matches with response <0.5 |
| --- | ---: | ---: | ---: | ---: |
| Current unconditional | 21.91% | 0.9003 | 0.29594 | 15.55% |
| Momentum ΔR <0.3 | 52.03% | 0.9295 | 0.11207 | 3.01% |
| Never merge | 58.31% | 0.8986 | 0.13085 | 2.79% |

The improvement is larger forward, where the earlier particle audit found the
most problematic broad clusters. On the **3,871 common forward matches**, the
pT widths are 0.23259 /0.08571 /0.10314 for current /gate /never merge. The result
therefore persists after controlling the matched truth-jet population.

The conclusion also survives a wider **ΔR <0.2 jet-matching cut**. Central
matching efficiencies become 74.84% /86.79% /87.37%, while the own-match pT widths
are 0.12745 /0.07766 /0.08102. The tighter primary cut is not solely responsible
for the gate's advantage.

### Hard-scatter truth and the energy cost

Comparing the **inclusive** target jets directly with hard-scatter-only truth
jets gives large positive responses under all rules because target jets still
contain PU200. For central truth jets above 20 GeV, the medians of own matches
are **2.046 /2.097 /2.080** for current /gate /never merge. The changing matched
cohort matters: on common hard-scatter references, the gate slightly lowers the
response. These distributions are useful as the stored `targetjet/genjet`
comparison, but they do not isolate the merging error from pileup contamination.

The oracle hard-scatter comparison isolates signal retention. For **6,614 central
hard-scatter truth jets** above 20 GeV:

| Rule | Matched | Mean pT response | Mean absolute pT response error |
| --- | ---: | ---: | ---: |
| Current unconditional | 6,613 | 0.998804 | 0.001437 |
| Momentum ΔR <0.3 | 6,611 | 0.997767 | 0.002544 |
| Never merge | 6,611 | 0.997201 | 0.003148 |

The central hard-scatter matching remains ≥99.95%, but dropping merges loses
some signal momentum. The gate's paired increase in mean absolute response
error is **+0.001108 [0.000987, 0.001240]**; the never-merge increase is
**+0.001713 [0.001573, 0.001866]**. Oracle separation removes pileup using truth
constituent identities, so these small errors do not describe a practical
pileup-mitigation algorithm.

Over the full sample, the momentum gate rejects **3.600% of total represented
target energy** and **0.942% of hard-scatter target energy**. Never merging
rejects **5.256%** and **1.437%**, respectively. This is consistent with the
gate preserving more useful energy than never merging, even though both improve
the geometry of the retained inclusive target set.

### Assessment

The follow-up [event-level MET comparison](colliderml-merge-met.md) checks the
complementary momentum-budget consequence: current four-vector merging preserves
the global sum, while the rejection rules change it and worsen inclusive MET
magnitude agreement despite improving the jet-locality measures.
On the same 1,100 events, the momentum gate raises inclusive MET vector RMS error
from 16.72 to 20.25 GeV; never merging gives 20.24 GeV. This event-level cost
qualifies the preference below, which concerns the particle/jet target prototype.

For this sample, **momentum ΔR <0.3 is preferable to never merging as a controlled
target-definition prototype**: it removes the previously identified pathological
particle direction shifts, improves inclusive jet matching, and gives narrower
jet response distributions with less negative bias than never merging. Its
benefit is especially large forward. The small signal-momentum loss is real and
should remain explicitly accounted for.

This is a target-construction comparison, not evidence of improved trained-model
performance. The threshold was motivated on shard 0 and the study includes that
shard; shards 1–10 extend the sample but are not presented as a separately
predeclared validation analysis. A production change should remain selectable,
retain rejection accounting, and be followed by training and downstream jet/MET
checks. Existing problematic original truth hosts are not fixed by a merge gate.

## Dataset and processing

The source is ttbar PU200, shards 00000–00010, pinned to ColliderML Release 1
revision `64c3d2f112df3d5d20979d22da7cfdff13e10c4b`. All four tables of each shard
are downloaded from that revision and checked against its Hugging Face LFS
SHA256 object IDs. The original local shard 00000 is reused after checksum
verification. The source occupies about 38 GiB.

Eight CPU worker processes run outside the sandbox, each processing one shard
at a time. Numerical libraries use one thread per worker. The reader checks
event alignment across particles, tracks, calorimeter hits and tracker hits.
The run checks that physical event IDs are unique across shards; an analysis
summary requires at least 1,000 completed distinct events.

CLUE with the PU200 preset, calibrated deposits, current visible truth selection
and `majority_energy` exclusive input assignment are identical under all three
merge rules. Rejected merge sources are dropped with explicit energy accounting;
no rule searches for an alternate host or changes the exclusive assignment.
Each accepted source is compared against the **original** host, preventing
transitive angular chaining.

The current rule is reconstructed from the actual shared allocator's merge
lineage and checked against its production output within float32 tolerance.
Accepted plus rejected energy is checked for every rule and event. The never-merge
rule retains only the particles that own exclusive track/cluster inputs; it does
not promote unassigned particles to extra targets.

## Truth references and pileup

Truth jets use the converter's **pre-visibility measurable truth**, including
surrogate ancestors, before allocation and merging. This is independent of the
choice of merge rule and matches the truth convention used by `genjet` and
`genjet_withpu`; it is not an unrestricted all-generator-particle reference.

Three comparisons separate different effects:

1. **Inclusive:** all target particles versus measurable truth including pileup.
   This is the primary comparison for isolating target construction at PU200.
2. **Hard-scatter reference:** all target particles versus hard-scatter-only
   measurable truth. This corresponds to the converter's stored `targetjet`
   versus `genjet` convention. Its response includes pileup contamination and
   must not be interpreted as a pure merging effect.
3. **Oracle hard scatter:** before jet clustering, retain only the exact original
   hard-scatter constituent four-vectors inside each accepted target group, then
   compare to hard-scatter truth jets. This uses truth knowledge; it is a diagnostic
   for signal retention, not a practical pileup-removal algorithm. It preserves
   hard-scatter contributions inside mixed pileup/hard-scatter targets rather
   than discarding targets using a cut on their fractional `ispu` label.

## Jets, matching and statistical populations

All jets use anti-kt R=0.4 with E-scheme recombination. Target and truth collections
are stored down to pT =3 GeV, matching the converter's convention. Reference jets
are selected at pT ≥20 GeV and |η| <4.5 before one-to-one matching. Matching
maximizes cardinality and then minimizes angular distance using the repository's
`_match_jets_event`. The primary match cut is ΔR <0.1; ΔR <0.2 is a sensitivity
check. No target/truth response cut is used to select an angular match.

Results are split into central |η| <2.5 and forward 2.5 ≤ |η| <4.5 regions, with
inclusive reference thresholds 20/50/100 GeV and bins 20–30, 30–50, 50–100,
100–200 and ≥200 GeV. Jet pT response is `pT(target)/pT(truth)`. Full momentum
`|p|` and energy ratios are also saved and summarized.

Unmatched reference jets remain explicit rows in the output and in matching
efficiency denominators. Response distributions are shown both for each rule's
own matches and for a **common truth-jet cohort matched by all three rules**.
The latter provides a paired comparison; the former preserves the actual rule's
response and efficiency. Neither is sufficient by itself: a narrow distribution
among surviving matches can conceal lost jets.

The robust width is half the central 68% response interval,
`sigma68 = (q84 - q16)/2`. Tail fractions are responses below 0.5 and above 1.5,
conditional on angular matching. Response-qualified efficiency additionally
requires 0.5 <response <1.5 and uses all selected reference jets as denominator.
Histogram densities are normalized over all matches, including any values
outside the plotted range.

Uncertainty estimates resample whole events 2,000 times, preserving jet
correlations and pairing between the three rules. Paired changes in absolute
response error use the common matched truth-jet population. Confidence intervals
describe event sampling in these shards, not independent validation of a threshold
or detector systematic uncertainty.

## Reproduction and artifacts

```bash
uv run python3 -m scripts.studies.download_colliderml_jet_sample \
  --shards 11 --workers 4

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python3 -m scripts.studies.colliderml_merge_jet_response \
    --shards 11 --workers 8 --wait-for-input

uv run python3 -m scripts.studies.plot_colliderml_merge_jet_response
```

The source defaults to `/tmp/colliderml-jet-response-source`, and output to
`experiments/colliderml_merge_jet_response`. `inputs.json` records the pinned
manifest, full source checksums, worker count, definitions and processing code
hash. Each shard writes `jets.parquet`, `matches.parquet`, `accounting.parquet`
and `done.json`; completed shards with the same processing code hash are reused.

Summary artifacts include `run_summary.json`, `response_summary.csv`,
`bootstrap.csv`, `paired_comparison.csv`, `candidate_matching.csv`,
`response_distributions.png`, `response_vs_pt.png` and
`momentum_energy_response.png`. Additional plots are
`response_distributions_forward.png`, `momentum_response_distributions.png`,
`energy_response_distributions.png`, and `response_error_tails.png`.
The matching table retains each reference jet's
event ID and jet index under every rule, so changes can be traced back to events.
`event_inventory.json` and `validation.json` record corpus alignment, distinct
event counts, processing consistency and relevant source-code hashes. Five new
physics-bookkeeping and matching-denominator tests passed; lint, formatting,
whitespace checks and visual inspection of the response plots also passed.
